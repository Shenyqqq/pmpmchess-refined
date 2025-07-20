import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 非常重要，负责CUDA初始化
import numpy as np
import os

# TRT日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class HostDeviceMem(object):
    """一个简单的数据结构，用于同时持有主机和设备的内存指针。"""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TensorRTWrapper:
    """
    一个包装器，用于加载TensorRT引擎并执行推理。
    它的 'predict_batch' 方法可以作为NNetWrapper中同名方法的替代品。
    """

    def __init__(self, game, args, engine_path):
        self.game = game
        self.args = args
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine file not found at: {engine_path}")

        print(f"Loading TensorRT engine from: {engine_path}")
        # 加载TensorRT引擎
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 分配输入/输出缓冲区
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        """在主机（CPU）和设备（GPU）上为输入和输出分配内存。"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            # 获取绑定（输入/输出）的尺寸和数据类型
            # 注意：我们使用engine.max_batch_size来分配最大可能的缓冲区
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 在GPU上分配内存
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)

            # 将设备内存指针添加到绑定列表中
            bindings.append(int(device_mem))

            # 在CPU上分配锁页内存（用于异步传输）
            host_mem = cuda.pagelocked_empty(size, dtype)

            mem = HostDeviceMem(host_mem, device_mem)
            if self.engine.binding_is_input(binding):
                inputs.append(mem)
            else:
                outputs.append(mem)

        return inputs, outputs, bindings, stream

    def predict_batch(self, boards):
        """
        使用加载的TensorRT引擎执行批量推理。

        Args:
            boards (list of np.array): 代表棋盘状态的Numpy数组列表。

        Returns:
            list of np.array: 包含策略、价值等预测结果的Numpy数组列表。
        """
        batch_size = len(boards)

        # 如果当前批次大小超过引擎的最大批次大小，则抛出错误
        if batch_size > self.engine.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds engine's max batch size {self.engine.max_batch_size}")

        # 将输入数据展平并复制到锁页内存中
        # .ravel() 将多维数组展平为一维，这是必需的
        np.copyto(self.inputs[0].host, np.array(boards, dtype=np.float32).ravel())

        # 异步地将输入数据从CPU(host)复制到GPU(device)
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        # 异步地执行推理
        # 对于动态尺寸，需要设置实际的binding形状
        self.context.set_binding_shape(0, (batch_size, self.args.num_channels, self.board_x, self.board_y))
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 异步地将输出数据从GPU复制回CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # 同步流，确保所有异步操作都已完成
        self.stream.synchronize()

        # 从展平的输出缓冲区中恢复数据并重塑
        # 顺序必须与模型forward返回和ONNX导出的顺序完全一致
        h_outputs = [out.host for out in self.outputs]

        # 计算每个输出的正确尺寸
        policy_size = self.action_size
        value_size = 1
        score_size = 1
        score_var_size = 1
        ownership_size = self.board_x * self.board_y

        # 按照ONNX导出时的顺序切分和重塑输出
        # 注意：h_outputs[i] 是一维数组，我们需要根据batch_size来切片
        policy_out = h_outputs[0][:batch_size * policy_size].reshape(batch_size, policy_size)
        win_out = h_outputs[1][:batch_size * value_size].reshape(batch_size, value_size)
        score_out = h_outputs[2][:batch_size * score_size].reshape(batch_size, score_size)
        score_var_out = h_outputs[3][:batch_size * score_var_size].reshape(batch_size, score_var_size)
        ownership_out = h_outputs[4][:batch_size * ownership_size].reshape(batch_size, 1, self.board_x, self.board_y)

        return [policy_out, win_out, score_out, score_var_out, ownership_out]

