import torch
from nn_model import NNet


def convert_to_onnx(game, args, checkpoint_path, output_path):
    """
    加载一个PyTorch检查点，并将模型转换为ONNX格式。

    Args:
        game: 游戏实例，用于获取棋盘尺寸。
        args: 包含模型配置（如通道数）的参数对象。
        checkpoint_path (str): PyTorch模型检查点文件（.pth.tar）的路径。
        output_path (str): 输出的ONNX文件的路径。
    """
    print(f"Loading PyTorch model from: {checkpoint_path}")

    # 实例化PyTorch模型
    model = NNet(game, args)

    # 从检查点加载权重。确保映射到CPU以避免设备问题。
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # 设置为评估模式

    # 获取棋盘尺寸和通道数
    board_x, board_y = game.getBoardSize()
    num_channels = args.num_channels

    # 创建一个符合模型输入的虚拟输入张量。
    # 这是导出ONNX所必需的。
    dummy_input = torch.randn(1, num_channels, board_x, board_y, requires_grad=False)

    print(f"Exporting model to ONNX format at: {output_path}")

    # 定义输入输出的名称，以及动态轴（允许变化的batch_size）
    input_names = ['input']
    output_names = ['policy_out', 'win_out', 'score_out', 'score_var_out', 'ownership_out']
    dynamic_axes = {name: {0: 'batch_size'} for name in input_names + output_names}

    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,  # 导出训练好的参数
        opset_version=12,  # ONNX算子集版本
        do_constant_folding=True,  # 执行常量折叠优化
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    print("ONNX export complete.")
