import os
from collections import deque
from pickle import Pickler, Unpickler
import numpy as np
from tqdm import tqdm
from Arena import Arena
import sys
import katago_cpp_core

import subprocess
from onnx_converter import convert_to_onnx
from trt_wrapper import TensorRTWrapper


class Coach:
    """
    This class executes the self-play + learning loop.
    """

    def __init__(self, game, nnet, args, writer):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.writer = writer
        self.pnet = self.nnet.__class__(self.game, self.nnet.nnet_class, self.args)
        self.initial_net = self.nnet.__class__(self.game, self.nnet.nnet_class, self.args)
        self.train_examples_history = []
        self.skip_first_self_play = False
        self.inference_net = None

    def _get_inference_net(self):
        """
        获取用于推理的网络。优先加载TensorRT引擎。
        如果引擎不存在或加载失败，则回退到使用PyTorch模型。
        """
        if self.inference_net is not None:
            return self.inference_net

        engine_path = os.path.join(self.args['checkpoint'], 'best.plan')
        if os.path.exists(engine_path):
            try:
                print("Loading existing TensorRT engine for self-play...")
                self.inference_net = TensorRTWrapper(self.game, self.nnet.args, engine_path)
                print("TensorRT engine loaded successfully.")
            except Exception as e:
                print(f"!!! Failed to load TensorRT engine: {e}", file=sys.stderr)
                print("!!! Falling back to PyTorch model for self-play.", file=sys.stderr)
                self.inference_net = self.nnet
        else:
            print("TensorRT engine not found. Using PyTorch model for self-play.")
            self.inference_net = self.nnet

        return self.inference_net

    def _build_and_load_trt_engine(self, checkpoint_path, engine_path):
        """
        将指定的PyTorch模型检查点转换为ONNX，然后构建并加载TensorRT引擎。

        Args:
            checkpoint_path (str): 源PyTorch模型检查点文件(.pth.tar)的路径。
            engine_path (str): 目标TensorRT引擎文件(.plan)的路径。
        """
        print("\n" + "─" * 50)
        print(f"Building TensorRT Engine from: {checkpoint_path}")

        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file for conversion not found: {checkpoint_path}")

            # ONNX文件可以作为临时文件生成
            onnx_path = os.path.splitext(engine_path)[0] + ".onnx"

            # 步骤 1: 将PyTorch模型转换为ONNX
            print(f"Step 1/3: Converting PyTorch model to ONNX at {onnx_path}...")
            convert_to_onnx(self.game, self.nnet.args, checkpoint_path, onnx_path)

            # 为trtexec获取优化的批处理大小
            opt_batch_size = self.args.get('numParallelGames', 8)
            max_batch_size = self.args.get('batch_size', 64)

            # 步骤 2: 使用trtexec构建TensorRT引擎
            print(f"Step 2/3: Building TensorRT engine and saving to {engine_path}...")
            build_command = (
                f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16 "
                f"--minShapes=input:1x13x9x9 "
                f"--optShapes=input:{opt_batch_size}x13x9x9 "
                f"--maxShapes=input:{max_batch_size}x13x9x9"
            )
            print(f"Executing: {build_command}")
            # 执行命令，并将输出打印到控制台
            subprocess.run(build_command, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)

            # 步骤 3: 加载新的TensorRT引擎以供下次迭代使用
            print(f"Step 3/3: Loading new TensorRT engine from {engine_path}...")
            self.inference_net = TensorRTWrapper(self.game, self.nnet.args, engine_path)
            print("─" * 50)
            print("TensorRT Engine built and loaded successfully!")
            print("─" * 50 + "\n")


        except Exception as e:
            print(f"!!! TensorRT engine build failed: {e}", file=sys.stderr)
            print("!!! Falling back to PyTorch model for self-play.", file=sys.stderr)
            # 如果构建失败，确保推理网络回退到普通的PyTorch网络
            if self.inference_net is None:
                self.inference_net = self.nnet

    def _print_board_debug(self, board, round_num, player, action):
        """Helper function to print the board state for debugging during self-play."""
        # (This function remains unchanged)
        print("\n" + "=" * 30, file=sys.stderr)
        print(f"[SELF-PLAY DEBUG] Round: {round_num}, Player {player} took action {action}", file=sys.stderr)
        print("--- Black Stones (X) ---", file=sys.stderr)
        p_b = np.where(board[0] == 1, 'X', '.')
        print(p_b, file=sys.stderr)
        print("--- White Stones (O) ---", file=sys.stderr)
        p_w = np.where(board[1] == 1, 'O', '.')
        print(p_w, file=sys.stderr)
        print("--- Black Control (+) ---", file=sys.stderr)
        c_b = np.where(board[2] == 1, 'x', '.')
        print(c_b, file=sys.stderr)
        print("--- White Control (-) ---", file=sys.stderr)
        c_w = np.where(board[3] == 1, 'o', '.')
        print(c_w, file=sys.stderr)
        print("=" * 30 + "\n", file=sys.stderr)
        sys.stderr.flush()

    def execute_episode_batch(self):
        """
        Performs one batch of self-play episodes.
        For each move, it randomly decides whether to use a "fast" or "deep" MCTS search.
        Only data from "deep" searches is collected for training.
        """
        num_games = self.args.get('numParallelGames', 8)
        initial_results = [self.game.getInitialBoard() for _ in range(num_games)]
        boards = [np.array(res[0], dtype=np.float32).reshape(13, 9, 9) for res in initial_results]
        hashes = [res[1] for res in initial_results]
        current_players = [1] * num_games
        dones = np.array([False] * num_games)

        deep_move_histories = [[] for _ in range(num_games)]

        pbar = tqdm(total=self.args.max_rounds, desc="Self-Play Batch", leave=False)
        rounds_played = 0

        inference_net = self._get_inference_net()

        while not dones.all():
            rounds_played += 1
            if rounds_played > self.args.max_rounds: break

            active_indices = np.where(dones == False)[0]
            if len(active_indices) == 0: break

            is_deep_simulation = np.random.rand() < self.args.get('deepProb', 0.25)
            if is_deep_simulation:
                num_sims = self.args.get('deepNumMCTSSims', 200)
            else:
                num_sims = self.args.get('fastNumMCTSSims', 50)

            canonical_boards, canonical_hashes = [], []
            for i in active_indices:
                c_board, c_hash = self.game.getCanonicalForm(boards[i], hashes[i], current_players[i])
                canonical_boards.append(np.array(c_board, dtype=np.float32).reshape(13, 9, 9))
                canonical_hashes.append(c_hash)

            temp = 1 if rounds_played < self.args.get('tempThreshold', 15) else 0
            mcts_args = katago_cpp_core.MCTSArgs()
            mcts_args.numMCTSSims = num_sims
            mcts_args.cpuct, mcts_args.dirichletAlpha, mcts_args.epsilon, mcts_args.factor_winloss = self.args.cpuct, self.args.dirichletAlpha, self.args.epsilon, self.args.factor_winloss
            mcts = katago_cpp_core.MCTS(self.game, inference_net.predict_batch, mcts_args)

            seeds = np.random.randint(0, 2 ** 32 - 1, size=len(canonical_boards), dtype=np.uint32)
            all_pis = mcts.getActionProbs(canonical_boards, canonical_hashes, seeds.tolist(), temp=temp)

            for i, pi in enumerate(all_pis):
                active_idx = active_indices[i]

                if is_deep_simulation:
                    deep_move_histories[active_idx].append([canonical_boards[i], current_players[active_idx], pi])

                action = np.random.choice(len(pi), p=pi)
                next_board, next_player, next_hash = self.game.getNextState(boards[active_idx],
                                                                            current_players[active_idx], action)
                boards[active_idx] = np.array(next_board, dtype=np.float32).reshape(13, 9, 9)
                current_players[active_idx], hashes[active_idx] = next_player, next_hash

                if self.game.getGameEnded(boards[active_idx], 1) != 0:
                    dones[active_idx] = True

            pbar.update(1)
        pbar.close()

        all_train_examples = []
        for i in range(num_games):
            if not deep_move_histories[i]:
                continue

            game_result_win_loss, game_result_score = self.game.getGameEnded(boards[i], 1), self.game.getScore(
                boards[i], 1)
            score_dist_target = np.zeros(self.nnet.max_score * 2 + 1, dtype=np.float32)
            score_idx = int(round(game_result_score)) + self.nnet.max_score
            if 0 <= score_idx < len(score_dist_target): score_dist_target[score_idx] = 1.0

            for canonical_board, player, pi in deep_move_histories[i]:
                symmetries = self.game.getSymmetries(canonical_board, pi)
                final_ownership_map = (boards[i][2, :, :] - boards[i][3, :, :]) * player
                ownership_symmetries = []
                temp_map = final_ownership_map
                for _ in range(4):
                    ownership_symmetries.append(temp_map)
                    ownership_symmetries.append(np.fliplr(temp_map))
                    temp_map = np.rot90(temp_map, k=-1)

                for j in range(len(symmetries)):
                    sym_board_list, sym_pi_list = symmetries[j]
                    sym_board = np.array(sym_board_list, dtype=np.float32).reshape(13, 9, 9)
                    sym_pi = np.array(sym_pi_list, dtype=np.float32)
                    sym_ownership = ownership_symmetries[j]
                    win_loss_target = game_result_win_loss * player
                    all_train_examples.append((sym_board, sym_pi, win_loss_target, score_dist_target, sym_ownership))

        return all_train_examples

    def learn(self):
        """
        主学习循环。
        """
        print("Saving initial model to be used as a baseline...")
        self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='initial.pth.tar')
        self.initial_net.load_checkpoint(folder=self.args['checkpoint'], filename='initial.pth.tar')
        print("Initial model saved and loaded.")
        sys.stdout.flush()

        source_folder, source_file = self.args.get('load_folder_file', (None, None))
        if source_folder and source_file:
            source_checkpoint_path = os.path.join(source_folder, source_file)
            # 引擎总是保存在当前迭代的checkpoint目录下，名为best.plan
            target_engine_path = os.path.join(self.args['checkpoint'], 'best.plan')

            # 如果源模型存在，但目标引擎不存在，则立即构建
            if os.path.exists(source_checkpoint_path) and not os.path.exists(target_engine_path):
                print("─" * 50)
                print(f"Found starting checkpoint '{source_checkpoint_path}' without a pre-built TensorRT engine.")
                print("Building engine before the first iteration...")
                # convert_to_onnx会从给定的路径加载模型，所以我们不需要手动加载到self.nnet
                self._build_and_load_trt_engine(source_checkpoint_path, target_engine_path)
                print("─" * 50)

        for i in range(1, self.args.get('numIters', 100) + 1):
            print(f'------ ITERATION {i} ------')
            sys.stdout.flush()

            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque(maxlen=self.args.get('maxlenOfQueue', 200000))

                eps_tqdm = tqdm(range(self.args.get('numEps', 1)), desc="Collecting Self-Play Episodes")
                for _ in eps_tqdm:
                    iteration_train_examples.extend(self.execute_episode_batch())

                print(f"\nFinished self-play. Collected {len(iteration_train_examples)} new training examples.")
                sys.stdout.flush()

                self.train_examples_history.append(iteration_train_examples)
                if len(self.train_examples_history) > self.args.get('numItersForTrainExamplesHistory', 20):
                    self.train_examples_history.pop(0)

                # --- 修改/新增：在保存前，对 self.train_examples_history 本身进行总样本量截断 ---
                max_samples = self.args.get('maxTrainExamples', None)
                if max_samples:
                    current_total_samples = sum(len(x) for x in self.train_examples_history)
                    if current_total_samples > max_samples:
                        print(
                            f"Total examples in history ({current_total_samples}) exceed max_samples ({max_samples}). Truncating history...")

                        samples_to_remove = current_total_samples - max_samples

                        # 从最老的迭代数据开始移除，直到满足数量要求
                        while samples_to_remove > 0 and self.train_examples_history:
                            oldest_deque = self.train_examples_history[0]
                            if len(oldest_deque) <= samples_to_remove:
                                # 如果最老的迭代窗口所有样本都不够删，则整个移除
                                samples_to_remove -= len(oldest_deque)
                                self.train_examples_history.pop(0)
                            else:
                                # 否则，从最老的迭代窗口中移除所需数量的样本
                                for _ in range(samples_to_remove):
                                    oldest_deque.popleft()
                                samples_to_remove = 0  # 已经移除完毕

                        new_total_samples = sum(len(x) for x in self.train_examples_history)
                        print(f"History truncated. New total examples: {new_total_samples}")
                        sys.stdout.flush()
                # --- 修改/新增结束 ---

                # 在截断历史数据之后再保存
                self.save_train_examples(i - 1)

            # 从可能已被截断的历史记录中加载所有样本
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)

            # 由于 self.train_examples_history 本身已经被截断，下面的代码块不再需要
            # if max_samples and len(train_examples) > max_samples: ...

            if not train_examples:
                print("No training examples available. Skipping training for this iteration.")
                continue

            np.random.shuffle(train_examples)

            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')

            self.nnet.train(train_examples, i, self.writer)

            print('PITTING AGAINST PREVIOUS VERSION')
            mcts_args = katago_cpp_core.MCTSArgs()
            mcts_args.numMCTSSims = self.args.get('arenaNumMCTSSims', self.args.get('deepNumMCTSSims', 200))
            mcts_args.cpuct, mcts_args.dirichletAlpha, mcts_args.epsilon = self.args.cpuct, 0, 0
            nnet_mcts = katago_cpp_core.MCTS(self.game, self.nnet.predict_batch, mcts_args)
            pnet_mcts = katago_cpp_core.MCTS(self.game, self.pnet.predict_batch, mcts_args)
            arena = Arena(nnet_mcts, pnet_mcts, self.game, self.args)
            nwins, pwins, draws,_,_ = arena.play_games_batch(self.args.get('arenaCompare', 40))

            self.writer.add_scalar('Arena/NewNetWins', nwins, i)
            self.writer.add_scalar('Arena/PrevNetWins', pwins, i)
            self.writer.add_scalar('Arena/Draws', draws, i)
            if (pwins + nwins) > 0:
                win_rate = float(nwins) / (pwins + nwins)
                self.writer.add_scalar('Arena/WinRate', win_rate, i)

            print(f'NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws}')

            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.get('updateThreshold', 0.6):
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='best.pth.tar')
                best_ckpt_path = os.path.join(self.args['checkpoint'], 'best.pth.tar')
                engine_path = os.path.join(self.args['checkpoint'], 'best.plan')
                self._build_and_load_trt_engine(best_ckpt_path, engine_path)

    def get_checkpoint_file(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args['checkpoint']
        if not os.path.exists(folder): os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f: Pickler(f).dump(self.train_examples_history)

    def load_separate_train_examples(self, filepath):
        if not os.path.isfile(filepath):
            print(f'File "{filepath}" with train examples not found!')
            return False

        print(f"Loading training examples from a separate file: {filepath}")
        try:
            with open(filepath, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            self.skip_first_self_play = True
            print("Training examples loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading training examples: {e}")
            return False

    def load_train_examples(self):
        model_file = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(f'File "{examples_file}" with train examples not found!')
            return False

        print(f"Loading training examples from file associated with model: {examples_file}")
        with open(examples_file, "rb") as f:
            self.train_examples_history = Unpickler(f).load()

        self.skip_first_self_play = False
        print("Training examples loaded successfully.")
        return True