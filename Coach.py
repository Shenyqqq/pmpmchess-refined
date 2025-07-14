import os
from collections import deque
from pickle import Pickler, Unpickler
import numpy as np
from tqdm import tqdm
from Arena import Arena
import sys
import katago_cpp_core


class Coach:
    """
    This class executes the self-play + learning loop.
    """

    def __init__(self, game, nnet, args, writer):  # --- 修改: 接收 writer ---
        self.game = game
        self.nnet = nnet
        self.args = args
        self.writer = writer  # --- 新增: 保存 writer ---
        self.pnet = self.nnet.__class__(self.game, self.nnet.nnet_class, self.args)
        self.train_examples_history = []
        self.skip_first_self_play = False

    def _print_board_debug(self, board, round_num, player, action):
        """Helper function to print the board state for debugging during self-play."""
        print("\n" + "=" * 30, file=sys.stderr)
        print(f"[SELF-PLAY DEBUG] Round: {round_num}, Player {player} took action {action}", file=sys.stderr)

        # Channel 0: Black Stones (X)
        print("--- Black Stones (X) ---", file=sys.stderr)
        p_b = np.where(board[0] == 1, 'X', '.')
        print(p_b, file=sys.stderr)

        # Channel 1: White Stones (O)
        print("--- White Stones (O) ---", file=sys.stderr)
        p_w = np.where(board[1] == 1, 'O', '.')
        print(p_w, file=sys.stderr)

        # Channel 2: Black Control (+)
        print("--- Black Control (+) ---", file=sys.stderr)
        c_b = np.where(board[2] == 1, 'x', '.')
        print(c_b, file=sys.stderr)

        # Channel 3: White Control (-)
        print("--- White Control (-) ---", file=sys.stderr)
        c_w = np.where(board[3] == 1, 'o', '.')
        print(c_w, file=sys.stderr)
        print("=" * 30 + "\n", file=sys.stderr)
        sys.stderr.flush()

    def execute_episode_batch(self):
        """
        Performs one batch of self-play episodes, generating training data.
        """
        num_games = self.args.get('numParallelGames', 8)
        initial_results = [self.game.getInitialBoard() for _ in range(num_games)]
        boards = [np.array(res[0], dtype=np.float32).reshape(13, 9, 9) for res in initial_results]
        hashes = [res[1] for res in initial_results]
        current_players = [1] * num_games
        dones = np.array([False] * num_games)
        episode_histories = [[] for _ in range(num_games)]
        pbar = tqdm(total=self.args.max_rounds, desc="Batch Self-Play", leave=False)
        rounds_played = 0
        while not dones.all():
            rounds_played += 1
            if rounds_played > self.args.max_rounds: break
            active_indices = np.where(dones == False)[0]
            if len(active_indices) == 0: break
            canonical_boards, canonical_hashes = [], []
            for i in active_indices:
                c_board, c_hash = self.game.getCanonicalForm(boards[i], hashes[i], current_players[i])
                canonical_boards.append(np.array(c_board, dtype=np.float32).reshape(13, 9, 9))
                canonical_hashes.append(c_hash)
            temp = 1 if rounds_played < self.args.get('tempThreshold', 15) else 0
            mcts_args = katago_cpp_core.MCTSArgs()
            min_sims = self.args.get('min_numSims', self.args.numMCTSSims)
            mcts_args.numMCTSSims = np.random.randint(min_sims, self.args.numMCTSSims + 1)
            mcts_args.cpuct, mcts_args.dirichletAlpha, mcts_args.epsilon, mcts_args.factor_winloss = self.args.cpuct, self.args.dirichletAlpha, self.args.epsilon, self.args.factor_winloss
            mcts = katago_cpp_core.MCTS(self.game, self.nnet.predict_batch, mcts_args)

            seeds = np.random.randint(0, 2 ** 32 - 1, size=len(canonical_boards), dtype=np.uint32)
            all_pis = mcts.getActionProbs(canonical_boards, canonical_hashes, seeds.tolist(), temp=temp)
            for i, pi in enumerate(all_pis):
                # active_idx = active_indices[i]
                # if active_idx == 0:
                #     self._print_board_debug(boards[active_idx], rounds_played, current_players[active_idx],
                #                             -1)  # Print state before action

                active_idx = active_indices[i]
                episode_histories[active_idx].append([canonical_boards[i], current_players[active_idx], pi])
                action = np.random.choice(len(pi), p=pi)
                next_board, next_player, next_hash = self.game.getNextState(boards[active_idx],
                                                                            current_players[active_idx], action)
                boards[active_idx] = np.array(next_board, dtype=np.float32).reshape(13, 9, 9)
                current_players[active_idx], hashes[active_idx] = next_player, next_hash
                # ==================== DEBUG BLOCK START ====================
                # 只打印第一个活动游戏(game 0)的调试信息，以避免信息刷屏
                # if active_idx == 0:
                #     print("\n\n--- DEBUG: BOARD STATE AFTER ACTION ---")
                #     print(f"Game Index (active_idx): {active_idx}")
                #     print(f"Round: {rounds_played}")
                #     print(f"Action Taken: {action}")
                #     print(f"Next Player: {next_player}")
                #     print("Next Board State (13 channels):")
                #
                #     # 获取已经reshape过的棋盘
                #     reshaped_board_to_print = boards[active_idx]
                #
                #     # 遍历并打印13个通道
                #     for channel_index in range(reshaped_board_to_print.shape[0]):
                #         print(f"--- Channel {channel_index} ---")
                #         print(reshaped_board_to_print[channel_index].astype(int))  # astype(int) for cleaner printing
                #
                #     print("--- DEBUG: END ---\n")
                # ===================== DEBUG BLOCK END =====================
                if self.game.getGameEnded(boards[active_idx], 1) != 0:
                    dones[active_idx] = True

            pbar.update(1)
        pbar.close()

        # DEBUG
        # for i in range(0,num_games):
        #     print("\n--- Final Self-Play Board State of Game 0 ---", file=sys.stderr)
        #     self._print_board_debug(boards[i], rounds_played, -1, -1)

        all_train_examples = []
        for i in range(num_games):
            game_result_win_loss, game_result_score = self.game.getGameEnded(boards[i], 1), self.game.getScore(
                boards[i], 1)
            score_dist_target = np.zeros(self.nnet.max_score * 2 + 1, dtype=np.float32)
            score_idx = int(round(game_result_score)) + self.nnet.max_score
            if 0 <= score_idx < len(score_dist_target): score_dist_target[score_idx] = 1.0
            for canonical_board, player, pi in episode_histories[i]:
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
        for i in range(1, self.args.get('numIters', 100) + 1):
            print(f'------ ITERATION {i} ------')
            sys.stdout.flush()
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque(maxlen=self.args.get('maxlenOfQueue', 200000))
                eps_tqdm = tqdm(range(self.args.get('numEps', 1)), desc="Collecting Self-Play Episodes")
                for _ in eps_tqdm:
                    iteration_train_examples.extend(self.execute_episode_batch())
                self.train_examples_history.append(iteration_train_examples)
                if len(self.train_examples_history) > self.args.get('numItersForTrainExamplesHistory',
                                                                    20): self.train_examples_history.pop(0)
                self.save_train_examples(i - 1)
            train_examples = []
            for e in self.train_examples_history: train_examples.extend(e)
            if not train_examples:
                print("No training examples generated. Skipping training for this iteration.")
                continue
            np.random.shuffle(train_examples)
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')

            # --- 修改: 将 writer 和迭代次数 i 传递给 train 函数 ---
            self.nnet.train(train_examples, i, self.writer)

            print('PITTING AGAINST PREVIOUS VERSION')
            mcts_args = katago_cpp_core.MCTSArgs()
            mcts_args.numMCTSSims, mcts_args.cpuct, mcts_args.dirichletAlpha, mcts_args.epsilon = self.args.numMCTSSims, self.args.cpuct, 0, 0
            nnet_mcts, pnet_mcts = katago_cpp_core.MCTS(self.game, self.nnet.predict_batch,
                                                        mcts_args), katago_cpp_core.MCTS(self.game,
                                                                                         self.pnet.predict_batch,
                                                                                         mcts_args)
            arena = Arena(nnet_mcts, pnet_mcts, self.game, self.args)
            nwins, pwins, draws = arena.play_games_batch(self.args.get('arenaCompare', 40))

            # --- 新增: 记录 Arena 对战结果到 TensorBoard ---
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

    def get_checkpoint_file(self, iteration):
        return f'checkpoint_{iteration}.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args['checkpoint']
        if not os.path.exists(folder): os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f: Pickler(f).dump(self.train_examples_history)

    def load_train_examples(self):
        model_file = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(f'File "{examples_file}" with train examples not found!')
            return False
        with open(examples_file, "rb") as f: self.train_examples_history = Unpickler(f).load()
        self.skip_first_self_play = True
        return True
