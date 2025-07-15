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

    def __init__(self, game, nnet, args, writer):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.writer = writer
        self.pnet = self.nnet.__class__(self.game, self.nnet.nnet_class, self.args)
        self.initial_net = self.nnet.__class__(self.game, self.nnet.nnet_class, self.args)
        self.train_examples_history = []
        self.skip_first_self_play = False

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
        Performs one batch of self-play episodes, generating training data.
        This batch is either a "fast" or "deep" simulation.

        Returns:
            tuple: (training_examples, is_deep_simulation_flag)
        """
        is_deep_simulation = np.random.rand() < self.args.get('deepProb', 0.25)

        if is_deep_simulation:
            num_sims = self.args.get('deepNumMCTSSims', 200)
            desc = "Deep Self-Play (Data Saved)"
        else:
            num_sims = self.args.get('fastNumMCTSSims', 50)
            desc = "Fast Self-Play (Data Discarded)"

        num_games = self.args.get('numParallelGames', 8)
        initial_results = [self.game.getInitialBoard() for _ in range(num_games)]
        boards = [np.array(res[0], dtype=np.float32).reshape(13, 9, 9) for res in initial_results]
        hashes = [res[1] for res in initial_results]
        current_players = [1] * num_games
        dones = np.array([False] * num_games)
        episode_histories = [[] for _ in range(num_games)]

        pbar = tqdm(total=self.args.max_rounds, desc=desc, leave=False)
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
            mcts_args.numMCTSSims = num_sims
            mcts_args.cpuct, mcts_args.dirichletAlpha, mcts_args.epsilon, mcts_args.factor_winloss = self.args.cpuct, self.args.dirichletAlpha, self.args.epsilon, self.args.factor_winloss
            mcts = katago_cpp_core.MCTS(self.game, self.nnet.predict_batch, mcts_args)

            seeds = np.random.randint(0, 2 ** 32 - 1, size=len(canonical_boards), dtype=np.uint32)
            all_pis = mcts.getActionProbs(canonical_boards, canonical_hashes, seeds.tolist(), temp=temp)
            for i, pi in enumerate(all_pis):
                active_idx = active_indices[i]
                episode_histories[active_idx].append([canonical_boards[i], current_players[active_idx], pi])
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

        return all_train_examples, is_deep_simulation

    def learn(self):
        """
        主学习循环。
        """
        print("Saving initial model to be used as a baseline...")
        self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='initial.pth.tar')
        self.initial_net.load_checkpoint(folder=self.args['checkpoint'], filename='initial.pth.tar')
        print("Initial model saved and loaded.")
        sys.stdout.flush()

        for i in range(1, self.args.get('numIters', 100) + 1):
            print(f'------ ITERATION {i} ------')
            sys.stdout.flush()

            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque(maxlen=self.args.get('maxlenOfQueue', 200000))
                num_eps = self.args.get('numEps', 1)
                num_deep_episodes = 0

                print(f"Starting self-play for {num_eps} episode batches...")
                sys.stdout.flush()

                eps_tqdm = tqdm(range(num_eps), desc="Collecting Self-Play Episodes")
                for _ in eps_tqdm:
                    new_examples, is_deep = self.execute_episode_batch()
                    if is_deep:
                        iteration_train_examples.extend(new_examples)
                        num_deep_episodes += 1

                print(
                    f"\nFinished self-play. Collected {len(iteration_train_examples)} training examples from {num_deep_episodes} deep-simulation episode batches.")
                sys.stdout.flush()

                self.train_examples_history.append(iteration_train_examples)
                if len(self.train_examples_history) > self.args.get('numItersForTrainExamplesHistory', 20):
                    self.train_examples_history.pop(0)
                self.save_train_examples(i - 1)

            train_examples = []
            for e in self.train_examples_history: train_examples.extend(e)
            if not train_examples:
                print("No training examples generated. Skipping training for this iteration.")
                continue
            np.random.shuffle(train_examples)
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')

            self.nnet.train(train_examples, i, self.writer)

            print('PITTING AGAINST PREVIOUS VERSION')
            mcts_args = katago_cpp_core.MCTSArgs()
            mcts_args.numMCTSSims, mcts_args.cpuct, mcts_args.dirichletAlpha, mcts_args.epsilon = self.args.numMCTSSims, self.args.cpuct, 0, 0
            nnet_mcts = katago_cpp_core.MCTS(self.game, self.nnet.predict_batch, mcts_args)
            pnet_mcts = katago_cpp_core.MCTS(self.game, self.pnet.predict_batch, mcts_args)
            arena = Arena(nnet_mcts, pnet_mcts, self.game, self.args)
            nwins, pwins, draws = arena.play_games_batch(self.args.get('arenaCompare', 40))

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

    def load_separate_train_examples(self, filepath):
        """
        从指定文件独立加载训练数据，用于训练新模型。
        加载后将跳过第一次自对弈。
        """
        if not os.path.isfile(filepath):
            print(f'File "{filepath}" with train examples not found!')
            return False

        print(f"Loading training examples from a separate file: {filepath}")
        try:
            with open(filepath, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            # 设置标志位，在下一轮学习中跳过自对弈环节
            self.skip_first_self_play = True
            print("Training examples loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading training examples: {e}")
            return False

    # --- 重命名函数: 加载模型及其附带的训练数据 ---
    def load_train_examples(self):
        """
        加载指定模型检查点及其附带的训练数据。
        """
        model_file = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(f'File "{examples_file}" with train examples not found!')
            return False

        print(f"Loading training examples from file associated with model: {examples_file}")
        with open(examples_file, "rb") as f:
            self.train_examples_history = Unpickler(f).load()

        # 设置标志位，在下一轮学习中跳过自对弈环节
        self.skip_first_self_play = True
        print("Training examples loaded successfully.")
        return True
