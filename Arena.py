import numpy as np
from tqdm import tqdm
import sys
import math  # ADDED: Import math module for isclose function


class Arena:
    """
    An Arena class to pit two models (MCTS instances) against each other.
    """

    def __init__(self, nnet_mcts, pnet_mcts, game, args):
        self.nnet_mcts = nnet_mcts
        self.pnet_mcts = pnet_mcts
        self.game = game
        self.args = args

    def _print_final_boards(self, boards, desc):
        """
        Helper function to print all final board states for a batch.
        打印一个批次中所有对局的最终棋盘状态。
        """
        print("\n" + "=" * 50, file=sys.stderr)
        print(f"Final Board States for: {desc}", file=sys.stderr)
        print("=" * 50, file=sys.stderr)
        for i, board in enumerate(boards):
            print(f"\n--- Final Board for Game {i} ---", file=sys.stderr)
            print("--- Black Stones (X) ---", file=sys.stderr)
            print(np.where(board[0] == 1, 'X', '.'), file=sys.stderr)
            print("--- White Stones (O) ---", file=sys.stderr)
            print(np.where(board[1] == 1, 'O', '.'), file=sys.stderr)
            print("--- Black Control (x) ---", file=sys.stderr)
            print(np.where(board[2] == 1, 'x', '.'), file=sys.stderr)
            print("--- White Control (o) ---", file=sys.stderr)
            print(np.where(board[3] == 1, 'o', '.'), file=sys.stderr)
        print("=" * 50 + "\n", file=sys.stderr)
        sys.stderr.flush()

    def play_games_batch(self, num_games, verbose=False):
        """
        Plays a batch of games between the new and previous networks.
        """
        # ADDED: Enable C++ logging for Arena
        # 新增: 为Arena对战开启C++日志
        if hasattr(self.game, 'setLogging'):
            self.game.setLogging(True)
        else:
            print("[Arena.py] Warning: game.setLogging() not found. C++ logs will not be printed.", file=sys.stderr)

        num_games_half = num_games // 2
        p1_starts_wins, p1_starts_losses, p1_starts_draws = self._play_batch(
            self.nnet_mcts, self.pnet_mcts, num_games_half, "New Net (P1) vs Prev Net (P2)", start_game_idx=0
        )
        p2_starts_wins, p2_starts_losses, p2_starts_draws = self._play_batch(
            self.pnet_mcts, self.nnet_mcts, num_games - num_games_half, "Prev Net (P1) vs New Net (P2)",
            start_game_idx=num_games_half
        )

        # ADDED: Disable C++ logging after Arena
        # 新增: 对战结束后关闭C++日志
        if hasattr(self.game, 'setLogging'):
            self.game.setLogging(False)

        new_net_wins = p1_starts_wins + p2_starts_losses
        prev_net_wins = p1_starts_losses + p2_starts_wins
        draws = p1_starts_draws + p2_starts_draws
        return new_net_wins, prev_net_wins, draws

    def _play_batch(self, p1_mcts, p2_mcts, num_games, desc, start_game_idx):
        """
        Helper function to play a batch of games where p1_mcts is always Player 1.
        """
        if num_games == 0:
            return 0, 0, 0

        wins, losses, draws = 0, 0, 0

        initial_results = [self.game.getInitialBoard() for _ in range(num_games)]
        boards = [np.array(res[0]).reshape(13, 9, 9) for res in initial_results]
        hashes = [res[1] for res in initial_results]

        current_players = [1] * num_games
        dones = np.array([False] * num_games)

        rounds_played = 0
        temp_threshold = self.args.get('tempArena', 15)
        pbar = tqdm(total=self.args.max_rounds, desc=desc)

        while not dones.all():
            rounds_played += 1
            temp = 1 if rounds_played <= temp_threshold else 0
            active_indices = np.where(dones == False)[0]
            if len(active_indices) == 0:
                break

            p1_canonical_boards, p1_canonical_hashes, p1_indices = [], [], []
            p2_canonical_boards, p2_canonical_hashes, p2_indices = [], [], []

            for i in active_indices:
                player = current_players[i]
                canonical_board, canonical_hash = self.game.getCanonicalForm(boards[i], hashes[i], player)
                canonical_board_np = np.array(canonical_board).reshape(13, 9, 9)

                if player == 1:
                    p1_canonical_boards.append(canonical_board_np)
                    p1_canonical_hashes.append(canonical_hash)
                    p1_indices.append(i)
                else:
                    p2_canonical_boards.append(canonical_board_np)
                    p2_canonical_hashes.append(canonical_hash)
                    p2_indices.append(i)

            actions = {}
            if p1_canonical_boards:
                seeds = np.random.randint(0, 2 ** 32 - 1, size=len(p1_canonical_boards), dtype=np.uint32)
                pis = p1_mcts.getActionProbs(p1_canonical_boards, p1_canonical_hashes, seeds.tolist(), temp=temp)
                for i, pi in enumerate(pis):
                    original_index = p1_indices[i]
                    actions[original_index] = np.argmax(pi) if temp == 0 else np.random.choice(len(pi), p=pi)

            if p2_canonical_boards:
                seeds = np.random.randint(0, 2 ** 32 - 1, size=len(p2_canonical_boards), dtype=np.uint32)
                pis = p2_mcts.getActionProbs(p2_canonical_boards, p2_canonical_hashes, seeds.tolist(), temp=temp)
                for i, pi in enumerate(pis):
                    original_index = p2_indices[i]
                    actions[original_index] = np.argmax(pi) if temp == 0 else np.random.choice(len(pi), p=pi)

            for i in active_indices:
                action = actions[i]
                next_board_list, next_player, next_hash = self.game.getNextState(boards[i], current_players[i], action)
                boards[i] = np.array(next_board_list).reshape(13, 9, 9)
                current_players[i] = next_player
                hashes[i] = next_hash

                game_result = self.game.getGameEnded(boards[i], 1)
                if game_result != 0:
                    dones[i] = True
                    if math.isclose(game_result, 1e-4, rel_tol=1e-6):
                        draws += 1
                    elif game_result > 0:  # Win for P1
                        wins += 1
                    else:  # Loss for P1 (game_result < 0)
                        losses += 1

            pbar.update(1)
        pbar.close()

        #self._print_final_boards(boards, desc)
        return wins, losses, draws
