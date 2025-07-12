import numpy as np
from tqdm import tqdm


class Arena:
    """
    An Arena class where two agents (neural networks) compete against each other.
    This implementation supports batching to play multiple games in parallel.
    """

    def __init__(self, player1, player2, game, args):
        """
        Args:
            player1, player2: Functions that take a board and return an action.
            game: The game object.
            args: A dictionary of hyperparameters.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.args = args

    def play_game(self, verbose=False):
        """
        DEPRECATED: Plays a single game. Use play_games_batch for efficiency.
        """
        players = [self.player2, None, self.player1]
        cur_player = 1
        board = self.game.get_initial_board()
        it = 0
        while self.game.get_game_ended(board, cur_player) == 0:
            it += 1
            canonical_board = self.game.get_canonical_form(board, cur_player)
            action = players[cur_player + 1](canonical_board)
            board, cur_player = self.game.get_next_state(board, cur_player, action)
        return self.game.get_game_ended(board, 1)

    def play_games_batch(self, num_games, verbose=False):
        """
        Plays num_games in parallel, half with player1 starting, half with player2.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws: games that ended in a draw
        """
        num_games_half = num_games // 2

        # --- Run games where player 1 starts ---
        p1_starts_wins, p1_starts_losses, p1_starts_draws = self._play_batch(
            self.player1, self.player2, num_games_half, "P1 (new) starts"
        )

        # --- Run games where player 2 starts ---
        p2_starts_wins, p2_starts_losses, p2_starts_draws = self._play_batch(
            self.player2, self.player1, num_games - num_games_half, "P2 (prev) starts"
        )

        one_won = p1_starts_wins + p2_starts_losses
        two_won = p1_starts_losses + p2_starts_wins
        draws = p1_starts_draws + p2_starts_draws

        return one_won, two_won, draws

    def _play_batch(self, p1, p2, num_games, desc):
        """Helper function to play a batch of games."""
        if num_games == 0:
            return 0, 0, 0

        wins, losses, draws = 0, 0, 0

        boards = [self.game.get_initial_board() for _ in range(num_games)]
        current_players = [1] * num_games
        dones = [False] * num_games

        pbar = tqdm(total=self.game.max_rounds, desc=desc)

        while not all(dones):
            active_indices = [i for i, done in enumerate(dones) if not done]

            # --- Get actions for all active games ---
            actions = []
            for i in active_indices:
                canonical_board = self.game.get_canonical_form(boards[i], current_players[i])
                if current_players[i] == 1:
                    actions.append(p1(canonical_board))
                else:
                    actions.append(p2(canonical_board))

            # --- Execute moves ---
            for i, action in zip(active_indices, actions):
                boards[i], current_players[i] = self.game.get_next_state(boards[i], current_players[i], action)

                # Check for game end
                r = self.game.get_game_ended(boards[i], 1)  # Result from P1's perspective
                if r != 0:
                    dones[i] = True
                    if r > 0:
                        wins += 1
                    elif r < 0:
                        losses += 1
                    else:
                        draws += 1

            pbar.update(1)
        pbar.close()

        return wins, losses, draws
