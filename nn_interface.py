import numpy as np
from game import Game as LogicGame  # Renaming to avoid confusion


class Game:
    """
    This class specifies the game interface for the Connect-Three-Go game.
    It's designed to be used by a training pipeline like AlphaZero.
    """

    def __init__(self, n=9, max_rounds=50):
        """
        Initializes the game.

        Args:
            n: The size of the board (n x n).
            max_rounds: The maximum number of rounds before the game ends.
        """
        self.n = n
        self.max_rounds = max_rounds
        # The board state is represented by 13 channels:
        # Channel 0: Player 1's stones (Black)
        # Channel 1: Player -1's stones (White)
        # Channel 2: Player 1's controlled territory (Black)
        # Channel 3: Player -1's controlled territory (White)
        # Channel 4: Round counter (normalized)
        # Channel 5: Player to move is Black (1.0 if black to move, else 0.0)
        # Channel 6: Player to move is White (1.0 if white to move, else 0.0)
        # Channel 7: Black's controlled count (normalized value)
        # Channel 8: White's controlled count (normalized value)
        # Channel 9: no stones tiles (1 for no stones, 0 otherwise)
        # Channel 10: uncontrolled tiles(1 for unpainted, 0 otherwise)
        # Channel 11: Black valid moves(1 for valid, 0 otherwise)
        # Channel 12: White valid moves(1 for valid, 0 otherwise)
        self.board_channels = 13

    def get_initial_board(self):
        """
        Returns the initial board state as a numpy array.
        All planes are zeroed out, except for those with initial values.
        """
        b = np.zeros((self.n, self.n, self.board_channels), dtype=np.float32)
        # At the start, Black (1) moves
        b[:, :, 5] = 1.0
        # All tiles have no stones and are uncontrolled
        b[:, :, 9] = 1.0
        b[:, :, 10] = 1.0
        # Set initial valid moves for both players
        temp_game = LogicGame(self.n)

        temp_game.current_player = 1  # Black
        black_valid_moves = np.zeros((self.n, self.n), dtype=np.float32)
        for x, y in temp_game.get_valid_moves():
            black_valid_moves[x, y] = 1
        b[:, :, 11] = black_valid_moves

        temp_game.current_player = -1  # White
        white_valid_moves = np.zeros((self.n, self.n), dtype=np.float32)
        for x, y in temp_game.get_valid_moves():
            white_valid_moves[x, y] = 1
        b[:, :, 12] = white_valid_moves

        return b

    def get_board_size(self):
        """
        Returns the dimensions of the board.
        """
        return (self.n, self.n)

    def get_action_size(self):
        """
        Returns the total number of possible actions (every spot on the board).
        """
        return self.n * self.n

    def get_next_state(self, board, player, action):
        """
        Calculates the board state after a move.

        Args:
            board: The current board state.
            player: The current player (1 or -1).
            action: The action taken by the player (an integer from 0 to n*n-1).

        Returns:
            A tuple of (next_board, next_player).
        """
        # Create a temporary game logic instance to compute the next state
        temp_game = LogicGame(self.n)

        # Unpack the board state into the logic game instance
        temp_game.board[board[:, :, 0] == 1] = 1
        temp_game.board[board[:, :, 1] == 1] = -1
        temp_game.controlled[board[:, :, 2] == 1] = 1
        temp_game.controlled[board[:, :, 3] == 1] = -1
        temp_game.round = int(board[0, 0, 4] * self.max_rounds)
        temp_game.current_player = player

        # Convert flat action to a board coordinate
        move = (action // self.n, action % self.n)

        # Apply the move using the game's logic
        temp_game.make_move(move)
        next_player = -player

        # Pack the new state into the multi-channel numpy array format
        next_board = np.zeros((self.n, self.n, self.board_channels), dtype=np.float32)

        # Channels 0-3: Stones and controlled territory
        next_board[:, :, 0][temp_game.board == 1] = 1
        next_board[:, :, 1][temp_game.board == -1] = 1
        next_board[:, :, 2][temp_game.controlled == 1] = 1
        next_board[:, :, 3][temp_game.controlled == -1] = 1

        # Channel 4: Round counter
        next_board[:, :, 4] = temp_game.round / self.max_rounds

        # Channels 5-6: Player to move
        if next_player == 1:
            next_board[:, :, 5] = 1.0
        else:
            next_board[:, :, 6] = 1.0

        # Channels 7-8: Controlled counts (normalized)
        total_squares = self.n * self.n
        next_board[:, :, 7] = np.sum(next_board[:, :, 2]) / total_squares
        next_board[:, :, 8] = np.sum(next_board[:, :, 3]) / total_squares

        # Channels 9-10: Empty and uncontrolled tiles
        next_board[:, :, 9] = (next_board[:, :, 0] == 0) & (next_board[:, :, 1] == 0)
        next_board[:, :, 10] = (next_board[:, :, 2] == 0) & (next_board[:, :, 3] == 0)

        # Channels 11-12: Valid moves for both players
        # We need to set the current player in the temp game to get the correct valid moves
        original_player = temp_game.current_player

        temp_game.current_player = 1  # Black
        black_valid_moves = np.zeros((self.n, self.n), dtype=np.float32)
        for x, y in temp_game.get_valid_moves():
            black_valid_moves[x, y] = 1
        next_board[:, :, 11] = black_valid_moves

        temp_game.current_player = -1  # White
        white_valid_moves = np.zeros((self.n, self.n), dtype=np.float32)
        for x, y in temp_game.get_valid_moves():
            white_valid_moves[x, y] = 1
        next_board[:, :, 12] = white_valid_moves

        temp_game.current_player = original_player  # Restore player

        return next_board, next_player

    def get_valid_moves(self, board, player):
        """
        Returns a binary vector indicating valid moves for the current player.

        Args:
            board: The current board state.
            player: The current player.

        Returns:
            A numpy array of size get_action_size() with 1s for valid moves and 0s otherwise.
        """
        # This function returns valid moves for the *current* player,
        # which can be read directly from the corresponding channel.
        if player == 1:
            valid_moves_plane = board[:, :, 11]
        else:
            valid_moves_plane = board[:, :, 12]
        return valid_moves_plane.flatten().astype(np.int32)

    def get_game_ended(self, board, player):
        """
        Determines if the game has ended and returns the winner.

        Args:
            board: The current board state.
            player: The player who just moved.

        Returns:
            1 if the player won, -1 if they lost, 1e-4 for a draw, 0 if not ended.
        """
        current_round = int(board[0, 0, 4] * self.max_rounds)

        if current_round < self.max_rounds:
            return 0  # Game has not ended

        # Game has ended, determine the winner based on controlled territory
        p1_score = np.sum(board[:, :, 2])
        p2_score = np.sum(board[:, :, 3])

        if p1_score > p2_score:
            return 1 if player == 1 else -1
        elif p2_score > p1_score:
            return 1 if player == -1 else -1
        else:
            return 1e-4  # Draw

    def get_canonical_form(self, board, player):
        """
        Returns the canonical form of the board.
        The canonical form is from the perspective of the current player (player).
        For example, if the player is White, the Black and White planes are swapped
        so the network always sees the game from the current player's viewpoint.
        """
        if player == 1:
            return board

        # player == -1 (White)
        canonical_board = np.copy(board)
        # Swap Black/White stone planes
        canonical_board[:, :, 0], canonical_board[:, :, 1] = board[:, :, 1], board[:, :, 0]
        # Swap Black/White control planes
        canonical_board[:, :, 2], canonical_board[:, :, 3] = board[:, :, 3], board[:, :, 2]
        # Swap Black/White to-move planes
        canonical_board[:, :, 5], canonical_board[:, :, 6] = board[:, :, 6], board[:, :, 5]
        # Swap Black/White control count planes
        canonical_board[:, :, 7], canonical_board[:, :, 8] = board[:, :, 8], board[:, :, 7]
        # Swap Black/White valid move planes
        canonical_board[:, :, 11], canonical_board[:, :, 12] = board[:, :, 12], board[:, :, 11]

        # Channels 4 (round), 9 (no stones), 10 (uncontrolled) are absolute and not swapped.
        return canonical_board

    def get_symmetries(self, board, pi):
        """
        Augments training data by creating symmetrical versions of a board state and policy.

        Args:
            board: The board state.
            pi: The policy vector.

        Returns:
            A list of (board, pi) tuples, each being a symmetry.
        """
        assert (len(pi) == self.n ** 2)
        pi_board = np.reshape(pi, (self.n, self.n))
        symmetries = []

        for i in range(1, 5):
            for j in [True, False]:
                # Rotate all channels of the board
                new_b = np.rot90(board, i, axes=(0, 1))
                new_pi = np.rot90(pi_board, i)
                if j:
                    new_b = np.fliplr(new_b)
                    new_pi = np.fliplr(new_pi)
                symmetries.append((new_b, new_pi.flatten()))
        return symmetries

    def string_representation(self, board):
        """
        Returns a string representation of the board for MCTS hashing.
        """
        return board.tobytes()

    def get_score(self, board, player):
        """
        Calculates the score from the perspective of the player.
        """
        p1_score = np.sum(board[:, :, 2])  # Black's score
        p2_score = np.sum(board[:, :, 3])  # White's score

        if player == 1:
            return p1_score - p2_score
        else:
            return p2_score - p1_score

