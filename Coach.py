import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
import numpy as np
from tqdm import tqdm
from Arena import Arena

from MCTS import MCTS


class Coach:
    """
    This class executes the self-play + learning loop. It uses a batch-based MCTS
    to generate training examples and then trains the neural network.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.game, self.nnet.nnet_class, self.args)  # The competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.train_examples_history = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()

    def execute_episode_batch(self):
        """
        Plays numParallelGames episodes in parallel to generate training examples.
        """
        num_games = self.args.get('numParallelGames', 8)
        episode_step = 0

        # --- Initialization of parallel games ---
        boards = [self.game.get_initial_board() for _ in range(num_games)]
        current_players = [1] * num_games
        dones = [False] * num_games

        # Store episode history for each parallel game
        # each element is a list of (board, player, pi)
        episode_histories = [[(self.game.get_canonical_form(b, p), p, None)] for b, p in zip(boards, current_players)]

        # Use tqdm for a progress bar over game steps
        pbar = tqdm(total=self.game.max_rounds, desc="Batch Self-Play")

        while not all(dones):
            episode_step += 1

            # --- MCTS step for all non-terminated games ---
            active_indices = [i for i, done in enumerate(dones) if not done]
            active_boards = [boards[i] for i in active_indices]
            active_players = [current_players[i] for i in active_indices]

            # Get canonical boards for the MCTS
            canonical_boards = np.array(
                [self.game.get_canonical_form(b, p) for b, p in zip(active_boards, active_players)])

            # Get action probabilities from MCTS
            # Note: MCTS itself handles batching of NN predictions internally
            temp = int(episode_step < self.args.get('tempThreshold', 15))
            pis = [self.mcts.getActionProb(b, temp=temp) for b in canonical_boards]

            # --- Update histories and execute moves ---
            for i, pi in zip(active_indices, pis):
                # Update history with the policy vector
                # The last element's pi was None, now we set it
                episode_histories[i][-1] = (episode_histories[i][-1][0], episode_histories[i][-1][1], pi)

                # Sample an action and execute the move
                action = np.random.choice(len(pi), p=pi)
                boards[i], current_players[i] = self.game.get_next_state(boards[i], current_players[i], action)

                # Append the new state to the history
                canonical_board = self.game.get_canonical_form(boards[i], current_players[i])
                episode_histories[i].append((canonical_board, current_players[i], None))

                # Check if the game has ended
                r = self.game.get_game_ended(boards[i], current_players[i])
                if r != 0:
                    dones[i] = True

            pbar.update(1)
        pbar.close()

        # --- Finalize and return training examples ---
        all_train_examples = []
        for i in range(num_games):
            game_result_win_loss = self.game.get_game_ended(boards[i], 1)  # Get result from Black's perspective
            game_result_score = self.game.get_score(boards[i], 1)

            # Create the one-hot score distribution target
            score_dist_target = np.zeros(self.nnet.max_score * 2 + 1, dtype=np.float32)
            score_idx = int(game_result_score) + self.nnet.max_score
            if 0 <= score_idx < len(score_dist_target):
                score_dist_target[score_idx] = 1.0

            for board, player, pi in episode_histories[i]:
                if pi is None: continue
                # The value target is from the perspective of the player at that state
                win_loss_target = game_result_win_loss * player

                # The ownership target can be computed from the final board state
                final_board = boards[i]
                ownership_target = (final_board[:, :, 2] - final_board[:, :,
                                                           3]) * player  # Black's control - White's control

                all_train_examples.append((board, pi, win_loss_target, score_dist_target, ownership_target))

        return all_train_examples

    def learn(self):
        """
        Performs numIters iterations with self-play, training, and pitting.
        """
        for i in range(1, self.args.get('numIters', 100) + 1):
            print(f'------ ITERATION {i} ------')

            # --- Self-Play ---
            if not self.skip_first_self_play or i > 1:
                iteration_train_examples = deque(self.execute_episode_batch(),
                                                 maxlen=self.args.get('maxlenOfQueue', 200000))

                # Append new examples to history
                self.train_examples_history.append(iteration_train_examples)
                if len(self.train_examples_history) > self.args.get('numItersForTrainExamplesHistory', 20):
                    print(
                        f"Removing oldest train examples from history, keeping {self.args.get('numItersForTrainExamplesHistory', 20)} iterations.")
                    self.train_examples_history.pop(0)
                self.save_train_examples(i - 1)

            # --- Training the Network ---
            # Combine all examples from history for training
            train_examples = []
            for e in self.train_examples_history:
                train_examples.extend(e)

            # Shuffle examples before training
            np.random.shuffle(train_examples)

            # Train the network
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.nnet.train(train_examples)

            # --- Pitting the Networks ---
            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(self.pnet.predict_batch([x])[0]),
                          lambda x: np.argmax(self.nnet.predict_batch([x])[0]), self.game, self.args)

            pwins, nwins, draws = arena.play_games_batch(self.args.get('arenaCompare', 40))
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
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)

    def load_train_examples(self):
        model_file = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            print(f'File "{examples_file}" with train examples not found!')
            return False
        with open(examples_file, "rb") as f:
            self.train_examples_history = Unpickler(f).load()
        self.skip_first_self_play = True
        return True

