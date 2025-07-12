import numpy as np
import math


class MCTS:
    """
    Monte Carlo Tree Search implementation designed for batch processing.
    V2: Uses a combined utility (win/loss + score) for guiding the search.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.Qsa = {}  # Stores Q values (Total Utility) for s,a pairs
        self.Nsa = {}  # Stores #times edge s,a was visited
        self.Ns = {}  # Stores #times board s was visited
        self.Ps = {}  # Stores initial policy (returned by neural net)
        self.Vs = {}  # Stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1, num_sims=None):
        # Determine the maximum number of simulations from args, default to 25
        max_sims_from_args = self.args.get('numMCTSSims', 200)
        min_random_sims = 50
        upper_bound_sims = max(min_random_sims, max_sims_from_args)
        sims_to_run = num_sims if num_sims is not None else np.random.randint(min_random_sims, upper_bound_sims + 1)

        mcts_batch_size = self.args.get('mcts_batch_size', 64)

        # Add Dirichlet noise to the root node policy for exploration
        s_root = self.game.string_representation(canonicalBoard)
        if s_root not in self.Ps:
            # First time visiting the root, do one prediction to initialize
            policy, _, _, _, _ = self.nnet.predict_batch(np.array([canonicalBoard]))
            self._expand_node(canonicalBoard, policy[0])

        dirichlet_alpha = self.args.get('dirichletAlpha', 0.1)
        if dirichlet_alpha > 0:
            self._add_dirichlet_noise(s_root, self.Vs[s_root])

        for i in range(0, sims_to_run, mcts_batch_size):
            leaf_nodes_data = []
            for _ in range(min(mcts_batch_size, sims_to_run - i)):
                leaf_data = self._find_leaf(np.copy(canonicalBoard))
                leaf_nodes_data.append(leaf_data)

            leaf_states = [data['leaf_state'] for data in leaf_nodes_data]
            if not leaf_states: continue

            # Batch prediction returns: policy, win_out, score_out, score_var_out, ownership_out
            predictions = self.nnet.predict_batch(np.array(leaf_states))
            policies, win_values, score_values, _, _ = predictions[0], predictions[1], predictions[2], predictions[3], \
            predictions[4]

            for idx, data in enumerate(leaf_nodes_data):
                self._expand_and_backpropagate(
                    data['leaf_state'], data['path'],
                    win_values[idx][0], score_values[idx][0], policies[idx]
                )

        s = self.game.string_representation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        if 'pruning_threshold' in self.args:
            for a in range(len(counts)):
                if (s, a) in self.Qsa and self.Qsa[(s, a)] < self.args['pruning_threshold']:
                    counts[a] = 0

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        if counts_sum == 0:
            print("Warning: All moves were pruned or had zero visits. Falling back.")
            valids = self.game.get_valid_moves(canonicalBoard, 1)
            return valids / np.sum(valids)

        probs = [x / counts_sum for x in counts]
        return probs

    def _find_leaf(self, board_state):
        path = []
        s_str = self.game.string_representation(board_state)

        while s_str in self.Ps:
            self.Ns[s_str] += 1
            valids = self.Vs[s_str]
            cur_best = -float('inf')
            best_act = -1

            for a in range(self.game.get_action_size()):
                if valids[a]:
                    qsa = self.Qsa.get((s_str, a), 0)
                    nsa = self.Nsa.get((s_str, a), 0)
                    u = qsa + self.args.get('cpuct', 1.0) * self.Ps[s_str][a] * math.sqrt(self.Ns[s_str]) / (1 + nsa)
                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            path.append((s_str, a))
            board_state, _ = self.game.get_next_state(board_state, 1, a)
            board_state = self.game.get_canonical_form(board_state, -1)
            s_str = self.game.string_representation(board_state)

        return {'path': path, 'leaf_state': board_state}

    def _expand_and_backpropagate(self, leaf_state, path, win_value, score_value, policy):
        game_ended_val = self.game.get_game_ended(leaf_state, 1)

        if game_ended_val != 0:
            # Terminal node: utility is determined solely by the game result
            total_utility = game_ended_val
        else:
            # Non-terminal node: expand and calculate combined utility
            self._expand_node(leaf_state, policy)

            # Calculate TotalUtility = UtilityValue + UtilityScore
            factor_winloss = self.args.get('factor_winloss', 1.0)
            utility_value = factor_winloss * win_value
            # f(x) = (2 / pi) * arctan(x/2)
            utility_score = (2 / math.pi) * math.atan(score_value / 2.0)
            total_utility = utility_value + utility_score

        # Backpropagate the total utility
        for s, a in reversed(path):
            self.Qsa[(s, a)] = (self.Nsa.get((s, a), 0) * self.Qsa.get((s, a), 0) + total_utility) / (
                        self.Nsa.get((s, a), 0) + 1)
            self.Nsa[(s, a)] = self.Nsa.get((s, a), 0) + 1
            total_utility = -total_utility  # Switch perspective

    def _expand_node(self, board_state, policy):
        """Expands a node, storing its policy and valid moves."""
        s_str = self.game.string_representation(board_state)
        self.Ps[s_str] = policy
        valids = self.game.get_valid_moves(board_state, 1)
        self.Ps[s_str] = self.Ps[s_str] * valids
        sum_Ps_s = np.sum(self.Ps[s_str])
        if sum_Ps_s > 0:
            self.Ps[s_str] /= sum_Ps_s
        self.Vs[s_str] = valids
        self.Ns[s_str] = 0

    def _add_dirichlet_noise(self, s_str, valids):
        """Adds Dirichlet noise to the policy of a node."""
        noise = np.random.dirichlet([self.args.get('dirichletAlpha', 0.1)] * int(np.sum(valids)))
        noisy_policy = self.Ps[s_str].copy()
        idx = 0
        for a, is_valid in enumerate(valids):
            if is_valid:
                noisy_policy[a] = (1 - self.args.get('epsilon', 0.25)) * noisy_policy[a] + self.args.get('epsilon',
                                                                                                         0.25) * noise[
                                      idx]
                idx += 1
        self.Ps[s_str] = noisy_policy

