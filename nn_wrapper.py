import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


class NNetWrapper:
    """
    Neural Network Wrapper for batch processing.
    V2: Implements a distributional loss for the auxiliary score head.
    """

    def __init__(self, game, nnet_class, args):
        self.game = game
        self.args = args
        self.nnet = nnet_class(game, args)
        self.nnet_class = nnet_class
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)

        # Define the range of possible scores for the distributional loss
        self.max_score = self.board_x * self.board_y
        self.score_range = torch.arange(-self.max_score, self.max_score + 1, device=self.device, dtype=torch.float32)

    def train(self, examples):
        """
        Args:
            examples: A list of tuples, where each tuple is:
                      (board, policy_target, win_loss_target, score_target_dist, ownership_target).
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.get('lr', 0.001))

        for epoch in range(self.args.get('epochs', 10)):
            print(f'EPOCH ::: {epoch + 1}')
            self.nnet.train()

            batch_count = int(len(examples) / self.args.get('batch_size', 64))
            t = tqdm(range(batch_count), desc='Training Net')

            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.get('batch_size', 64))
                boards, pis, vs, score_dists, ownerships = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards).astype(np.float32)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32)).to(self.device)
                target_score_dists = torch.FloatTensor(np.array(score_dists)).to(self.device)
                target_ownerships = torch.FloatTensor(np.array(ownerships).astype(np.float32)).to(self.device)

                boards = boards.view(-1, self.game.board_channels, self.board_x, self.board_y)
                target_ownerships = target_ownerships.view(-1, 1, self.board_x, self.board_y)

                out_pi, out_v_win, out_score, out_score_var, out_ownership = self.nnet(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v_win)
                l_own = self.loss_ownership(target_ownerships, out_ownership)
                l_score = self.loss_score_dist(target_score_dists, out_score, out_score_var)

                w_spdf = self.args.get('w_spdf', 0.02)  # Weight for the score distribution loss
                total_loss = l_pi + l_v + l_own + w_spdf * l_score

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                t.set_postfix(Loss_pi=l_pi.item(), Loss_v=l_v.item(), Loss_score=l_score.item())

    def predict_batch(self, boards):
        boards_tensor = torch.FloatTensor(boards.astype(np.float32)).to(self.device)
        boards_tensor = boards_tensor.view(-1, self.game.board_channels, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            results = self.nnet(boards_tensor)

        np_results = [r.cpu().numpy() for r in results]
        return np_results

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def loss_ownership(self, targets, outputs):
        return torch.sum((targets.view(-1) - outputs.view(-1)) ** 2) / targets.numel()

    def loss_score_dist(self, target_dist, pred_mean, pred_var):
        """
        Calculates the cross-entropy loss for the score distribution.
        It creates a normal distribution from the predicted mean (score) and variance,
        then discretizes it and compares it to the target one-hot distribution.
        """
        # Clamp variance for numerical stability
        pred_var = torch.clamp(pred_var, min=1e-4)

        # Create a normal distribution N(pred_mean, pred_var)
        # Shape of score_range: [num_possible_scores]
        # Shape of pred_mean/pred_var: [batch_size, 1]
        # We need to broadcast them to [batch_size, num_possible_scores]
        normal_dist = torch.distributions.normal.Normal(pred_mean, torch.sqrt(pred_var))

        # Calculate the log probability of each possible score point
        log_probs = normal_dist.log_prob(self.score_range.unsqueeze(0))

        # Normalize to get a log-softmax-like output
        predicted_log_dist = F.log_softmax(log_probs, dim=1)

        # Calculate cross-entropy loss
        # -sum(target * log(prediction))
        return -torch.sum(target_dist * predicted_log_dist) / target_dist.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = self.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
