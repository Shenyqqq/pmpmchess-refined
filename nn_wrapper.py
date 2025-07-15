import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F


class NNetWrapper:
    def __init__(self, game, nnet_class, args):
        self.game = game
        self.args = args
        self.nnet = nnet_class(game, args)
        self.nnet_class = nnet_class
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.board_channels = args.num_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nnet.to(self.device)
        self.max_score = self.board_x * self.board_y
        self.score_range = torch.arange(-self.max_score, self.max_score + 1, device=self.device, dtype=torch.float32)

    def train(self, examples, iteration, writer):  # --- 修改: 接收 iteration 和 writer ---
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.get('lr', 0.001))

        epochs = self.args.get('epochs',10)
        batch_count = len(examples)/self.args.get('batch_size',64)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.get('lr', 0.001),
            total_steps=int(epochs * batch_count)
        )

        for epoch in range(self.args.get('epochs', 10)):
            print(f'EPOCH ::: {epoch + 1}')
            self.nnet.train()

            # 用于累加一个 epoch 内的损失值
            total_loss_acc, pi_loss_acc, v_loss_acc, own_loss_acc, score_loss_acc = 0.0, 0.0, 0.0, 0.0, 0.0

            batch_count = int(len(examples) / self.args.get('batch_size', 64))
            if batch_count == 0: continue

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.args.get('batch_size', 64))
                boards, pis, vs, score_dists, ownerships = list(zip(*[examples[i] for i in sample_ids]))

                boards = torch.FloatTensor(np.array(boards).astype(np.float32)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32)).to(self.device)
                target_score_dists = torch.FloatTensor(np.array(score_dists)).to(self.device)
                target_ownerships = torch.FloatTensor(np.array(ownerships).astype(np.float32)).to(self.device)
                target_ownerships = target_ownerships.view(-1, 1, self.board_x, self.board_y)

                out_pi, out_v_win, out_score, out_score_var, out_ownership = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v_win)
                l_own = self.loss_ownership(target_ownerships, out_ownership)
                l_score = self.loss_score_dist(target_score_dists, out_score, out_score_var)
                w_pi = self.args.get('w_pi', 1.0)
                w_v = self.args.get('w_v', 5.0)
                w_own = self.args.get('w_own', 4.0)
                w_score = self.args.get('w_score', 1.0)
                total_loss = w_pi * l_pi + w_v * l_v + w_own * l_own + w_score * l_score

                total_loss_acc += total_loss.item()
                pi_loss_acc += l_pi.item()
                v_loss_acc += l_v.item()
                own_loss_acc += l_own.item()
                score_loss_acc += l_score.item()

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                t.set_postfix(Loss=total_loss.item())

            # --- 新增: 在每个 epoch 结束后记录平均损失 ---
            global_step = (iteration - 1) * self.args.get('epochs', 10) + epoch
            writer.add_scalar('Loss/Total', total_loss_acc / batch_count, global_step)
            writer.add_scalar('Loss/Policy', pi_loss_acc / batch_count, global_step)
            writer.add_scalar('Loss/Value', v_loss_acc / batch_count, global_step)
            writer.add_scalar('Loss/Ownership', own_loss_acc / batch_count, global_step)
            writer.add_scalar('Loss/Score', score_loss_acc / batch_count, global_step)
            current_lr = optimizer.param_groups[0]['lr']
            global_step = (iteration - 1) * epochs + epoch
            writer.add_scalar('LearningRate', current_lr, global_step)

    def predict_batch(self, boards):
        boards_np = np.array(boards, dtype=np.float32)
        boards_tensor = torch.FloatTensor(boards_np).to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            results = self.nnet(boards_tensor)
        return [r.cpu().numpy() for r in results]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * torch.log(outputs.clamp(min=1e-9))) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def loss_ownership(self, targets, outputs):
        return torch.sum((targets.view(-1) - outputs.view(-1)) ** 2) / targets.numel()

    def loss_score_dist(self, target_dist, pred_mean, pred_var):
        pred_var = torch.clamp(pred_var, min=1e-4)
        normal_dist = torch.distributions.normal.Normal(pred_mean, torch.sqrt(pred_var))
        log_probs = normal_dist.log_prob(self.score_range.unsqueeze(0))
        predicted_log_dist = F.log_softmax(log_probs, dim=1)
        return -torch.sum(target_dist * predicted_log_dist) / target_dist.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath): raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])
