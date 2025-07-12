import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Skip connection
        return F.relu(out)

class NNet(nn.Module):
    """
    The main neural network model for the Connect-Three-Go game.
    V2: Value head now only predicts win/loss. Auxiliary head predicts score.
    """
    def __init__(self, game, args):
        super(NNet, self).__init__()
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        # --- Backbone ---
        self.conv_stem = nn.Conv2d(args.num_channels, args.num_filters, kernel_size=3, stride=1, padding=1)
        self.bn_stem = nn.BatchNorm2d(args.num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(args.num_filters, args.num_filters) for _ in range(args.num_residual_blocks)])
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # --- Prediction Heads ---

        # 1. Policy Head (predicts move probabilities)
        self.policy_conv = nn.Conv2d(args.num_filters, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        # 2. Value Head (predicts win/loss ONLY)
        self.value_conv = nn.Conv2d(args.num_filters, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 3. Ownership Head (predicts final territory ownership)
        self.ownership_conv1 = nn.Conv2d(args.num_filters, 4, kernel_size=1, stride=1)
        self.ownership_bn = nn.BatchNorm2d(4)
        self.ownership_conv2 = nn.Conv2d(4, 1, kernel_size=1, stride=1)

        # 4. Auxiliary Score Head (predicts score and score variance)
        self.aux_fc1 = nn.Linear(args.num_filters, 128)
        self.aux_fc_score = nn.Linear(128, 1)
        self.aux_fc_score_var = nn.Linear(128, 1)


    def forward(self, s):
        s = s.view(-1, self.args.num_channels, self.board_x, self.board_y)
        x = F.relu(self.bn_stem(self.conv_stem(s)))
        for block in self.res_blocks:
            x = block(x)
        res_out = x
        pooled_out = self.global_pooling(res_out)
        pooled_out_flat = pooled_out.view(-1, self.args.num_filters)

        # Policy Head
        pi = F.relu(self.policy_bn(self.policy_conv(res_out)))
        pi = pi.view(-1, 2 * self.board_x * self.board_y)
        pi = self.policy_fc(pi)
        policy_out = F.log_softmax(pi, dim=1)

        # Value Head (Win/Loss)
        v = F.relu(self.value_bn(self.value_conv(res_out)))
        v = v.view(-1, self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        win_out = torch.tanh(self.value_fc2(v))

        # Ownership Head
        own = F.relu(self.ownership_bn(self.ownership_conv1(res_out)))
        own = self.ownership_conv2(own)
        ownership_out = torch.tanh(own)

        # Auxiliary Score Head
        aux = F.relu(self.aux_fc1(pooled_out_flat))
        score_out = self.aux_fc_score(aux)
        score_var_out = F.softplus(self.aux_fc_score_var(aux)) # Ensure variance is non-negative

        return policy_out, win_out, score_out, score_var_out, ownership_out
