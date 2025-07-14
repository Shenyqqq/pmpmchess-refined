from Coach import Coach
from nn_wrapper import NNetWrapper
from nn_model import NNet
from utils import dotdict
import katago_cpp_core
from torch.utils.tensorboard import SummaryWriter

args = dotdict({
    'numIters': 50,
    'numEps': 1,  # 每个迭代的自对弈批次数
    'tempThreshold': 20,
    'tempArena': 20,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 600,  # 自对弈时MCTS模拟次数的上限
    'min_numSims': 200,  # 自对弈时MCTS模拟次数的下限
    'arenaCompare': 48,  # 新旧模型对抗的棋局数
    'cpuct': 2.0,
    'dirichletAlpha': 0.1,
    'epsilon': 0.25,
    'max_rounds': 50,
    'checkpoint': './temp/',
    'log_dir': './logs/',
    'load_model': False,
    'load_folder_file': ('./models/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'numParallelGames': 128,  # 并行游戏数
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 13,
    'num_filters': 128,
    'num_residual_blocks': 8,
    'factor_winloss': 1.0,
    'w_pi': 1.0,
    'w_v': 1.0,
    'w_score': 0.2,
    'w_own': 2.0
})


def main():
    g = katago_cpp_core.Game(n=9, max_rounds=args.max_rounds)
    nnet = NNetWrapper(g, NNet, args)

    writer = SummaryWriter(log_dir=args.log_dir)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args, writer)

    if args.load_model:
        c.load_train_examples()

    c.learn()

    writer.close()


if __name__ == "__main__":
    main()
