from Coach import Coach
from nn_wrapper import NNetWrapper
from nn_model import NNet
from utils import dotdict
import katago_cpp_core
from torch.utils.tensorboard import SummaryWriter

args = dotdict({
    'numIters': 50,
    'numEps': 8,  # 每个迭代的自对弈批次数
    'tempThreshold': 15,
    'tempArena': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'maxTrainExamples': 3200000,
    'deepProb': 0.25,  # 进行深度模拟的概率
    'deepNumMCTSSims': 900,  # 深度模拟的MCTS次数
    'fastNumMCTSSims': 150,
    'arenaNumMCTSSims': 400,
    'arenaCompare': 100,  # 新旧模型对抗的棋局数
    'cpuct': 1.5,
    'dirichletAlpha': 0.1,
    'epsilon': 0.25,
    'max_rounds': 50,
    'checkpoint': './temp/',
    'log_dir': './logs/',
    'load_model': True,
    'load_separate_examples': False,
    'load_separate_file': 'save/checkpoint_4.pth.tar.examples', # 单独加载训练数据
    'load_folder_file': ('./save/', 'checkpoint_1.pth.tar'), # 加载模型和训练数据
    'numItersForTrainExamplesHistory': 20,
    'numParallelGames': 240,  # 并行游戏数
    'lr': 0.00001,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 13,
    'num_filters': 128,
    'num_residual_blocks': 8,
    'factor_winloss': 1.0,
    'w_pi': 1.0,
    'w_v': 1.0,
    'w_score': 0.2,
    'w_own': 2.0,
    'initialCompare': 40,
    'num_threads': 24
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
    if args.load_separate_examples:
        c.load_separate_train_examples(args.load_separate_file)

    c.learn()

    writer.close()


if __name__ == "__main__":
    main()
