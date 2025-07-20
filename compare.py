import argparse
import sys
import numpy as np
from tqdm import tqdm
import katago_cpp_core
from nn_wrapper import NNetWrapper
from nn_model import NNet as nnet_class
from utils import *
from Arena import Arena

import katago_cpp_core



def main():
    """
    主执行函数
    Main execution function
    """
    dotdict_args = dotdict({
        'numIters': 50,
        'numEps': 8,  # 每个迭代的自对弈批次数
        'tempThreshold': 15,
        'tempArena': 15,
        'updateThreshold': 0.55,
        'maxlenOfQueue': 200000,
        'deepProb': 0.25,  # 进行深度模拟的概率
        'deepNumMCTSSims': 600,  # 深度模拟的MCTS次数
        'fastNumMCTSSims': 100,
        'arenaNumMCTSSims': 600,
        'arenaCompare': 64,  # 新旧模型对抗的棋局数
        'cpuct': 1.5,
        'dirichletAlpha': 0.1,
        'epsilon': 0.25,
        'max_rounds': 50,
        'checkpoint': './temp/',
        'log_dir': './logs/',
        'load_model': True,
        'load_separate_examples': False,
        'load_separate_file': 'save/checkpoint_15.pth.tar.examples',
        'load_folder_file': ('./save/', 'checkpoint_15.pth.tar'),
        'numItersForTrainExamplesHistory': 20,
        'numParallelGames': 240,  # 并行游戏数
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
        'w_own': 2.0,
        'initialCompare': 40,
        'num_threads': 24,
        'numGames': 100
    })


    g = katago_cpp_core.Game(n=9, max_rounds=dotdict_args.max_rounds)
    NNet = NNetWrapper
    nnet_class_placeholder = nnet_class

    print("加载模型和 MCTS...")

    # 2. 加载模型1
    #    Load Model 1
    n1 = NNet(g, nnet_class_placeholder, dotdict_args)
    n1.load_checkpoint(folder='save', filename='checkpoint_2.pth.tar')
    mcts_args1 = katago_cpp_core.MCTSArgs()
    mcts_args1.numMCTSSims = dotdict_args.arenaNumMCTSSims
    mcts_args1.cpuct = dotdict_args.cpuct
    mcts_args1.dirichletAlpha = 0  # 在评估时不使用噪声
    mcts_args1.epsilon = 0  # In evaluation, no noise is used
    mcts1 = katago_cpp_core.MCTS(g, n1.predict_batch, mcts_args1)

    # 3. 加载模型2
    #    Load Model 2
    n2 = NNet(g, nnet_class_placeholder, dotdict_args)
    n2.load_checkpoint(folder='save', filename='checkpoint_2.pth.tar')
    mcts_args2 = katago_cpp_core.MCTSArgs()
    mcts_args2.numMCTSSims = dotdict_args.arenaNumMCTSSims
    mcts_args2.cpuct = dotdict_args.cpuct
    mcts_args2.dirichletAlpha = 0
    mcts_args2.epsilon = 0
    mcts2 = katago_cpp_core.MCTS(g, n2.predict_batch, mcts_args2)

    # 4. 创建 Arena 并开始对战
    #    Create Arena and start the match
    #    Arena 会让两个模型轮流执先手
    #    Arena will have the two models take turns playing first
    arena = Arena(nnet_mcts=mcts1, pnet_mcts=mcts2, game=g, args=dotdict_args)

    print("\n" + "=" * 50)
    print(f"开始对战: (模型1) vs  (模型2)")
    print(f"总对战局数: {dotdict_args.numGames}")
    print("=" * 50 + "\n")

    # play_games_batch 返回 (nnet胜场, pnet胜场, 平局)
    # play_games_batch returns (nnet_wins, pnet_wins, draws)
    model1_wins, model2_wins, draws, firstwinsrate, secondwinsrate = arena.play_games_batch(dotdict_args.numGames, verbose=True)

    # 5. 打印最终结果
    #    Print final results
    total_games = model1_wins + model2_wins + draws
    print("\n" + "=" * 50)
    print("对 战 结 果")
    print("=" * 50)
    print(f"总对局数: {total_games}")
    print(f"模型 1 胜场: {model1_wins}")
    print(f"模型 2 胜场: {model2_wins}")
    print(f"平局: {draws}")
    print(f"先手胜率：{firstwinsrate}")
    print(f'后手胜率：{secondwinsrate}')
    print("-" * 50)

    if total_games > 0:
        win_rate_model1 = model1_wins / (model1_wins+model2_wins)
        print(f"模型 1 胜率: {win_rate_model1:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
