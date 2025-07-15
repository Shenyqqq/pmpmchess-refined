[中文](https://github.com/Shenyqqq/pmpmchess-refined/blob/master/README.md.zh)[English](https://github.com/Shenyqqq/pmpmchess-refined/blob/master/README.md)
# 泡姆棋（重制版）

Pmpmchess-Refined 是针对 Pmpmchess 游戏开发的最先进的 AI，它完全从零开始重新设计。该项目受到 KataGo 开创性架构的启发，超越了最初的 AlphaZero 框架，提供了更卓越的性能。

本项目的核心是一个高性能的 C++ 引擎和一个复杂的多头神经网络（multi-headed neural network）。这一新架构将自我对弈数据的生成过程加速了 30 倍以上，使得在消费级硬件上仅需 4 小时就能训练出具备人类竞争水平的 AI。

点击此处立即开始游戏！
[🤗 Hugging Face](https://huggingface.co/spaces/gumigumi/pmpmchess)

## 🎮 如何玩 Pmpmchess

你玩过五子棋（Gomoku）或黑白棋（Othello）吗？Pmpmchess 是独立游戏 *Popucom* 中一个超酷的迷你游戏，它巧妙地将这两种游戏的玩法融合在一起！游戏目标是既要将你的棋子连成一线，*同时*也要占领领土。

以下是规则详解：

  * **♟️ 基础规则 (棋盘与回合)**
        游戏在 9x9 的棋盘上进行。这是一场快速对局，总共持续 50 回合，这意味着你和对手每人可以下 25 步棋。

  * **💥 如何占领**
        将你的三颗棋子连成一条不间断的直线（横向、纵向或斜向），然后——BAM！——这条线上的方格就成为你的领土。但要小心！如果你的对手在这条潜在的连线上有哪怕一颗棋子，你的占领就会被阻挡。

  * **🤔 行动规则**
        你有两种落子选择：棋盘上的任意空位，或者你已经占领的领土内的任意位置。没错，你可以重复利用你自己的地盘！

  * **🏆 获胜条件**
        非常简单！当游戏结束时（50 回合全部用完），占领了更多领土方格的玩家即为获胜者。

## 演示
![gif](https://raw.githubusercontent.com/Shenyqqq/pmpmchess-refined/master/gif/1.gif)
## 关键架构升级

本项目对原有的 pmpmchessAI 进行了根本性的重构。以下的增强功能构成了其卓越性能和智能的核心。

### 1\. 高性能 C++ 引擎

整个游戏模拟和搜索逻辑已用 C++ 重写，以消除 Python 的性能瓶颈。关键特性包括：

  \* **C++ 游戏内核与 MCTS:** 核心逻辑现在是原生代码，极大地加速了自我对弈循环的每个阶段。
  \* **多线程与批处理:** 在 MCTS（蒙特卡洛树搜索）期间，通过并行批处理游戏状态，充分利用了现代多核 CPU 的性能，最大化硬件使用率。
  \* **Zobrist 哈希:** 实现了一种高效的哈希方案，可以即时识别并检索先前遇到过的棋盘状态的评估结果，节省了冗余计算。

**成果:** 这些优化使得自我对弈数据的生成速度比原始 Python 实现提高了 **30 倍以上**。

### 2\. 受 KataGo 启发的神经网络

AI 的“大脑”已从简单的策略价值网络升级为更强大、多面的架构，能够更深入地理解游戏。

  \* **多头架构 (Multi-Head Architecture):** 除了预测获胜者（价值头 Value Head）之外，网络现在还具有：
      \* **归属头 (Ownership Head):** 预测棋盘上每个点的最终归属者，使 AI 能够对领土进行推理。
      \* **辅助分数头 (Auxiliary Score Head):** 估计最终的分数差异，提供比简单的输赢预测更精细的评估。
  \* **复合损失函数 (Composite Loss Function):** 模型训练基于所有头的加权总损失（$w\_{\\pi}, w\_{v}, w\_{score}, w\_{own}$），使其能够同时学习策略、领土和得分。
  \* **扩展的输入特征 (Expanded Input Features):** 网络输入从 4 个通道扩展到 13 个通道，为 AI 提供了游戏状态的全面视图（例如，回合历史、气等），这使得 AI 能够隐式且快速地掌握游戏规则和细微差别。
  \* **高级优化 (Advanced Optimizations):** 融合了受 KataGo 启发的全局池化（Global Pooling）等现代技术，增强了模型的特征提取能力和整体性能。

### 3\. 高级 MCTS 与 GUI

搜索算法和用户界面已显著改进，以充分利用新的引擎和网络。

  \* **双目标搜索 (Dual-Objective Search):** MCTS 搜索现在同时由输赢预测（价值）和分数估计（辅助分数）共同引导，从而产生更稳健、更高质量的决策。
  \* **KataGo 风格的模拟上限 (Playout Cap):** 实施了一种复杂的双重模拟策略来平衡数据生成的速度和质量。对于大多数走法（$p=0.75$），AI 执行模拟次数较少的“快速”模拟。对于少部分走法（$p=0.25$），AI 执行模拟次数显著增多的“深度”模拟。只有这些经过深度分析的高质量棋局才会被用作训练样本。
  \* **带有 AI 分析的增强型 GUI:** 图形界面现在是一个强大的分析工具，提供对 AI 思考过程的实时洞察，并提供更好的可用性：
      \* **实时策略图 (Live Policy Map):** 可视化 AI 正在考虑的最佳走法。
      \* **输赢预测 (Win/Loss Prediction):** 显示 AI 从当前局面获胜的信心度。
      \* **分数估计 (Score Estimation):** 显示预测的最终分数差异。
      \* **归属预测 (Ownership Prediction):** 渲染预期最终棋盘控制权的热力图。
      \* **交互式控制 (Interactive Controls):** 包括悔棋等基本游戏功能。

## 训练与性能

Pmpmchess-Refined 架构不仅更强大，而且效率也显著提高。

  \* **快速训练:** 使用当前参数，一个能够击败大多数人类玩家的强大 AI 可以在大约 **4 小时**内训练完成。
  \* **可扩展性:** 系统设计具备可扩展性。要达到世界级的超人性能，只需增加神经网络的规模和 MCTS 的模拟次数即可。

关于超参数调整的说明：某些参数，例如 `$tempThreshold$`（控制从探索性移动到贪婪移动的转换）和损失权重（$w\_{\\pi}, w\_{v}, w\_{score}, w\_{own}$）非常敏感。当前值提供了一个强大的基线，但进一步调整可能会释放额外的性能潜力。

## 安装

### 对于玩家 (推荐)

最简单的游戏方式是下载预编译版本。

1.  导航到本仓库的 **Releases** 页面。
2.  下载适用于您操作系统的最新压缩包。
3.  解压存档并运行可执行文件 (`.exe`)。

### 对于开发者 (从源码安装)

要训练您自己的模型或修改代码，您需要从源码构建项目。

1.  **克隆仓库:**
    ` bash     git clone https://github.com/Shenyqqq/pmpmchess-refined.git     cd pmpmchess-refined      `
2.  **创建虚拟环境:**
    ` bash     python -m venv venv     # 在 Windows 上激活     .\venv\Scripts\activate     # 在 macOS/Linux 上激活     # source venv/bin/activate      `
3.  **安装依赖:**
    ` bash     pip install -r requirements.txt      `
4.  **编译并安装 C++ 引擎:**
    ` bash     python setup.py build_ext     python setup.py install      `

## 使用方法

  \* **训练新模型:**
    ` bash     python main.py      `
  \* **与 AI 对战:**
    ` bash     python pit.py      `

## 项目结构

```
.
├── cpp/                # 包含高性能引擎的所有 C++ 源文件
│   ├── game.cpp        # 核心游戏逻辑实现
│   ├── GameInterface.cpp # 提供游戏状态与神经网络之间的接口
│   ├── MCTS.cpp        # Monte Carlo Tree Search 的 C++ 实现
│   ├── Zobrist.cpp     # Zobrist 哈希的实用库
│   └── pybind_wrapper.cpp # 使用 pybind11 将 C++ 函数暴露给 Python
├── main.py             # 训练模型的主入口点。可在此调整超参数。
├── pit.py              # 与 AI 对战的主入口点。
├── Coach.py            # 实现主训练循环逻辑。
├── Arena.py            # 让两个网络相互对战以评估性能。
├── nn_model.py         # 使用 PyTorch 定义神经网络架构。
└── nn_wrapper.py       # 神经网络的包装器，用于处理数据转换和预测。
```

## 致谢

  \* 本项目从 [KataGo 论文](https://arxiv.org/abs/1902.10565)中提出的杰出研究和架构概念中获得了重要启发。
  \* 特别感谢 [liemark 的 popucom\_chess\_c-python 项目](https://www.google.com/search?q=https://github.com/liemark/popucom_chess_c-python)，它提供了宝贵的初步灵感。

## 作者

**gumigumi** - 项目开发者

  \* **GitHub:** [https://github.com/Shenyqqq/](https://github.com/Shenyqqq/)
  \* **Hugging Face:** [https://huggingface.co/gumigumi](https://huggingface.co/gumigumi)
