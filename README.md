[中文](https://github.com/Shenyqqq/pmpmchess-refined/blob/master/README.zh.md)[English](https://github.com/Shenyqqq/pmpmchess-refined/blob/master/README.md)
# Pmpmchess-Refined

Pmpmchess-Refined is a state-of-the-art AI for the game of Pmpmchess, completely re-engineered from the ground up. Inspired by the pioneering architecture of KataGo, this project moves beyond the original AlphaZero framework to deliver much higher performance.

The core of this project features a high-performance C++ engine and a sophisticated, multi-headed neural network. This new architecture accelerates the self-play data generation process by over 30 times, enabling the training of a human-competitive AI in just 4 hours on consumer hardware.

Click here to play right now!
[🤗Hugging Face](https://huggingface.co/spaces/gumigumi/pmpmchess)


## 🎮 How to Play Pmpmchess 

Ever played Gomoku (Five-in-a-Row) or Reversi (Othello)? Pmpmchess is a super cool mini-game from the indie title *Popucom* that cleverly mashes them together! The goal is to both line up your pieces *and* capture territory.

Here's the lowdown on the rules:

* **♟️ The Basics (Board & Turns)**
    The game takes place on a 9x9 board. It's a quick match, lasting exactly 50 turns, which means you and your opponent each get to place 25 pieces.

* **💥 How to Capture**
    Get three of your pieces in an unbroken line (horizontally, vertically, or diagonally) and—BAM!—you capture this line as your territory. But watch out! If your opponent has even one piece in that potential line, your capture is blocked.

* **🤔 Where You Can Move**
    You have two options for placing a piece: any empty square on the board, OR any square inside territory you've already captured. That's right, you can reuse your own turf!

* **🏆 How You Win**
    It's simple! When the game ends (after all 50 turns are up), the player who has captured more squares of territory is crowned the winner.

## Demonstration
![gif](https://raw.githubusercontent.com/Shenyqqq/pmpmchess-refined/master/gif/1.gif)

## Key Architectural Upgrades

This project represents a fundamental overhaul of the original pmpmchessAI. The following enhancements form the backbone of its superior performance and intelligence.

### 1\. High-Performance C++ Engine

The entire game simulation and search logic has been rewritten in C++ to eliminate Python's performance bottlenecks. Key features include:

  * **C++ Game Kernel & MCTS:** The core logic is now native code, drastically speeding up every phase of the self-play loop.
  * **Multi-threading & Batching:** Fully leverages modern multi-core CPUs by processing game states in parallel batches during MCTS, maximizing hardware utilization.
  * **Zobrist Hashing:** Implements an efficient hashing scheme to instantly recognize and retrieve evaluations for previously encountered board states, saving redundant computation.

**Result:** These optimizations deliver a greater than **30x speedup** in self-play data generation compared to the original Python implementation.

### 2\. KataGo-Inspired Neural Network

The AI's "brain" has been upgraded from a simple policy-value network to a more powerful, multi-faceted architecture that captures a deeper understanding of the game.

  * **Multi-Head Architecture:** In addition to predicting the winner (value head), the network now features:
      * **Ownership Head:** Predicts the final owner of each point on the board, allowing the AI to reason about territory.
      * **Auxiliary Score Head:** Estimates the final score difference, providing a more granular evaluation than a simple win/loss prediction.
  * **Composite Loss Function:** The model is trained on a total loss composed of weights from all heads ($w\_{\\pi}, w\_{v}, w\_{score}, w\_{own}$), enabling it to learn strategy, territory, and scoring simultaneously.
  * **Expanded Input Features:** The network input has been enriched from 4 to 13 channels, providing it with a comprehensive view of the game state (e.g., turn history, liberties, etc.), which allows the AI to grasp game rules and nuances implicitly and rapidly.
  * **Advanced Optimizations:** Incorporates modern techniques like Global Pooling inspired by KataGo, enhancing the model's feature extraction capabilities and overall performance.

### 3\. Advanced MCTS & GUI

The search algorithm and user interface have been significantly improved to leverage the new engine and network.

  * **Dual-Objective Search:** The MCTS search is now guided by both the win/loss prediction (value) and the score estimation (auxiliary score), leading to more robust and higher-quality move decisions.
  * **KataGo-Style Playout Cap:** Implements a sophisticated dual-simulation strategy to balance data generation speed and quality. For most moves ($p=0.75$), the AI performs a "fast" simulation with fewer playouts. For a select fraction of moves ($p=0.25$), it executes a "deep" simulation with significantly more playouts. Only these high-quality, deeply analyzed positions are used as training samples.
  * **Enhanced GUI with AI Analysis:** The graphical interface now serves as a powerful analysis tool, providing real-time insights into the AI's thought process and offering better usability:
      * **Live Policy Map:** Visualizes the top moves the AI is considering.
      * **Win/Loss Prediction:** Displays the AI's confidence in winning from the current position.
      * **Score Estimation:** Shows the predicted final score differential.
      * **Ownership Prediction:** Renders a heatmap of the expected final board control.
      * **Interactive Controls:** Includes essential gameplay features like the ability to undo moves.

## Training & Performance

The pmpmchess-refined architecture is not only more powerful but also significantly more efficient.

  * **Rapid Training:** With the current parameters, a strong AI capable of defeating most human players can be trained in approximately **4 hours**.
  * **Scalability:** The system is designed for scalability. To achieve world-class, super-human performance, one can simply increase the neural network size and the number of MCTS simulations.

A note on hyperparameter tuning: Certain parameters, such as `$tempThreshold$` (which controls the transition from exploratory to greedy play) and the loss weights ($w\_{\\pi}, w\_{v}, w\_{score}, w\_{own}$), are highly sensitive. The current values provide a strong baseline, but further tuning may unlock additional performance.

## Installation

### For Players (Recommended)

The easiest way to play is to download a pre-compiled version.

1.  Navigate to the **Releases** page of this repository.
2.  Download the latest compressed package for your operating system.
3.  Unzip the archive and run the executable file (`.exe`).

### For Developers (from Source)

To train your own models or modify the code, you'll need to build the project from the source.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Shenyqqq/pmpmchess-refined.git
    cd pmpmchess-refined
    ```
2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate on Windows
    .\venv\Scripts\activate
    # Activate on macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Compile and Install the C++ Engine:**
    ```bash
    python setup.py build_ext
    python setup.py install
    ```

## Usage

  * **Train a New Model:**
    ```bash
    python main.py
    ```
  * **Play Against an AI:**
    ```bash
    python pit.py
    ```

## Project Structure

```
.
├── cpp/                # Contains all C++ source files for the high-performance engine
│   ├── game.cpp        # Core game logic implementation
│   ├── GameInterface.cpp # Provides an interface between the game state and the NN
│   ├── MCTS.cpp        # C++ implementation of the Monte Carlo Tree Search
│   ├── Zobrist.cpp     # Utility library for Zobrist hashing
│   └── pybind_wrapper.cpp # Exposes C++ functions to Python using pybind11
├── main.py             # Main entry point for training the model. Hyperparameters can be adjusted here.
├── pit.py              # Main entry point for playing against the AI.
├── Coach.py            # Implements the main training loop logic.
├── Arena.py            # Pits two networks against each other to evaluate performance.
├── nn_model.py         # Defines the neural network architecture using PyTorch.
└── nn_wrapper.py       # A wrapper for the NN to handle data transformations and predictions.
```

## Acknowledgements

  * This project drew significant inspiration from the brilliant research and architectural concepts presented in the [KataGo paper](https://arxiv.org/abs/1902.10565).
  * Special thanks to [liemark's popucom\_chess\_c-python project](https://www.google.com/search?q=https://github.com/liemark/popucom_chess_c-python), which provided valuable initial inspiration.

## Author

**gumigumi** - Project Developer

  * **GitHub:** [https://github.com/Shenyqqq/](https://github.com/Shenyqqq/)
  * **Hugging Face:** [https://huggingface.co/gumigumi](https://huggingface.co/gumigumi)
