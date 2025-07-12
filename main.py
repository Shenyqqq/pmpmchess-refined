from Coach import Coach
from nn_wrapper import NNetWrapper
from nn_interface import Game
from nn_model import NNet
from utils import dotdict

# A dictionary of hyperparameters
args = dotdict({
    'numIters': 10,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 20,  #
    'updateThreshold': 0.55,  # During arena playoff, new model should win by this fraction to be accepted.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 300,  # Number of MCTS simulations per move.
    'arenaCompare': 40,  # Number of games to play during arena part pitting.
    'cpuct': 2,
    'dirichletAlpha': 0.1,
    'epsilon': 0.25,  # Fraction of noise to add to the root policy

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x8_100checkpoints', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    # --- Batching Hyperparameters ---
    'numParallelGames': 64,  # Number of games to play in parallel during self-play.
    'mcts_batch_size': 32,  # Batch size for MCTS's internal NN predictions.

    # --- Neural Network Hyperparameters ---
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 13,  # Must match the game interface
    'num_filters': 128,
    'num_residual_blocks': 8,
    'factor_winloss': 1.0,  # Weight for the win/loss utility in MCTS
    'w_spdf': 0.1,  # Weight for the score distribution loss
})


def main():
    # Initialize the game
    g = Game(n=9)

    # Initialize the neural network
    # The nnet_class is passed so the wrapper can instantiate it
    nnet = NNetWrapper(g, NNet, args)

    if args['load_model']:
        print("Loading checkpoint...")
        nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    else:
        print("Starting from scratch...")

    # Initialize the Coach
    c = Coach(g, nnet, args)

    if args['load_model']:
        print("Loading train examples...")
        c.load_train_examples()

    # Start the training loop
    print("Starting the training process...")
    c.learn()


if __name__ == "__main__":
    main()

