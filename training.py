import torch
import numpy as np
from ResNet import ResNet
from new_chess import ChessRL
from tictactoe import TicTacToe
from AlphaZeroParallel import AlphaZeroParallel

game = ChessRL()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 4, 64, device)

# temperature ->  eploitation / exploration tradeoff, same role as gamma in Q-learning. High temperature -> more exploration (rd distribution), low temperature -> more exploitation (peak distribution)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
 
args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 10,
    'num_processes': 10,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()

