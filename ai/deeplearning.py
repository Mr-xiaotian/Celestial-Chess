from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 25)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepLearningAI(AIAlgorithm):
    def __init__(self, model_path):
        self.model = ChessModel().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def find_best_move(self, game: ChessGame):
        board_state = np.array(game.chessboard).reshape(1, 5, 5, 2)
        board_state = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        with torch.no_grad():
            outputs = self.model(board_state)
            move_index = torch.argmax(outputs).item()
            move = (move_index // 5, move_index % 5)
        return move