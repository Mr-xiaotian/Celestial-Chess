from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class DeepLearningAI(AIAlgorithm):
    def __init__(self, model_path):
        self.model = ChessModel().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def find_best_move(self, game: ChessGame):
        board_state = np.array(game.get_board_value()).reshape(1, 5, 5, 2)
        board_state = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        with torch.no_grad():
            outputs = self.model(board_state)
            move_index = torch.argmax(outputs).item()
            move = (move_index // 5, move_index % 5)
        return move