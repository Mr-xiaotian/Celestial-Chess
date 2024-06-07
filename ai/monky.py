import random
from typing import Tuple
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame

class MonkyAI(AIAlgorithm):
    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        # 随机选择一个合法的移动
        moves_list = game.get_all_moves()
        possible_moves = random.choice(moves_list)
        game.set_current_win_rate()

        return possible_moves