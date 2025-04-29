from typing import Tuple

from cc_game.chess_game import ChessGame
from .ai_algorithm import AIAlgorithm, logger

class MonkyAI(AIAlgorithm):
    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        # 随机选择一个合法的移动
        possible_moves = game.get_random_move()
        # 设置当前胜率
        game.set_current_win_rate()

        return possible_moves