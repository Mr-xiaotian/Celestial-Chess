import random
from typing import Tuple
from ..chess_game import ChessGame
from .base_ai import BaseAI, logger
from .dialogues import MONKY_ATTITUDES, MONKY_CHAOS_SOUNDS, MONKY_ENDINGS


class MonkyAI(BaseAI):
    _name = "MonkyAI"

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        # 随机选择一个合法移动
        move = game.get_random_move()

        # 胜率其实没用，但保持接口
        game.set_current_win_rate()

        # 随机生成一条猴子意识流
        self._build_monky_msg()

        return move

    def _build_monky_msg(self):
        self._msg = random.choice(MONKY_CHAOS_SOUNDS + MONKY_ATTITUDES + MONKY_ENDINGS)

