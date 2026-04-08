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

        return move

    @property
    def msg(self):
        return random.choice(MONKY_CHAOS_SOUNDS + MONKY_ATTITUDES + MONKY_ENDINGS)

    @property
    def deepseek_msg(self):
        context = (
            f"example: {str(self.msg)}"
        )
        return self._request_deepseek_reply(context)
