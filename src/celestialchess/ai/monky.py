import random
from typing import Tuple
from ..chess_game import ChessGame
from .base_ai import BaseAI, logger


class MonkyAI(BaseAI):
    name = "MonkyAI"

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        # 随机选择一个合法移动
        move = game.get_random_move()

        # 胜率其实没用，但保持接口
        game.set_current_win_rate()

        # 随机生成一条猴子意识流
        self._msg = self._build_monky_msg()

        return move

    def _build_monky_msg(self):
        """生成一条混沌的、脑褪速式的猴子发言。"""

        chaos_sounds = [
            "ababa blop blop hrrrraa–waa!",
            "wubba wubba flrrrrt—banana??",
            "gagagaga skrrahh–pitatapita!",
            "wobblob wobblob eeek eeek—BRRRRT!",
            "grrraaa–tikitiki–blop–SKWAA!",
        ]

        attitudes = [
            "我感觉这一步……挺顺眼！理由？我没理由！",
            "我想了零秒钟，这就是天才猴的直觉！",
            "要是这步输了，那肯定不是我的问题，是宇宙的。",
            "哈哈哈下一步会怎样？我不知道我也不想知道！",
            "别问我为什么落这里，我的大脑刚才在想香蕉。",
        ]

        endings = [
            "噗噜噜噜噜噜——啪！",
            "呼嘿嘿嘿——滑滑滑滑！",
            "呜哇哇哇哇——啵叽！",
            "嘶嘶嘶——叽里呱啦啪！",
            "吱哇——啪叽——咕噜！",
        ]

        return random.choice(chaos_sounds + attitudes + endings)


    @property
    def msg(self):
        # 如果还没生成过，给个默认乱语
        return getattr(self, "_msg", "bluh bluh wobba-wobba monky brain offline...")
