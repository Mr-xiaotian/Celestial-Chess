import json
import random
from pathlib import Path

# from functools import lru_cache
from typing import Tuple

from ..chess_game import ChessGame
from .base_ai import BaseAI, logger


class MinimaxAI(BaseAI):
    def __init__(self, depth: int, log_mode: bool = False) -> None:
        self.name = f"MinimaxAI"

        self.depth = depth
        self.log_mode = log_mode
        self.transposition_mode = False

    def set_transposition_mode(
        self,
        chess_state: Tuple[Tuple[int, int], int] = ((5, 5), 2),
        transposition_path: str = "./transposition_table",
    ) -> None:
        """
        启用 transposition table 模式，并指定存储路径
        :param transposition_path: 存储 transposition table 的文件夹路径
        """
        self.transposition_mode = True
        self.transposition_path = transposition_path
        self.load_transposition_table(chess_state)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        depth = self.depth
        color = game.get_color()
        if self.log_mode:
            logger.debug(
                f"MinimaxAI is thinking in depth {depth}...\n{game.get_format_board()}"
            )

        best_score = float("-inf") if color == 1 else float("inf")
        for move in game.get_all_moves():
            current_game = game.copy()
            current_game.update_chessboard(*move, color)
            score = self.minimax(
                current_game, depth, -color, float("-inf"), float("inf")
            )
            if (color == 1 and score > best_score) or (
                color == -1 and score < best_score
            ):
                best_score = score
                best_move = move

        # Minimax 没有真正 win_rate，这里保持接口一致但用不到
        game.set_current_win_rate()

        # ✦ 新增：生成带人格的 Minimax 消息 ✦
        self._msg = self._build_minimax_msg(best_score, color, depth, self.iterate_time)

        return best_move

    # @lru_cache(maxsize=None)
    def minimax(
        self, game: ChessGame, depth: int, color: int, alpha: float, beta: float
    ) -> float:
        self.iterate_time += 1

        if self.log_mode:
            logger.debug(f"Iteration {self.iterate_time} in depth {depth}")

        if self.transposition_mode:
            board_key = game.get_board_key()
            if (
                board_key in self.transposition_table
                and self.transposition_table[board_key]["depth"] >= depth
            ):
                return self.transposition_table[board_key]["score"]

        if depth == 0 or game.is_game_over():
            score = game.get_score()
            if self.transposition_mode:
                self.update_transposition_table(
                    board_key,
                    score,
                    depth,
                    game.get_format_board_value() if self.log_mode else None,
                )
            return score

        # 初始化最大化或最小化的评估值
        best_eval = float("-inf") if color == 1 else float("inf")
        comparison_func = max if color == 1 else min
        alpha_beta_update = max if color == 1 else min

        # 遍历所有可能的移动
        for move in game.get_all_moves():
            current_game = game.copy()
            current_game.update_chessboard(*move, color)

            eval = self.minimax(current_game, depth - 1, -color, alpha, beta)

            best_eval = comparison_func(best_eval, eval)
            if color == 1:
                alpha = alpha_beta_update(alpha, eval)
            else:
                beta = alpha_beta_update(beta, eval)

            # Alpha-Beta 剪枝
            if beta <= alpha:
                break

        # 更新置换表
        if self.transposition_mode:
            self.update_transposition_table(
                board_key,
                best_eval,
                depth,
                game.get_format_board_value() if self.log_mode else None,
            )

        return best_eval

    def load_transposition_table(
        self, chess_state: Tuple[Tuple[int, int], int]
    ) -> None:
        """
        加载transposition table
        """
        (row_len, col_len), power = chess_state
        self.transposition_file = Path(
            f"{self.transposition_path}/transposition_table({row_len}_{col_len}&{power})(sha256).json"
        )
        self.transposition_file.parent.mkdir(parents=True, exist_ok=True)

        self.transposition_table = {}
        self.transposition_table_change = False

        try:
            if self.transposition_file.exists():
                with open(self.transposition_file, "r", encoding="utf-8") as file:
                    self.transposition_table = json.load(file)
                    (
                        logger.info(
                            f"Loaded transposition table: {self.transposition_file}"
                        )
                        if self.log_mode
                        else None
                    )
            else:
                # 创建空 json
                with open(self.transposition_file, "w", encoding="utf-8") as file:
                    json.dump({}, file, indent=4)
        except json.JSONDecodeError:
            logger.warning(
                f"Transposition table corrupt, resetting: {self.transposition_file}"
            )
            with open(self.transposition_file, "w", encoding="utf-8") as file:
                json.dump({}, file, indent=4)

    def update_transposition_table(
        self, key: str, score: int, depth: int, format_board_value: str = None
    ) -> None:
        """
        更新transposition table
        """
        old_value = self.transposition_table.get(key)
        new_value = {"score": score, "depth": depth}
        if old_value is None or old_value["depth"] < new_value["depth"]:
            old_value = self.transposition_table.get(key, None)
            self.transposition_table[key] = new_value
            self.transposition_table_change = True
            (
                logger.info(
                    f"Update transposition table: {old_value} -> {new_value}\n{format_board_value:>20}"
                )
                if self.log_mode
                else None
            )

    def save_transposition_table(self) -> None:
        """
        保存transposition table到文件
        """
        if not self.transposition_table_change:
            self.transposition_table = {}
            self.transposition_table_change = False
            return

        with open(self.transposition_file, "w", encoding="utf-8") as file:
            json.dump(self.transposition_table, file)

        (
            logger.info(f"Saved transposition table to {self.transposition_file}")
            if self.log_mode
            else None
        )

        self.transposition_table = {}
        self.transposition_table_change = False

    def _build_minimax_msg(self, best_score: float, color: int, depth: int, iters: int) -> str:
        """
        根据评分生成带人格的 Minimax 文本。
        Minimax 走冷静毒舌、学霸式“我计算过了”的风格。
        """
        # 从当前行动方视角看：正数 = 我方占优
        signed_score = best_score if color == 1 else -best_score

        # 分档语气
        if signed_score < -5:
            mood = (
                "计算完成。\n"
                "推断结果：我方局势已接近不可逆转的崩坏。\n"
                "我现在只是在思考：怎么输得不那么丢脸。"
            )
        elif signed_score < -1:
            mood = (
                "分析结束。\n"
                "当前态势对我方明显不利。\n"
                "这不是悲观，只是数学。"
            )
        elif -1 <= signed_score <= 1:
            mood = (
                "评估结果：大致均势。\n"
                "不过你每下一步跟‘随机退火’一样混乱，\n"
                "让我很难继续维持所谓的“平衡”。"
            )
        elif signed_score <= 5:
            mood = (
                "局面评估完成。\n"
                "我方略占优势。\n"
                "这不是运气好，而是因为我在思考，而你在祈祷。"
            )
        else:
            mood = (
                "搜索结束。\n"
                "我方胜势极大，几乎无需再计算。\n"
                "接下来只是策略演示，不是对弈。"
            )

        meta = (
            f"搜索深度 = {depth}, 节点数 ≈ {iters}。"
        )

        if random.random() < 0.8:
            self.name = f"MinimaxAI"
            return mood
        else:
            self.name = "【Minimax 报告】"
            return meta

    @property
    def msg(self):
        return self._msg

    def end_game(self):
        pass

    def end_model(self):
        self.save_transposition_table() if self.transposition_mode else None
