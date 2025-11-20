import json
from pathlib import Path

# from functools import lru_cache
from typing import Tuple

from ..chess_game import ChessGame
from .base_ai import AIAlgorithm, logger


class MinimaxAI(AIAlgorithm):
    def __init__(self, depth: int, log_mode: bool = False) -> None:
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
                f"MinimaxAI is thinking in depth {depth}...\n{game.format_matrix(game.chessboard)}"
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

        game.set_current_win_rate()
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
            json.dump(self.transposition_table, file, indent=4)

        (
            logger.info(f"Saved transposition table to {self.transposition_file}")
            if self.log_mode
            else None
        )

        self.transposition_table = {}
        self.transposition_table_change = False

    def end_game(self):
        pass

    def end_model(self):
        self.save_transposition_table() if self.transposition_mode else None
