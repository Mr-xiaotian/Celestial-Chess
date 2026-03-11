import json
import random
from enum import IntEnum
from pathlib import Path

from typing import Tuple

from ..chess_game import ChessGame
from .base_ai import BaseAI, logger
from .dialogues import MINIMAX_DIALOGUES


class AlphaBetaFlag(IntEnum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


class MinimaxAI(BaseAI):
    _name = "MinimaxAI" if random.random() < 0.8 else "【Minimax 报告】"
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
        :param chess_state: 初始棋盘状态
        :param transposition_path: 存储 transposition table 的文件夹路径
        """
        self.transposition_mode = True
        self.transposition_path = transposition_path
        self.load_transposition_table(chess_state)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        self.depth = self.depth
        self.color = game.get_color()
        if self.log_mode:
            logger.debug(
                f"MinimaxAI is thinking in depth {self.depth}...\n{game.get_format_board()}"
            )

        self.best_score = float("-inf") if self.color == 1 else float("inf")
        for move in game.get_all_moves():
            current_game = game.copy()
            current_game.update_chessboard(*move, self.color)
            score = self.minimax(
                current_game, self.depth, -self.color, float("-inf"), float("inf")
            )
            if (self.color == 1 and score > self.best_score) or (
                self.color == -1 and score < self.best_score
            ):
                self.best_score = score
                best_move = move

        return best_move

    def minimax(
        self, game: ChessGame, depth: int, color: int, alpha: float, beta: float
    ) -> float:
        
        self.iterate_time += 1
        old_alpha, old_beta = alpha, beta

        if self.log_mode:
            logger.debug(f"Iteration {self.iterate_time} in depth {depth}")

        # ===== ① TT 查询 =====
        board_key = game.get_board_key() if self.transposition_mode else None
        if self.transposition_mode and board_key in self.transposition_table:
            entry = self.transposition_table[board_key]
            if entry["depth"] >= depth:
                flag = entry["flag"]
                score = entry["score"]

                if flag == AlphaBetaFlag.EXACT:   
                    return score
                elif flag == AlphaBetaFlag.LOWER: 
                    alpha = max(alpha, score)
                elif flag == AlphaBetaFlag.UPPER:
                    beta = min(beta, score)

                if alpha >= beta:
                    return score

        # ===== ② 叶子节点 =====
        if depth == 0 or game.is_game_over():
            score = game.get_score()
            if self.transposition_mode:
                self.update_transposition_table(board_key, score, depth, AlphaBetaFlag.EXACT)
            return score

        # ===== ③ 搜索子节点 =====
        best_eval = float("-inf") if color == 1 else float("inf")

        for move in game.get_all_moves():
            g2 = game.copy()
            g2.update_chessboard(*move, color)

            eval = self.minimax(g2, depth - 1, -color, alpha, beta)

            if color == 1:
                best_eval = max(best_eval, eval)
                alpha = max(alpha, eval)
            else:
                best_eval = min(best_eval, eval)
                beta = min(beta, eval)

            if beta <= alpha:
                break

        # ===== ④ 判断 flag 类型 =====
        if best_eval <= old_alpha:
            flag = AlphaBetaFlag.UPPER
        elif best_eval >= old_beta:
            flag = AlphaBetaFlag.LOWER
        else:
            flag = AlphaBetaFlag.EXACT

        # ===== ⑤ 写入 TT =====
        if self.transposition_mode:
            self.update_transposition_table(board_key, best_eval, depth, flag)

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
        self, key: str, score: int, depth: int, flag: AlphaBetaFlag
    ) -> None:
        """
        更新transposition table
        """
        old_value = self.transposition_table.get(key, None)
        new_value = {"score": score, "depth": depth, "flag": flag}

        if old_value is None or old_value["depth"] < new_value["depth"]:
            self.transposition_table[key] = new_value
            self.transposition_table_change = True
            (
                logger.info(
                    f"Update transposition table: {old_value} -> {new_value}"
                )
                if self.log_mode
                else None
            )

    def save_transposition_table(self) -> None:
        """
        保存transposition table到文件
        """
        if not self.transposition_table_change:
            return

        with open(self.transposition_file, "w", encoding="utf-8") as file:
            json.dump(self.transposition_table, file)

        (
            logger.info(f"Saved transposition table to {self.transposition_file}")
            if self.log_mode
            else None
        )

        self.transposition_table_change = False
    
    @property
    def msg(self) -> str:
        # 从当前行动方视角看：正数 = 我方占优
        signed_score = self.best_score if self.color == 1 else -self.best_score
        
        # 根据 score 选择分组
        if signed_score < -5:
            selected = "crushed"
        elif signed_score < -1:
            selected = "losing"
        elif -1 <= signed_score <= 1:
            selected = "even"
        elif signed_score <= 5:
            selected = "winning"
        else:
            selected = "crushing"

        mood = random.choice(MINIMAX_DIALOGUES[selected])

        meta = (
            f"搜索深度 = {self.depth}, 节点数 ≈ {self.iterate_time}, 评估 = {signed_score:.3f}。"
        )

        if self._name == "MinimaxAI":
            return mood
        elif self._name == "【Minimax 报告】":
            return meta

    def end_game(self):
        pass

    def end_model(self):
        self.save_transposition_table() if self.transposition_mode else None
