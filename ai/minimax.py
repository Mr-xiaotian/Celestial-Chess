import pickle
from functools import lru_cache
from typing import Tuple
from game.chess_game import ChessGame
from .ai_algorithm import AIAlgorithm, logger


class MinimaxAI(AIAlgorithm):
    def __init__(self, depth: int, chess_state: Tuple[Tuple[int, int], int] = ((5, 5), 2), 
                 debug_mode: bool = False, transposition_mode: bool = True) -> None:
        self.depth = depth
        self.debug_mode = debug_mode
        self.transposition_mode = transposition_mode

        self.load_transposition_table(chess_state) if self.transposition_mode else None

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        depth = self.depth
        color = game.get_color()
        if self.debug_mode:
            logger.debug(f"MinimaxAI is thinking in depth {depth}...\n{game.format_matrix(game.chessboard)}")
        
        best_score = float("-inf") if color == 1 else float("inf")
        for move in game.get_all_moves():
            current_game = game.copy()
            current_game.update_chessboard(*move, color)
            score = self.minimax(current_game, depth, -color, float("-inf"), float("inf"))
            if (color == 1 and score > best_score) or (color == -1 and score < best_score):
                best_score = score
                best_move = move

        game.set_current_win_rate()
        return best_move

    # @lru_cache(maxsize=None)
    def minimax(self, game: ChessGame, depth: int, color: int, alpha: float, beta: float) -> float:
        self.iterate_time += 1

        if self.debug_mode:
            logger.debug(f"Iteration {self.iterate_time} in depth {depth}")

        if self.transposition_mode:
            board_key = game.get_board_key()
            if board_key in self.transposition_table and \
                self.transposition_table[board_key]['depth'] >= depth:
                return self.transposition_table[board_key]['score']
        
        if depth == 0 or game.is_game_over():
            score = game.get_score()
            if self.transposition_mode:
                self.update_transposition_table(
                    board_key, score, depth, 
                    game.get_format_board_value() if self.debug_mode else None)
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
                board_key, best_eval, depth,
                game.get_format_board_value() if self.debug_mode else None
            )

            return best_eval
        
    def load_transposition_table(self, chess_state: Tuple[Tuple[int, int], int]) -> None:
        """
        加载transposition table
        """
        (row_len, col_len), power = chess_state
        self.transposition_file = f"./transposition_table/transposition_table({row_len}_{col_len}&{power})(sha256).pickle"
        
        self.transposition_table = {}
        self.transposition_table_change = False
        try:
            with open(self.transposition_file, "rb") as file:
                self.transposition_table = pickle.load(file)
                logger.info(f"load transposition table from {self.transposition_file}") if self.debug_mode else None
        except (FileNotFoundError, EOFError):
            with open(self.transposition_file, "wb") as file:
                pickle.dump(self.transposition_table, file)

    def update_transposition_table(self, key: str, score: int, depth: int, format_board_value: str = None) -> None:
        """
        更新transposition table
        """
        old_value = self.transposition_table.get(key)
        new_value = {'score': score, 'depth': depth}
        if old_value is None or old_value["depth"] < new_value["depth"]:
            old_value = self.transposition_table.get(key, None)
            self.transposition_table[key] = new_value
            self.transposition_table_change = True
            logger.info(f"update transposition table: {old_value} -> {new_value}\n{format_board_value:>20}") if self.debug_mode else None
            
    def save_transposition_table(self) -> None:
        """
        保存transposition table到文件
        """
        if not self.transposition_table_change:
            self.transposition_table = {}
            self.transposition_table_change = False
            return
        with open(self.transposition_file, "wb") as file:
            pickle.dump(self.transposition_table, file)
            self.transposition_table = {}
            self.transposition_table_change = False
            logger.info(f"save transposition table to {self.transposition_file}") if self.debug_mode else None

    def end_game(self):
        pass
        
    def end_model(self):
        self.save_transposition_table() if self.transposition_mode else None