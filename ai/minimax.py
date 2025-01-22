import pickle
from functools import lru_cache
from typing import Tuple
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class MinimaxAI(AIAlgorithm):
    def __init__(self, depth: int, board_state: Tuple[Tuple[int, int], int] = ((5, 5), 2), 
                 debug_mode: bool = False, transposition_mode: bool = True) -> None:
        self.depth = depth
        self.debug_mode = debug_mode
        self.transposition_mode = transposition_mode

        self.load_transposition_table(board_state) if self.transposition_mode else None

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
                    board_key, {'score': score, 'depth': depth}, 
                    game.get_format_board_value() if self.debug_mode else None)
            return score

        if color == 1:
            max_eval = float("-inf")
            for move in game.get_all_moves():
                current_game = game.copy()
                current_game.update_chessboard(*move, color)
                eval = self.minimax(current_game, depth - 1, -1, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            if self.transposition_mode:
                self.update_transposition_table(
                    board_key, {'score': max_eval, 'depth': depth}, 
                    game.get_format_board_value() if self.debug_mode else None)

            return max_eval
        elif color == -1:
            min_eval = float("inf")
            for move in game.get_all_moves():
                current_game = game.copy()
                current_game.update_chessboard(*move, color)
                eval = self.minimax(current_game, depth - 1, 1, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            
            if self.transposition_mode:
                self.update_transposition_table(
                    board_key, {'score': min_eval, 'depth': depth}, 
                    game.get_format_board_value() if self.debug_mode else None)

            return min_eval
        
    def load_transposition_table(self, board_state: Tuple[Tuple[int, int], int]) -> None:
        """
        加载transposition table
        """
        (row_len, col_len), power = board_state
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

    def update_transposition_table(self, key: str, value: int, format_board_value: str = None) -> None:
        """
        更新transposition table
        """
        old_value = self.transposition_table.get(key)
        if old_value is None or old_value["depth"] < value["depth"]:
            old_value = self.transposition_table.get(key, None)
            self.transposition_table[key] = value
            self.transposition_table_change = True
            logger.info(f"update transposition table: {old_value} -> {value}\n{format_board_value:>20}") if self.debug_mode else None
            
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