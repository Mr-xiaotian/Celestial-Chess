import pickle
from typing import Tuple
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class MinimaxAI(AIAlgorithm):
    def __init__(self, depth: int = 3) -> None:
        self.depth = depth

        self.transposition_table = dict()
        self.transposition_table_change = False

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        depth = self.depth
        color = game.get_color()
        logger.debug(f"MinimaxAI is thinking in depth {depth}...\n{game.format_matrix(game.chessboard)}")
        self.load_transposition_table(game)

        if color == 1:
            best_score = float("-inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                score = self.minimax(game, depth, color * -1, float("-inf"), float("inf"))
                game.undo()
                if score > best_score:
                    best_score = score
                    best_move = move

        elif color == -1:
            best_score = float("inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                score = self.minimax(game, depth, color * -1, float("-inf"), float("inf"))
                game.undo()
                if score < best_score:
                    best_score = score
                    best_move = move

        self.save_transposition_table(game)
        game.set_current_win_rate()
        return best_move

    def minimax(self, game: ChessGame, depth: int, color: int, alpha: float, beta: float) -> float:
        self.iterate_time += 1
        logger.debug(f"Iteration {self.iterate_time} in depth {depth}")

        board_key = game.get_board_key()
        if board_key in self.transposition_table and \
            self.transposition_table[board_key]['depth'] >= depth:
            return self.transposition_table[board_key]['score']
        
        if depth == 0 or game.is_game_over():
            score = game.get_score()
            self.update_transposition_table(board_key, {'score': score, 'depth': depth}, game.get_format_board_value())
            return score

        if color == 1:
            max_eval = float("-inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                eval = self.minimax(game, depth - 1, -1, alpha, beta)
                game.undo()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.update_transposition_table(board_key, {'score': max_eval, 'depth': depth}, game.get_format_board_value())
            return max_eval
        elif color == -1:
            min_eval = float("inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                eval = self.minimax(game, depth - 1, 1, alpha, beta)
                game.undo()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.update_transposition_table(board_key, {'score': min_eval, 'depth': depth}, game.get_format_board_value())
            return min_eval
        
    def load_transposition_table(self, game: ChessGame) -> None:
        """
        加载transposition table
        """
        transposition_file = f"transposition_table/transposition_table({game.power}&{game.board_range[0]}_{game.board_range[1]})(sha256).pickle"
        try:
            with open(transposition_file, "rb") as file:
                self.transposition_table = pickle.load(file)
                logger.info(f"load transposition table from {transposition_file}")
        except (FileNotFoundError, EOFError):
            with open(transposition_file, "wb") as file:
                pickle.dump(self.transposition_table, file)

    def update_transposition_table(self, key: str, value: int, format_board_value: str) -> None:
        """
        更新transposition table
        """
        if (
            key not in self.transposition_table
            or self.transposition_table[key]["depth"] < value["depth"]
        ):
            old_value = self.transposition_table.get(key, None)
            self.transposition_table[key] = value
            self.transposition_table_change = True
            logger.info(f"update transposition table: {old_value} -> {value}\n{format_board_value:>20}")
            
    def save_transposition_table(self, game: ChessGame) -> None:
        """
        保存transposition table到文件
        """
        transposition_file = f"transposition_table/transposition_table({game.power}&{game.board_range[0]}_{game.board_range[1]})(sha256).pickle"
        if not self.transposition_table_change:
            self.transposition_table = dict()
            self.transposition_table_change = False
            return
        with open(transposition_file, "wb") as file:
            pickle.dump(self.transposition_table, file)
            self.transposition_table = dict()
            self.transposition_table_change = False
            logger.info(f"save transposition table to {transposition_file}")
