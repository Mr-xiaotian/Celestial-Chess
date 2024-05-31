from time import time
from typing import List, Tuple
from chess_game import ChessGame
from loguru import logger

# Configure logging
logger.remove()  # remove the default handler
logger.add(f"logs/thread_manager.log", format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")

class AIAlgorithm:
    def find_best_move(self, game: ChessGame, color: int, depth: int) -> Tuple[int, int]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class MinimaxAI(AIAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    def find_best_move(self, game: ChessGame, color: int, depth: int = 3) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        logger.info(f"MinimaxAI is thinking in depth {depth}...")

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

        game.save_transposition_table()

        return best_move

    def minimax(self, game: ChessGame, depth: int, color: int, alpha: float, beta: float) -> float:
        self.iterate_time += 1
        logger.info(f"Iteration {self.iterate_time} in depth {depth}")

        board_key = game.get_board_key()
        if board_key in game.transposition_table and \
            game.transposition_table[board_key]['depth'] >= depth:
            return game.transposition_table[board_key]['score']
        
        if depth == 0 or game.is_game_over():
            score = game.get_score()
            game.update_transposition_table(board_key, {'score': score, 'depth': depth})
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
            game.update_transposition_table(board_key, {'score': max_eval, 'depth': depth})
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
            game.update_transposition_table(board_key, {'score': min_eval, 'depth': depth})
            return min_eval
        
    def test_time(self, depth = 3):
        test_game = ChessGame()

        first_time = time()
        while True:
            last_time = time()
            color = 1 if test_game.step%2==0 else -1
            move = self.find_best_move(test_game, color, depth)
            print(time()-last_time)
            test_game.update_chessboard(*move, color)
            test_game.show_chessboard()
            print()
            # sleep(0)
            if test_game.is_game_over():
                print(f'总分数:{test_game.get_score()}\n用时:{time()-first_time}')
                break


if __name__ == '__main__':
    minimax_ai = MinimaxAI()
    minimax_ai.test_time(20)