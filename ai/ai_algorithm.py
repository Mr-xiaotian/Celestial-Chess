from typing import Tuple
from loguru import logger
from time import strftime, localtime
from game.chess_game import ChessGame


# Configure logging
logger.remove()  # remove the default handler
now_time = strftime("%Y-%m-%d", localtime())
logger.add(f"logs/chess_manager({now_time}).log", 
           format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
           level="INFO")


class AIAlgorithm:
    def find_best_move(self, game: ChessGame, color: int) -> Tuple[int, int]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    