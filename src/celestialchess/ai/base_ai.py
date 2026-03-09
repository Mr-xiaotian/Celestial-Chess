from typing import Tuple
import os
from loguru import logger
from time import strftime, localtime

from ..chess_game import ChessGame


# Configure logging
logger.remove()  # remove the default handler
now_time = strftime("%Y-%m-%d", localtime())
os.makedirs("logs", exist_ok=True)
logger.add(
    f"logs/chess_log({now_time}).log",
    format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
    level="INFO",
)


class BaseAI:
    _name = "BaseAI"
    _msg = ""
    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def end_game(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def end_model(self):
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    @property
    def name(self):
        return self._name

    @property
    def msg(self):
        return self._msg
