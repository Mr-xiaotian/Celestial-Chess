from .base_ai import BaseAI, logger

# from .deeplearning import ChessModel, DeepLearningAI
from .minimax import MinimaxAI
from .monky import MonkyAI
from .mcts import MCTSAI, MCTSPlusAi
from .monky import MonkyAI
from .evaluate_ai import ai_battle, get_model_score_by_mcts

__all__ = [
    "BaseAI",
    "logger",
    "MinimaxAI",
    "MonkyAI",
    "MCTSAI",
    "MCTSPlusAi",
    "ai_battle",
    "get_model_score_by_mcts",
]