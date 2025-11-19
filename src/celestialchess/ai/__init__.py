from .base_ai import AIAlgorithm, logger

# from .deeplearning import ChessModel, DeepLearningAI
from .minimax import MinimaxAI
from .monky import MonkyAI
from .mcts import MCTSAI, MCTSPlusAi
from .monky import MonkyAI
from .evaluate_ai import ai_battle, get_model_score_by_mcts
