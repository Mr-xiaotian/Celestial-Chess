from celestialchess.ai.deeplearning import DeepLearningAI
from celestialchess import MCTSAI
from celestialchess.tools.evaluate_tool import get_model_score_by_mcts

chess_state = ((5,5), 2)

if __name__ == "__main__":
    # policy_model = DeepLearningAI('models/dl_model(06-22-21-18)(136090)(32-64-128-256).pth')
    # model = DeepLearningAI(model_path)
    # get_model_score_by_mcts(model, chess_state)

    model = DeepLearningAI(r"models\2024-06-28\dl_model(06-28-15-00)(136090).pth")
    get_model_score_by_mcts(model, chess_state)