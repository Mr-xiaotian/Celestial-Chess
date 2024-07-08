import sys
import os
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)

from time import time
from tqdm import tqdm
from ai import AIAlgorithm, MinimaxAI, MCTSAI
from ai.deeplearning import DeepLearningAI
from game.chess_game import ChessGame

def get_model_score_by_mcts(test_model: AIAlgorithm, game_state):
    score_dict = dict()
    for mcts_iter in tqdm(range(10, 10000, 10)):
        win = 0
        test_mcts = MCTSAI(mcts_iter, complate_mode=False)
        for _ in range(100):
            test_game = ChessGame(*game_state)
            test_game.init_history()
            over_game = ai_battle(test_model, test_mcts, test_game, display=False)
            winner = over_game.who_is_winner()
            if winner == 1:
                win += 1
            elif winner == 0:
                win += 0.5
            # test_model.end_battle()
            # test_mcts.end_battle()

        score_dict[f"{mcts_iter}"] = win / 100
        if (10 * sum(score_dict.values()))/mcts_iter < 0.6:
            return mcts_iter - 10, score_dict
        
    return score_dict

def ai_battle(ai_blue: AIAlgorithm, ai_red: AIAlgorithm, test_game: ChessGame = ChessGame(), display=True):
    print(f'游戏开始！\n蓝方AI: {ai_blue.__class__.__name__}\n红方AI: {ai_red.__class__.__name__}\n') if display else None
    ai_blue_name = "蓝方" + ai_blue.__class__.__name__
    ai_red_name = "红方" + ai_red.__class__.__name__
    color = 1

    first_time = time()
    while True:
        last_time = time()
        if color == 1:
            move = ai_blue.find_best_move(test_game)
        else:
            move = ai_red.find_best_move(test_game)

        test_game.update_chessboard(*move, color)
        test_game.update_history(*move)
        color *= -1

        if display:
            test_game.show_chessboard()
            print(f'第{test_game.step}步: {ai_blue_name if color==1 else ai_red_name} 落子在 {test_game.get_current_move()}')
            print(f'获胜概率: {test_game.get_current_win_rate():.2%}')
            print(f'分数: {test_game.get_score()}')
            print(f'用时: {time()-last_time:.2f}s\n')

        if test_game.is_game_over():
            ai_blue.end_game()
            ai_red.end_game()

            if not display:
                break
            winner = test_game.who_is_winner()
            if winner==1:
                print(f'{ai_blue_name} 获胜！')
            elif winner==-1:
                print(f'{ai_red_name} 获胜！')
            else:
                print('平局！')
            print(f'总用时:{time()-first_time:.2f}s')
            break

    return test_game

if __name__ == '__main__':
    chess_state = ((5, 5), 2)
    test_game = ChessGame(*chess_state)
    test_game.init_cfunc()
    test_game.init_history()

    # minimax_ai = MinimaxAI(5, *chess_state, complate_mode=True)
    mcts_ai_0 = MCTSAI(10000, complate_mode=True)
    mcts_ai_1 = MCTSAI(1000, complate_mode=True)
    # deeplearning_ai = DeepLearningAI('ai/models/dl_model(06-22-21-18)(136090)(32-64-128-256).pth', complate_mode=True)

    # policy_model = DeepLearningAI('ai/models/dl_model(06-22-21-18)(136090)(32-64-128-256).pth', complate_mode=False)
    # mcts_ai_policy_0 = MCTSAI(1000, c_param=0.5, policy_net=policy_model, complate_mode=True)

    ai_battle(mcts_ai_0, mcts_ai_0, test_game)