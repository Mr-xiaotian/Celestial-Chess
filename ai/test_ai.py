import sys
import os
project_root = os.path.abspath(".")
sys.path.insert(0, project_root)

from time import time
from tqdm import tqdm
from typing import Tuple, Dict
from ai import AIAlgorithm, MinimaxAI, MCTSAI
from ai.deeplearning import DeepLearningAI
from game.chess_game import ChessGame

def get_model_score_by_mcts(
        test_model: AIAlgorithm, game_state: Tuple[Tuple[int, int], int],
        start_mcts_iter: float = 1e1, end_mcts_iter: float = 1e4, mcts_step: int=10, simulate_num: int = 100
        ) -> Tuple[int, Dict[str, float]]:
    
    mcts_iter = 10
    score_dict = dict()

    for mcts_iter in range(int(start_mcts_iter), int(end_mcts_iter), mcts_step):
        win = 0
        test_mcts = MCTSAI(mcts_iter, complate_mode=False)
        for _ in tqdm(range(simulate_num), desc=f"Mcts Iter {mcts_iter}"):
            test_game = ChessGame(*game_state)
            test_game.init_history()
            
            over_game = ai_battle(test_model, test_mcts, test_game, display=False)
            winner = over_game.who_is_winner()
            if winner == 1:
                win += 1
            elif winner == 0:
                win += 0.5
        test_mcts.end_model()

        score_dict[f"{mcts_iter}"] = win / simulate_num
        if sum(score_dict.values())/len(score_dict) < 0.6:
            break
        
    test_model.end_model()
    return mcts_iter - 10, score_dict

def ai_battle(ai_blue: AIAlgorithm, ai_red: AIAlgorithm, test_game: ChessGame = ChessGame(), display=True):
    if display:
        print(f'游戏开始！\n蓝方AI: {ai_blue.__class__.__name__}\n红方AI: {ai_red.__class__.__name__}\n')
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
            print(f'第{test_game.step}步: {ai_red_name if color==1 else ai_blue_name} 落子在 {test_game.get_current_move()}')
            print(f'获胜概率: {test_game.get_current_win_rate():.2%}')
            print(f'分数: {test_game.get_score()}')
            print(f'用时: {time()-last_time:.2f}s\n')

        if test_game.is_game_over():
            ai_blue.end_game()
            ai_red.end_game()

            if not display:
                break
            ai_blue.end_model()
            ai_red.end_model()

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
    chess_state = ((19, 19), 4)
    test_game = ChessGame(*chess_state)
    test_game.init_cfunc()
    test_game.init_history()

    # minimax_ai = MinimaxAI(5, *chess_state, complate_mode=True)
    mcts_ai_0 = MCTSAI(10000, complate_mode=True)
    mcts_ai_1 = MCTSAI(10000, complate_mode=True)
    deeplearning_ai = DeepLearningAI('ai/models/dl_model(06-28-15-00)(136090).pth', complate_mode=True)

    # policy_model = DeepLearningAI('ai/models/dl_model(06-22-21-18)(136090)(32-64-128-256).pth', complate_mode=False)
    # mcts_ai_policy_0 = MCTSAI(1000, c_param=0.5, policy_net=policy_model, complate_mode=True)

    ai_battle(mcts_ai_0, mcts_ai_1, test_game)
    #Microsoft_767292