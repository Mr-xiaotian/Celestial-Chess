import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from time import time
from ai import AIAlgorithm, MinimaxAI, MCTSAI
from game.chess_game import ChessGame

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

        if display:
            test_game.update_history(*move)
            test_game.show_chessboard()
            print(f'第{test_game.step}步: {ai_blue_name if color==1 else ai_red_name} 落子在 {move}')
            print(f'获胜概率: {test_game.get_current_win_rate():.2%}')
            print(f'分数: {test_game.get_score()}')
            print(f'用时: {time()-last_time:.2f}s\n')

        color *= -1

        if test_game.is_game_over():
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
    test_game = ChessGame((5, 5), 2)
    test_game.init_cfunc()
    test_game.init_history()

    minimax_ai = MinimaxAI(5)
    mcts_ai_0 = MCTSAI(10000, flag=True)
    mcts_ai_1 = MCTSAI(1000, flag=True)

    ai_battle(mcts_ai_0, minimax_ai, test_game)