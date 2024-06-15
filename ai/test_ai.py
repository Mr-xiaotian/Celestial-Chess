import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from time import time
from ai import AIAlgorithm, MinimaxAI, MCTSAI
from game.chess_game import ChessGame

def ai_battle(ai_blue: AIAlgorithm, ai_red: AIAlgorithm, test_game: ChessGame = ChessGame(), display=True):
    ai_blue_name = ai_blue.__class__.__name__
    ai_red_name = ai_red.__class__.__name__
    print(f'游戏开始！\n蓝方AI: {ai_blue_name}\n红方AI: {ai_red_name}\n') if display else None

    first_time = time()
    while True:
        last_time = time()
        color = test_game.get_color()
        if color == 1:
            move = ai_blue.find_best_move(test_game)
        else:
            move = ai_red.find_best_move(test_game)

        test_game.update_chessboard(*move, color)
        test_game.show_chessboard() if display else None
        print(f'第{test_game.step}步: {"蓝方" if color==1 else "红方"} 落子在 {move}') if display else None
        print(f'获胜概率: {test_game.get_current_win_rate():.2%}') if display else None
        print(f'分数: {test_game.get_score()}') if display else None
        print(f'用时: {time()-last_time:.2f}s\n') if display else None

        if test_game.is_game_over():
            if test_game.who_is_winner()==1:
                print(f'{"蓝方"+ai_blue_name} 获胜！') if display else None
            elif test_game.who_is_winner()==-1:
                print(f'{"红方"+ai_red_name} 获胜！') if display else None
            else:
                print('平局！') if display else None
            print(f'总用时:{time()-first_time:.2f}s') if display else None
            break

    return test_game

if __name__ == '__main__':
    minimax_ai = MinimaxAI(3)
    mcts_ai_0 = MCTSAI(10000, flag=True)
    mcts_ai_1 = MCTSAI(10000, flag=False)
    test_game = ChessGame((5, 5), 2)

    ai_battle(mcts_ai_0, mcts_ai_0, test_game)