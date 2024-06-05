import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from time import time
from ai import AIAlgorithm, MinimaxAI, MCTSAI
from game.chess_game import ChessGame

def ai_battle(ai_blue: AIAlgorithm, ai_red: AIAlgorithm, test_game: ChessGame = ChessGame()):
    ai_blue_name = ai_blue.__class__.__name__
    ai_red_name = ai_red.__class__.__name__
    print(f'游戏开始！\n蓝方AI: {ai_blue_name}\n红方AI: {ai_red_name}\n')

    first_time = time()
    while True:
        last_time = time()
        color = test_game.get_color()
        if color == 1:
            move = ai_blue.find_best_move(test_game, color)
        else:
            move = ai_red.find_best_move(test_game, color)

        test_game.update_chessboard(*move, color)
        test_game.show_chessboard()
        print(f'第{test_game.step}步: {"蓝方" if color==1 else "红方"} 落子在 {move}')
        print(f'分数: {test_game.get_score()}')
        print(f'用时: {time()-last_time}\n')

        if test_game.is_game_over():
            if test_game.who_is_winner()==1:
                print(f'{"蓝方"+ai_blue_name} 获胜！')
            elif test_game.who_is_winner()==-1:
                print(f'{"红方"+ai_red_name} 获胜！')
            else:
                print('平局！')
            print(f'用时:{time()-first_time}')
            break

if __name__ == '__main__':
    minimax_ai = MinimaxAI(6)
    mcts_ai = MCTSAI(1000)
    test_game = ChessGame((3,2), 2)

    ai_battle(mcts_ai, mcts_ai, test_game)