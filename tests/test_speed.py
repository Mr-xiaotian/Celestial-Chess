import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cProfile
from pathlib import Path
from time import strftime, localtime
from ai import AIAlgorithm, MinimaxAI, MCTSAI
from game.chess_game import ChessGame
import subprocess


'''
ncalls：函数被调用的次数。
tottime：函数自身执行的总时间（不包括子函数的时间）。
percall：每次调用的平均时间（tottime/ncalls）。
cumtime：函数执行的总时间（包括子函数的时间）。
percall：每次调用的平均时间（cumtime/ncalls）。
filename
(function)：文件名、行号和函数名。
'''

def get_output_file_name(target_func):
    now_data = strftime("%Y-%m-%d", localtime())
    now_time = strftime("%H-%M", localtime())

    parent_path = Path(f'profile/{now_data}')
    parent_path.mkdir(parents=True, exist_ok=True)
    
    output_file = f'{parent_path}/{target_func}({now_time}).prof'
    return output_file

def profile_mcts():
    mcts_ai.find_best_move(game)
    mcts_ai.end_model()

def profile_minimax():
    minimax_ai.find_best_move(game)
    minimax_ai.end_model()

game_state = ((5, 5), 2)
game = ChessGame(*game_state)
game.init_cfunc()
mcts_ai = MCTSAI(50000, complate_mode=False)
minimax_ai = MinimaxAI(10, *game_state, complate_mode=True)

target_func = 'profile_mcts'
output_file = get_output_file_name(target_func)

cProfile.run(target_func + '()', output_file)
subprocess.run(['snakeviz', output_file])
# subprocess.run(['gprof2dot', '-f', 'pstats', 'profile/profile_output', '|', 'dot', '-Tpng', '-o', f'profile/profile_results({now_time}).png'])