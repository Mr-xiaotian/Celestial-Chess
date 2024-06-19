import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cProfile
import pstats
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
def profile_mcts():
    mcts_ai.find_best_move(game)

game = ChessGame(board_range=(5, 5), power=2)
game.init_cfunc()
mcts_ai = MCTSAI(50000, flag=True)

now_time = strftime("%m-%d-%H-%M", localtime())
output_file = f'profile/profile_output({now_time}).prof'
cProfile.run('profile_mcts()', output_file)

# with open(f'profile/profile_results({now_time}).txt', 'w') as f:
#     stats = pstats.Stats(output_file, stream=f)
#     stats.sort_stats('cumulative')
#     stats.print_stats()

subprocess.run(['snakeviz', output_file])
# subprocess.run(['gprof2dot', '-f', 'pstats', 'profile/profile_output', '|', 'dot', '-Tpng', '-o', f'profile/profile_results({now_time}).png'])