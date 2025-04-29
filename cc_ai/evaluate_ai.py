from time import time
from tqdm import tqdm
from typing import Tuple, Dict

from cc_game.chess_game import ChessGame
from .ai_algorithm import AIAlgorithm
from .mcts import MCTSAI

def get_model_score_by_mcts(
        test_model: AIAlgorithm, game_state: Tuple[Tuple[int, int], int],
        start_mcts_iter: float = 1e1, end_mcts_iter: float = 1e4, mcts_step: int=10, simulate_num: int = 100
        ) -> Tuple[int, Dict[str, float]]:
    """
    使用MCTS测试AI模型的得分

    :param test_model: 待测试的AI模型
    :param game_state: 游戏状态
    :param start_mcts_iter: MCTS的迭代次数的起始值
    :param end_mcts_iter: MCTS的迭代次数的终止值
    :param mcts_step: MCTS的迭代次数的步长
    :param simulate_num: 模拟的次数
    :return: 返回MCTS的迭代次数和得分字典
    """
    score_dict = dict()

    for mcts_iter in range(int(start_mcts_iter), int(end_mcts_iter), mcts_step):
        win = 0
        test_mcts = MCTSAI(mcts_iter, complete_mode=False)
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

def get_best_c_param(game_state, start_c_param=0.0, test_game_num=1000):
    best_c_param = start_c_param
    c_param_dict = {}
    best_mcts = MCTSAI(100, c_param=best_c_param, complete_mode=False)

    for param in range(0, 11, 1):
        win = 0
        test_mcts = MCTSAI(100, c_param=param/10, complete_mode=False)

        for _ in tqdm(range(test_game_num), desc=f"C param {param/10}"):
            test_game = ChessGame(*game_state)
            test_game.init_history()
            over_game = ai_battle(best_mcts, test_mcts, test_game, display=False)
            winner = over_game.who_is_winner()
            if winner == 1:
                win += 1
            elif winner == 0:
                win += 0.5

        c_param_dict[f"{best_c_param} : {param/10}"] = win
        if win < test_game_num / 2:
            best_c_param = param/10
            best_mcts = MCTSAI(100, c_param=best_c_param, complete_mode=False)
        
    return best_c_param, c_param_dict

def ai_battle(ai_blue: AIAlgorithm, ai_red: AIAlgorithm, test_game: ChessGame = ChessGame(), display=True):
    """
    对两个AI进行对战

    :param ai_blue: 蓝方AI
    :param ai_red: 红方AI
    :param test_game: 游戏对象
    :param display: 是否显示游戏过程
    :return: 游戏对象
    """
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

