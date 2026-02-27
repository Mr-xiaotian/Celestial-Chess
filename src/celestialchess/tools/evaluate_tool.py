from time import time
from typing import Tuple, Dict
from celestialflow import TaskManager

from ..chess_game import ChessGame
from ..ai import BaseAI, MCTSAI


class TestAIManager(TaskManager):
    def set_envirement(self, ai_0, ai_1, chess_state):
        self.ai_0 = ai_0
        self.ai_1 = ai_1
        self.chess_state = chess_state

    def get_args(self, obj: object):
        train_game = ChessGame(*self.chess_state)
        train_game.init_history()
        return (self.ai_0, self.ai_1, train_game)

    def process_result_dict(self):
        win = 0
        success_dict = self.get_success_dict()

        for winner in success_dict.values():
            if winner == 1:
                win += 1
            elif winner == 0:
                win += 0.5

        return win 
    

def get_model_score_by_mcts(
    test_model: BaseAI,
    chess_state: Tuple[Tuple[int, int], int],
    execution_mode: str = "process",
    start_mcts_iter: float = 1e1,
    end_mcts_iter: float = 1e4,
    mcts_step: int = 10,
    simulate_num: int = 100,
) -> Tuple[int, Dict[str, float]]:
    """
    使用MCTS测试AI模型的得分

    :param test_model: 待测试的AI模型
    :param chess_state: 游戏状态
    :param execution_mode: 执行模式，支持 "serial" 和 "process"
    :param start_mcts_iter: MCTS的迭代次数的起始值
    :param end_mcts_iter: MCTS的迭代次数的终止值
    :param mcts_step: MCTS的迭代次数的步长
    :param simulate_num: 模拟的次数
    :return: 返回MCTS的迭代次数和得分字典
    """
    score_dict = dict()

    for mcts_iter in range(int(start_mcts_iter), int(end_mcts_iter), mcts_step):
        win = 0
        test_mcts = MCTSAI(mcts_iter, complate_mode=False)
        
        test_ai_manager = TestAIManager(
            battle_for_eval,
            execution_mode=execution_mode,
            worker_limit = 5,
            enable_result_cache=True,
            progress_desc=f"Mcts Iter {mcts_iter}",
            show_progress=True,
        )
        test_ai_manager.set_envirement(test_model, test_mcts, chess_state)

        test_ai_manager.start(range(simulate_num))
        win = test_ai_manager.process_result_dict()

        score_dict[f"{mcts_iter}"] = win / simulate_num
        if sum(score_dict.values()) / len(score_dict) < 0.6:
            break

    test_model.end_model()
    return mcts_iter - 10, score_dict


def get_best_c_param(
    chess_state,
    start_c_param=0.0,
    test_game_num=1000,
    execution_mode="process"
):
    # 历史所有挑战者
    tested_params = []

    # 当前最优参数
    best_c_param = start_c_param

    # 保存所有 pair 的胜负记录
    win_record = {}   # key: (a, b)  value: win_rate(a vs b)

    def play_match(c1, c2):
        """返回 c1 对 c2 的胜率"""
        ai1 = MCTSAI(100, c_param=c1, complate_mode=False)
        ai2 = MCTSAI(100, c_param=c2, complate_mode=False)

        manager = TestAIManager(
            battle_for_eval,
            execution_mode=execution_mode,
            worker_limit=5,
            enable_result_cache=True,
            progress_desc=f"C param {c1} vs {c2}",
            show_progress=True,
        )
        manager.set_envirement(ai1, ai2, chess_state)
        manager.start(range(test_game_num))
        win = manager.process_result_dict()
        return win / test_game_num

    # 主循环：依次测试 0.0 ～ 1.0
    for param in [p / 10 for p in range(11)]:

        # 跳过起始参数
        if param == best_c_param:
            continue

        tested_params.append(param)

        win_rate = play_match(best_c_param, param)
        win_record[(best_c_param, param)] = win_rate

        # challenger 把 best 打败了
        if win_rate < 0.5:
            challenger = param

            # challenger 成为新 best
            best_c_param = challenger

            # 新 best 必须重新挑战所有老参数
            for old_param in tested_params:
                if old_param == challenger:
                    continue

                # 只在没打过的情况下比赛
                if (challenger, old_param) not in win_record:
                    win_back = play_match(challenger, old_param)
                    win_record[(challenger, old_param)] = win_back
                else:
                    win_back = win_record[(challenger, old_param)]

                # 打输了，就必须把 old_param 设为 best，继续验证
                if win_back < 0.5:
                    best_c_param = old_param
                    # challenger 被打下天梯，不用继续验证旧的 challenger vs others
                    break  # 回到主循环（继续未来 param 的挑战）

    return best_c_param, win_record


def battle_for_eval(ai_blue: BaseAI, ai_red: BaseAI, game: ChessGame):
    color = 1

    while not game.is_game_over():

        if color == 1:
            move = ai_blue.find_best_move(game)
        else:
            move = ai_red.find_best_move(game)

        game.update_chessboard(*move, color)

        color *= -1

    return game.who_is_winner()


def ai_battle(
    ai_blue: BaseAI,
    ai_red: BaseAI,
    test_game: ChessGame
):
    """
    对两个AI进行对战

    :param ai_blue: 蓝方AI
    :param ai_red: 红方AI
    :param test_game: 游戏对象
    :return: 游戏对象
    """
    print(
        f"游戏开始！\n蓝方AI: {ai_blue.name}\n红方AI: {ai_red.name}\n"
    )
    ai_blue_name = "蓝方" + ai_blue.name
    ai_red_name = "红方" + ai_red.name
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

        test_game.show_chessboard()
        print(
            f"第{test_game.step}步: {ai_red_name if color==1 else ai_blue_name} 落子在 {test_game.get_current_move()}"
        )
        print(f"获胜概率: {test_game.get_current_win_rate():.2%}")
        print(f"分数: {test_game.get_score()}")
        print(f"用时: {time()-last_time:.2f}s\n")

        if test_game.is_game_over():
            ai_blue.end_game()
            ai_red.end_game()

            ai_blue.end_model()
            ai_red.end_model()

            winner = test_game.who_is_winner()
            if winner == 1:
                print(f"{ai_blue_name} 获胜！")
            elif winner == -1:
                print(f"{ai_red_name} 获胜！")
            else:
                print("平局！")
            print(f"总用时:{time()-first_time:.2f}s")
            break

    return test_game
