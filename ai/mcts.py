from __future__ import annotations
import numpy as np
from typing import Tuple, List
from .ai_algorithm import AIAlgorithm, logger
from .deeplearning import DeepLearningAI
from game.chess_game import ChessGame
from tools.mcts_func import *


class MCTSNode:
    def __init__(self, game_state: ChessGame, parent=None, root_color=None):
        self.game_state = game_state  # 当前节点的游戏状态
        self.parent: MCTSNode = parent  # 父节点
        self.root_color = root_color # 目标颜色

        self.children: List[MCTSNode] = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数

        self.current_color = self.game_state.get_color()  # 当前颜色

        untried_moves = game_state.get_all_moves()  # 未尝试的走法列表
        self.untried_moves = untried_moves
        self.max_untried_index = len(untried_moves)  # 未尝试的走法最大索引
        self.untried_index = 0  # 未尝试的走法索引

    def is_fully_expanded(self) -> bool:
        """检查节点是否已完全展开"""
        return self.untried_index == self.max_untried_index
    
    def get_win_rate(self) -> float:
        """计算节点的胜率"""
        return self.wins / self.visits
    
    def get_current_move(self) -> Tuple[int, int]:
        """获取当前节点的移动"""
        return self.game_state.get_current_move()
    
    def get_child_win_rate_board(self):
        """获取子节点的胜率"""
        row_len, col_len = self.game_state.board_range
        child_win_rate_board = np.zeros((row_len, col_len), dtype=np.float64)
        for child in self.children:
            move = child.get_current_move()
            child_win_rate_board[move] = child.get_win_rate()
        return child_win_rate_board
    
    def get_best_child(self, policy_net: DeepLearningAI=None, c_param=0.6):
        """使用UCB1策略选择最佳子节点"""
        if policy_net:
            move_probs = policy_net.get_move_probs(self.game_state)
            rates_visits_probs = np.array([[child.wins / child.visits, child.visits, move_probs[child.get_current_move()[0], child.get_current_move()[1]]] 
                                           for child in self.children], dtype=np.float64)
            # test_arr = [(win_rate * policy_prob, math.sqrt(self.visits) / (1 + child_visit), policy_prob) 
            #             for win_rate, child_visit, policy_prob in rates_visits_probs]
            best_index = get_best_index_by_puct(rates_visits_probs, self.visits, c_param)
        else:
            rates_visits = np.array([[child.wins / child.visits, child.visits] for child in self.children], dtype=np.float64)
            # test_arr = [(win_rate, math.sqrt((math.log(self.visits) / child_visit))) for win_rate, child_visit in rates_visits]
            best_index = get_best_index_by_ucb1(rates_visits, self.visits, c_param)

        return self.children[best_index]

    def expand(self):
        """
        扩展一个新子节点并返回
        """
        # 选择第一个未尝试的移动并删除它
        move = self.untried_moves[self.untried_index]
        self.untried_index += 1

        new_game_state = self.game_state.copy()
        new_game_state.update_chessboard(*move, self.current_color)
        child_node = MCTSNode(new_game_state, parent=self, 
                              root_color=self.root_color)
        self.children.append(child_node)
        return child_node

    def simulate(self) -> float:
        """
        从当前节点进行一次完整的随机模拟
        """
        root_color = self.root_color
        current_simulation_state = self.game_state.copy()
        winner = current_simulation_state.simulate_by_random()
        
        # return {root_color: 1.0, -root_color: 0.0}.get(winner, 0.5) # about 74% speed of next line
        return 1.0 if winner == root_color else (0.0 if winner == -root_color else 0.5)

    def backpropagate(self, result: float):
        """将模拟结果向上传播到根节点"""
        node = self
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent

def tree_policy(node: MCTSNode, policy_net, c_param) -> MCTSNode:
    """根据选择策略递归选择子节点，直到达到未完全展开或未被访问的节点"""
    while not node.game_state.is_game_over():
        if not node.is_fully_expanded():
            return node.expand()
        else:
            node = node.get_best_child(policy_net=policy_net, c_param=c_param)
    return node

class MCTSAI(AIAlgorithm):
    def __init__(self, itermax: int = 1000, c_param = 0.9, policy_net=None, value_net=None, complate_mode=True) -> None:
        self.itermax = itermax
        self.c_param = c_param
        self.policy_net = policy_net
        self.value_net = value_net
        self.complate_mode = complate_mode

        init_rates_visits = np.ones((2, 2), dtype=np.float64)
        get_best_index_by_ucb1(init_rates_visits, 1, 1)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        """使用 MCTS 算法选择最佳移动"""
        root = MCTSNode(game.copy(), root_color=game.get_color()) # 创建一个MCTSNode对象，表示根节点
        best_child = self.MCTS(root) # 使用MCTS算法选择最佳的子节点

        if self.complate_mode:
            best_win_rate = best_child.get_win_rate()
            current_win_rate_board = root.get_child_win_rate_board()
            next_win_rate_board = best_child.get_child_win_rate_board()

            game.set_current_win_rate(best_win_rate)
            game.set_MCTSscore_board(next_win_rate_board)
            print(current_win_rate_board)

        return best_child.get_current_move()

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        for _ in range(self.itermax):
            node: MCTSNode = tree_policy(root, policy_net=self.policy_net, c_param=self.c_param)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.get_best_child(c_param=0)
    