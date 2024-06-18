from __future__ import annotations
import random
import numpy as np
from copy import deepcopy
from typing import Tuple, List
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame
from tools.mcts_func import *


class MCTSNode:
    def __init__(self, game_state: ChessGame, parent=None, target_color=None, flag=True):
        self.game_state = game_state  # 当前节点的游戏状态
        self.parent: MCTSNode = parent  # 父节点
        self.target_color = target_color # 目标颜色

        self.children: List[MCTSNode] = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数

        self.flag = flag  # 是否启用另一种走法
        if flag:
            self.untried_moves = game_state.get_all_moves()  # 未尝试的走法列表
        else:
            self.untried_moves = game_state.get_perfect_moves()  # 另一种走法列表

    def is_fully_expanded(self) -> bool:
        """检查节点是否已完全展开"""
        return len(self.untried_moves) == 0
    
    def get_win_rate(self) -> float:
        """计算节点的胜率"""
        return self.wins / self.visits if self.visits > 0 else 0
    
    def get_current_move(self) -> Tuple[int, int]:
        """获取当前节点的移动"""
        return self.game_state.get_current_move()

    def update(self, win: bool):
        """更新节点的胜利次数和访问次数"""
        self.visits += 1
        self.wins += win
    
    def get_best_child(self, c_param=0.9):
        """使用UCB1策略选择最佳子节点"""
        rates_visits = np.array([[child.get_win_rate(),child.visits] for child in self.children], dtype=np.float64)
        best_index = get_best_index(rates_visits, self.visits, c_param)

        return self.children[best_index]

    def expand(self):
        """扩展一个新子节点并返回"""

        # 选择第一个未尝试的移动并删除它
        move = self.untried_moves[0]
        self.untried_moves = np.delete(self.untried_moves, 0, axis=0)

        color = self.game_state.get_color()
        target_color = self.target_color

        new_game_state = self.game_state.copy()
        new_game_state.update_chessboard(*move, color)
        child_node = MCTSNode(new_game_state, parent=self, target_color=target_color, flag=self.flag)
        self.children.append(child_node)
        return child_node

    def simulate(self) -> float:
        """从当前节点进行一次完整的随机模拟"""
        current_simulation_state = self.game_state.copy()
        current_color = current_simulation_state.get_color()
        while not current_simulation_state.is_game_over():
            if self.flag:
                possible_moves = current_simulation_state.get_all_moves()
            else:
                possible_moves = current_simulation_state.get_perfect_moves()  # 未尝试的走法列表
            move = random.choice(possible_moves)
            current_color *= -1
            current_simulation_state.update_chessboard(*move, current_color)
        
        winner = current_simulation_state.who_is_winner()
        if winner == self.target_color:
            return 1.0
        elif winner == -1 * self.target_color:
            return 0.0
        else:
            return 0.5

    def backpropagate(self, result: float):
        """将模拟结果向上传播到根节点"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

def tree_policy(node: MCTSNode) -> MCTSNode:
    """根据选择策略递归选择子节点，直到达到未完全展开或未被访问的节点"""
    while not node.game_state.is_game_over():
        if not node.is_fully_expanded():
            return node.expand()
        else:
            node = node.get_best_child()
    return node

class MCTSAI(AIAlgorithm):
    def __init__(self, itermax: int = 1000, flag=True) -> None:
        self.itermax = itermax
        self.flag = flag

        init_rates_visits = np.ones((itermax, 2), dtype=np.float64)
        get_best_index(init_rates_visits, 1, 1)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        """使用 MCTS 算法选择最佳移动"""
        root = MCTSNode(game.copy(), target_color=game.get_color(), flag=self.flag) # 创建一个MCTSNode对象，表示根节点
        best_child = self.MCTS(root) # 使用MCTS算法选择最佳的子节点
        game.set_current_win_rate(best_child.get_win_rate())

        return best_child.get_current_move()

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        for _ in range(self.itermax):
            node = tree_policy(root)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.get_best_child(c_param=0)
    
