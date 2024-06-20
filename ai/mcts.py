from __future__ import annotations
import random
import numpy as np
from typing import Tuple, List
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame
from tools.mcts_func import *


class MCTSNode:
    def __init__(self, game_state: ChessGame, parent=None, target_color=None):
        self.game_state = game_state  # 当前节点的游戏状态
        self.parent: MCTSNode = parent  # 父节点
        self.target_color = target_color # 目标颜色

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
    
    def get_best_child(self, c_param=0.9):
        """使用UCB1策略选择最佳子节点"""
        rates_visits = np.array([[child.wins / child.visits, child.visits] for child in self.children], dtype=np.float64)
        best_index = get_best_index(rates_visits, self.visits, c_param)

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
                              target_color=self.target_color)
        self.children.append(child_node)
        return child_node

    def simulate(self) -> float:
        """
        从当前节点进行一次完整的随机模拟
        """
        current_color = self.current_color
        target_color = self.target_color
        current_simulation_state = self.game_state.copy()

        while not current_simulation_state.is_game_over():
            possible_moves = current_simulation_state.get_all_moves()  # 未尝试的走法列表
            move = random.choice(possible_moves)
            current_simulation_state.update_chessboard(*move, current_color)
            current_color *= -1
        
        winner = current_simulation_state.who_is_winner(current_color)
        if winner == target_color:
            return 1.0
        elif winner == -1 * target_color:
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
    def __init__(self, itermax: int = 1000, complate_mode=True) -> None:
        self.itermax = itermax
        self.complate_mode = complate_mode

        init_rates_visits = np.empty((2, 2), dtype=np.float64)
        get_best_index(init_rates_visits, 1, 1)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        """使用 MCTS 算法选择最佳移动"""
        root = MCTSNode(game.copy(), target_color=game.get_color()) # 创建一个MCTSNode对象，表示根节点
        best_child = self.MCTS(root) # 使用MCTS算法选择最佳的子节点
        game.set_current_win_rate(best_child.get_win_rate()) if self.complate_mode else None

        return best_child.get_current_move()

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        for _ in range(self.itermax):
            node = tree_policy(root)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.get_best_child(c_param=0)
    
