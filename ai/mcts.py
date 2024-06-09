import math
import random
from __future__ import annotations
from copy import deepcopy
from typing import Tuple, List
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class MCTSNode:
    def __init__(self, game_state: ChessGame, parent=None, target_color=None):
        self.game_state = deepcopy(game_state)  # 当前节点的游戏状态
        self.parent: MCTSNode = parent  # 父节点
        self.target_color = target_color # 目标颜色

        self.children: List[MCTSNode] = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数
        self.untried_moves = game_state.get_all_moves()  # 未尝试的走法列表

    def is_fully_expanded(self) -> bool:
        """检查节点是否已完全展开"""
        return len(self.untried_moves) == 0
    
    def get_win_rate(self) -> float:
        """计算节点的胜率"""
        return self.wins / self.visits if self.visits > 0 else 0
    
    def get_current_move(self) -> Tuple[int, int]:
        """获取当前节点的移动"""
        return self.game_state.get_current_move()
    
    def UCB1(self, child: MCTSNode, c_param=1.4):
        """使用UCB1策略选择最佳子节点"""
        return child.get_win_rate() + c_param * math.sqrt((math.log(self.visits) / child.visits))

    def update(self, win: bool):
        """更新节点的胜利次数和访问次数"""
        self.visits += 1
        self.wins += win

    def best_child(self, c_param=1.4):
        """使用UCB1策略选择最佳子节点"""
        choices_weights = [
            self.UCB1(child, c_param)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """扩展一个新子节点并返回"""
        new_game_state = deepcopy(self.game_state)
        move = self.untried_moves.pop()
        color = self.game_state.get_color()
        target_color = self.target_color
        new_game_state.update_chessboard(*move, color)
        child_node = MCTSNode(new_game_state, parent=self, target_color=target_color)
        self.children.append(child_node)
        return child_node

    def simulate(self) -> float:
        """从当前节点进行一次完整的随机模拟"""
        current_simulation_state = deepcopy(self.game_state)
        while not current_simulation_state.is_game_over():
            possible_moves = current_simulation_state.get_all_moves()
            move = random.choice(possible_moves)
            color = current_simulation_state.get_color()
            current_simulation_state.update_chessboard(*move, color)
        
        if current_simulation_state.who_is_winner() == self.target_color:
            return 1.0
        elif current_simulation_state.who_is_winner() == -1 * self.target_color:
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
            node = node.best_child()
    return node

class MCTSAI(AIAlgorithm):
    def __init__(self, itermax: int = 1000) -> None:
        self.itermax = itermax

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        target_color = game.get_color()
        root = MCTSNode(game, target_color=target_color) # 创建一个MCTSNode对象，表示根节点
        best_child = self.MCTS(root) # 使用MCTS算法选择最佳的子节点
        game.set_current_win_rate(best_child.get_win_rate())

        return best_child.get_current_move()

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        for _ in range(self.itermax):
            node = tree_policy(root)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.best_child(c_param=0)
    
