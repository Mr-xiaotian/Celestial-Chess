import math
import random
from copy import deepcopy
from typing import Tuple, List
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class MCTSNode:
    def __init__(self, game_state: ChessGame, parent=None, move=None):
        self.game_state = deepcopy(game_state)  # 当前节点的游戏状态
        self.parent: MCTSNode = parent  # 父节点
        self.move = move  # 移动的棋子

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

    def best_child(self, c_param=1.4):
        """使用UCB1策略选择最佳子节点"""
        choices_weights = [
            child.get_win_rate() + c_param * math.sqrt((math.log(self.visits) / child.visits))
            for child in self.children
        ]
        color = self.game_state.get_color()
        choices_weights = [color * choice for choice in choices_weights]
        return self.children[choices_weights.index(max(choices_weights))]

    def expand(self):
        """扩展一个新子节点并返回"""
        new_game_state = deepcopy(self.game_state)
        move = self.untried_moves.pop()
        color = self.game_state.get_color()
        new_game_state.update_chessboard(*move, color)
        child_node = MCTSNode(new_game_state, parent=self, move=move)
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
        return current_simulation_state.who_is_winner()

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
        root = MCTSNode(game) # 创建一个MCTSNode对象，表示根节点
        best_child = self.MCTS(root) # 使用MCTS算法选择最佳的子节点
        game.set_current_win_rate(best_child.get_win_rate())

        return best_child.move

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        for _ in range(self.itermax):
            node = tree_policy(root)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.best_child(c_param=0)
    
