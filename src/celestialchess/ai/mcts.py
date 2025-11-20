from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

from ..chess_game import ChessGame
from ..tools.mcts_tool import *
from .base_ai import BaseAI, logger
from .deeplearning import DeepLearningAI


class MCTSNode:
    def __init__(
        self,
        chess_game: ChessGame,
        parent: Optional[MCTSNode] = None,
        root_color: Optional[int] = None,
    ):
        self.chess_game = chess_game  # 当前节点的游戏状态
        self.parent: MCTSNode = parent  # 父节点
        self.root_color = root_color  # 目标颜色

        self.children: List[MCTSNode] = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数

        self.current_color = self.chess_game.get_color()  # 当前颜色

        untried_moves = chess_game.get_all_moves()  # 未尝试的走法列表
        self.untried_moves = untried_moves
        self.max_untried_index = len(untried_moves)  # 未尝试的走法最大索引
        self.untried_index = 0  # 未尝试的走法索引

    def is_fully_expanded(self) -> bool:
        """检查节点是否已完全展开"""
        return self.untried_index == self.max_untried_index

    def change_root_color(self):
        self.root_color *= -1
        self.wins = self.visits - self.wins
        for child in self.children:
            child.change_root_color()

    def get_win_rate(self) -> float:
        """计算节点的胜率"""
        return self.wins / self.visits

    def get_current_move(self) -> Tuple[int, int]:
        """获取当前节点的移动"""
        return self.chess_game.get_current_move()

    def get_child_win_rate_board(self):
        """
        获取子节点的胜率
        """
        row_len, col_len = self.chess_game.board_range
        child_win_rate_board = np.zeros((row_len, col_len, 2), dtype=np.float64)
        for child in self.children:
            row, col = child.get_current_move()
            child_win_rate_board[row, col, 0] = child.get_win_rate()
            child_win_rate_board[row, col, 1] = child.visits
        return child_win_rate_board

    def get_best_child(self, c_param=0.6):
        """
        使用UCB1策略选择最佳子节点
        """
        wins_visits = np.array(
            [(child.wins, child.visits) for child in self.children], dtype=np.float64
        )
        # test_arr = [(win_rate, math.sqrt((math.log(self.visits) / child_visit))) for win_rate, child_visit in rates_visits]
        best_index = get_best_index_by_ucb1(wins_visits, self.visits, c_param)

        return self.children[best_index]

    def get_best_child_with_net(
        self, policy_net: Optional[DeepLearningAI], c_param=0.6
    ):
        """
        使用UCB1策略选择最佳子节点
        """
        move_probs = policy_net.get_move_probs(self.chess_game)
        wins_visits_probs = np.array(
            [
                [
                    child.wins,
                    child.visits,
                    move_probs[
                        child.get_current_move()[0], child.get_current_move()[1]
                    ],
                ]
                for child in self.children
            ],
            dtype=np.float64,
        )
        # test_arr = [(win_rate * policy_prob, math.sqrt(self.visits) / (1 + child_visit), policy_prob)
        #             for win_rate, child_visit, policy_prob in rates_visits_probs]
        best_index = get_best_index_by_puct(wins_visits_probs, self.visits, c_param)
        return self.children[best_index]

    def expand(self):
        """
        扩展一个新子节点并返回
        """
        # 选择第一个未尝试的移动并删除它
        move = self.untried_moves[self.untried_index]
        self.untried_index += 1

        new_chess_game = self.chess_game.copy()
        new_chess_game.update_chessboard(*move, self.current_color)
        child_node = MCTSNode(new_chess_game, parent=self, root_color=self.root_color)
        self.children.append(child_node)
        return child_node

    def simulate(self) -> float:
        """
        从当前节点进行一次完整的随机模拟
        """
        root_color = self.root_color
        current_simulation_state = self.chess_game.copy()
        winner = current_simulation_state.simulate_by_random()

        # return {root_color: 1.0, -root_color: 0.0}.get(winner, 0.5) # about 74% speed of next line
        return 1.0 if winner == root_color else (0.0 if winner == -root_color else 0.5)

    def backpropagate(self, result: float):
        """
        将模拟结果向上传播到根节点
        """
        node = self
        while node:
            node.visits += 1
            node.wins += result
            node = node.parent


class MCTSAI(BaseAI):
    def __init__(self, itermax: int = 1000, c_param=0.8, complete_mode=True) -> None:
        self.itermax = itermax
        self.c_param = c_param
        self.complete_mode = complete_mode

        self.cache = {}

        init_rates_visits = np.ones((2, 2), dtype=np.float64)
        get_best_index_by_ucb1(init_rates_visits, 1, 1)

    def find_best_move(self, game: ChessGame) -> Tuple[int, int]:
        """使用 MCTS 算法选择最佳移动"""
        current_move = game.get_current_move()
        if current_move in self.cache:
            root: MCTSNode = self.cache[current_move]
            root.change_root_color() if game.get_color() != root.root_color else None
        else:
            root = MCTSNode(
                game.copy(), root_color=game.get_color()
            )  # 创建一个MCTSNode对象，表示根节点

        best_child = self.MCTS(root)  # 使用MCTS算法选择最佳的子节点
        best_move = best_child.get_current_move()

        # 将最佳子节点和根节点存储在缓存中
        self.cache = {best_move: best_child}
        for child in best_child.children:
            move = child.get_current_move()
            self.cache[move] = child

        # 如果是完整模式，则更新游戏状态和评分板
        (
            self.update_game_with_mcts_results(game, root, best_child)
            if self.complete_mode
            else None
        )

        return best_move

    def update_game_with_mcts_results(
        self, game: ChessGame, root: MCTSNode, best_child: MCTSNode
    ) -> None:
        """
        根据 MCTS 搜索结果更新游戏状态和评分板。

        :param game: 当前棋盘游戏对象
        :param root: MCTS 树的根节点
        :param best_child: MCTS 搜索的最佳子节点
        """
        best_win_rate = best_child.get_win_rate()
        current_win_rate_board = root.get_child_win_rate_board()
        next_win_rate_board = best_child.get_child_win_rate_board()

        game.set_current_win_rate(best_win_rate)
        game.set_MCTSscore_board(next_win_rate_board)
        print(game.get_format_board(current_win_rate_board, (3, 0)))

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        # 提前终止条件
        # if len(root.untried_moves) <= 1:
        #     return root.get_best_child(c_param=0)

        for _ in range(self.itermax):
            node = self.tree_policy(root, c_param=self.c_param)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.get_best_child(c_param=0)

    def tree_policy(self, node: MCTSNode, c_param) -> MCTSNode:
        """根据选择策略递归选择子节点，直到达到未完全展开或未被访问的节点"""
        while not node.chess_game.is_game_over():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.get_best_child(c_param)
        return node

    def end_game(self):
        self.cache = {}

    def end_model(self):
        pass


class MCTSPlusAi(MCTSAI):
    def __init__(
        self,
        itermax: int = 1000,
        c_param=0.8,
        policy_net: Optional[DeepLearningAI] = None,
        value_net: Optional[DeepLearningAI] = None,
        complete_mode=True,
    ) -> None:

        super().__init__(itermax, c_param, complete_mode)
        self.policy_net = policy_net
        self.value_net = value_net

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        # 提前终止条件
        # if len(root.untried_moves) <= 1:
        #     return root.get_best_child(c_param=0)

        for _ in range(self.itermax):
            node = self.tree_policy_with_net(
                root, policy_net=self.policy_net, c_param=self.c_param
            )
            reward = node.simulate()
            node.backpropagate(reward)
        return root.get_best_child(c_param=0)

    def tree_policy_with_net(
        self, node: MCTSNode, policy_net: Optional[DeepLearningAI], c_param
    ) -> MCTSNode:
        """根据选择策略递归选择子节点，直到达到未完全展开或未被访问的节点"""
        while not node.chess_game.is_game_over():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.get_best_child_with_net(
                    policy_net=policy_net, c_param=c_param
                )
        return node
