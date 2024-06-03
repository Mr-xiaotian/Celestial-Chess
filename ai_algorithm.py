
import math
import random
from collections import deque
from copy import deepcopy
from typing import Tuple
from time import time, strftime, localtime
from chess_game import ChessGame
from loguru import logger


# Configure logging
logger.remove()  # remove the default handler
now_time = strftime("%Y-%m-%d", localtime())
logger.add(f"logs/chess_manager({now_time}).log", 
           format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}",
           level="INFO")


class AIAlgorithm:
    def find_best_move(self, game: ChessGame, color: int) -> Tuple[int, int]:
        raise NotImplementedError("This method should be overridden by subclasses.")
    

class MinimaxAI(AIAlgorithm):
    def __init__(self, depth: int = 3) -> None:
        self.depth = depth

    def find_best_move(self, game: ChessGame, color: int) -> Tuple[int, int]:
        best_move = None
        self.iterate_time = 0
        depth = self.depth
        logger.debug(f"MinimaxAI is thinking in depth {depth}...\n{game.format_matrix(game.chessboard)}")
        game.load_transposition_table()

        if color == 1:
            best_score = float("-inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                score = self.minimax(game, depth, color * -1, float("-inf"), float("inf"))
                game.undo()
                if score > best_score:
                    best_score = score
                    best_move = move

        elif color == -1:
            best_score = float("inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                score = self.minimax(game, depth, color * -1, float("-inf"), float("inf"))
                game.undo()
                if score < best_score:
                    best_score = score
                    best_move = move

        game.save_transposition_table()

        return best_move

    def minimax(self, game: ChessGame, depth: int, color: int, alpha: float, beta: float) -> float:
        self.iterate_time += 1
        logger.debug(f"Iteration {self.iterate_time} in depth {depth}")

        board_key = game.get_board_key()
        if board_key in game.transposition_table and \
            game.transposition_table[board_key]['depth'] >= depth:
            return game.transposition_table[board_key]['score']
        
        if depth == 0 or game.is_game_over():
            score = game.get_score()
            game.update_transposition_table(board_key, {'score': score, 'depth': depth})
            return score

        if color == 1:
            max_eval = float("-inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                eval = self.minimax(game, depth - 1, -1, alpha, beta)
                game.undo()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            game.update_transposition_table(board_key, {'score': max_eval, 'depth': depth})
            return max_eval
        elif color == -1:
            min_eval = float("inf")
            for move in game.get_all_moves():
                game.update_chessboard(*move, color)
                eval = self.minimax(game, depth - 1, 1, alpha, beta)
                game.undo()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            game.update_transposition_table(board_key, {'score': min_eval, 'depth': depth})
            return min_eval
        

class MCTSNode:
    def __init__(self, game_state: ChessGame, parent=None):
        self.game_state = deepcopy(game_state)  # 当前节点的游戏状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数
        self.untried_moves = game_state.get_all_moves()  # 未尝试的走法列表

    def is_fully_expanded(self) -> bool:
        """检查节点是否已完全展开"""
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.4):
        """使用UCB1策略选择最佳子节点"""
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt((math.log(self.visits) / child.visits))
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
        child_node = MCTSNode(new_game_state, parent=self)
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

    def find_best_move(self, game: ChessGame, color: int) -> Tuple[int, int]:
        root = MCTSNode(game) # 创建一个MCTSNode对象，表示根节点
        best_child = self.MCTS(root) # 使用MCTS算法选择最佳的子节点

        # 遍历根节点的所有子节点
        for child in root.children:
            # 如果子节点的棋盘状态与最佳子节点的棋盘状态相同
            if child.game_state.chessboard == best_child.game_state.chessboard:
                # 遍历所有可能的移动
                for move in game.get_all_moves():
                    new_game_state = deepcopy(game) # 创建一个新的棋盘状态
                    new_game_state.update_chessboard(*move, color) # 更新棋盘状态
                    # 如果新的棋盘状态与最佳子节点的棋盘状态相同
                    if new_game_state.chessboard == best_child.game_state.chessboard:
                        # 返回移动
                        return move
        return None

    def MCTS(self, root: MCTSNode) -> MCTSNode:
        """执行迭代次数为 itermax 的 MCTS 搜索，返回最佳子节点"""
        for _ in range(self.itermax):
            node = tree_policy(root)
            reward = node.simulate()
            node.backpropagate(reward)
        return root.best_child(c_param=0)
    

def ai_battle(ai_blue: AIAlgorithm, ai_red: AIAlgorithm):
    test_game = ChessGame()
    ai_blue_name = ai_blue.__class__.__name__
    ai_red_name = ai_red.__class__.__name__
    print(f'游戏开始！\n蓝方AI: {ai_blue_name}\n红方AI: {ai_red_name}\n')

    first_time = time()
    while True:
        last_time = time()
        color = test_game.get_color()
        if color == 1:
            move = ai_blue.find_best_move(test_game, color)
        else:
            move = ai_red.find_best_move(test_game, color)

        test_game.update_chessboard(*move, color)
        test_game.show_chessboard()
        print(f'第{test_game.step}步: {"蓝方" if color==1 else "红方"} 落子在 {move}')
        print(f'分数: {test_game.get_score()}')
        print(f'用时: {time()-last_time}\n')

        if test_game.is_game_over():
            if test_game.who_is_winner()==1:
                print(f'{"蓝方"+ai_blue_name} 获胜！')
            elif test_game.who_is_winner()==-1:
                print(f'{"红方"+ai_red_name} 获胜！')
            else:
                print('平局！')
            print(f'用时:{time()-first_time}')
            break

if __name__ == '__main__':
    minimax_ai = MinimaxAI(5)
    mcts_ai = MCTSAI(10000)

    ai_battle(minimax_ai, mcts_ai)