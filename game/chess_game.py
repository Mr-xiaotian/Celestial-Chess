"""
Author: 晓天
Vision: 1.4
"""
import hashlib
import numpy as np
from tools.chess_func import *

class ChessGame:
    BLACK_HOLE = np.inf

    def __init__(self, board_range=(5, 5), power=2) -> None:
        self.chessboard = np.zeros((board_range[0], board_range[1], 2), dtype=float)

        # 仅定义模拟运行必须的参数
        self.board_range = board_range
        self.power: int = power
        self.threshold: int = self.power * 2 + 1
        self.balance_num: float = self.get_balance_num()

        self.current_color = 1
        self.current_move = (-1, -1)

    def init_history(self):
        """
        初始化历史棋局状态, 只在正式运行时启用。
        """
        self.step: int = 0

        self.current_max_step: int = 0
        row_len, col_len = self.board_range
        max_steps = row_len * col_len * 2

        self.history_board = np.empty((max_steps, row_len, col_len, 2), dtype=float)
        self.history_move = np.empty((max_steps, 2), dtype=int)
        self.history_board[0] = self.chessboard
        self.history_move[0] = (-1, -1)

    def init_cfunc(self):
        """
        初始化numba计算函数, 第一次实例化ChessGame时使用。
        """
        init_board = np.empty((5, 5, 2), dtype=float)

        calculate_no_inf(init_board)
        optimized_not_exist_zero_index(init_board)
        get_all_zero_index(init_board, (5,5))
        get_random_zero_index(init_board, (5,5))
        get_first_channel(init_board, (5,5))
        bfs_expand_with_power_threshold(init_board, (5,5), 0, 0, 1, 1, 1)
        simulate_to_over_by_random(init_board, (5,5), 1, 2, 5, 2.5)

    def copy(self):
        """
        复制当前棋局状态，用于回溯。
        """
        new_game = ChessGame(self.board_range, self.power)
        new_game.chessboard = np.copy(self.chessboard)
        new_game.current_move = self.current_move
        new_game.current_color = self.current_color
        return new_game
        
    def update_chessboard(self, row, col, color):
        """
        根据新规则更新棋盘状态，并考虑黑洞点的影响。
        """
        # 用bfs扩散落子点及周围点的值, 然后更新黑洞点
        bfs_expand_with_power_threshold(self.chessboard, self.board_range, row, col, color, self.power, self.threshold)

        # 更新必要棋盘状态
        self.current_color *= -1
        self.current_move = (row, col)

    def update_history(self, row, col):
        """
        更新历史记录
        """
        self.step += 1
        self.current_max_step += 1
        self.history_board[self.step] = np.copy(self.chessboard)
        self.history_move[self.step] = (row, col)

    def undo(self):
        """悔棋"""
        self.step -= 1 if self.step >= 1 else 0
        self.chessboard = np.copy(self.history_board[self.step])
        self.current_move = self.history_move[self.step]

    def redo(self):
        """重悔"""
        if self.step + 1 <= self.current_max_step:
            self.step += 1
            self.chessboard = np.copy(self.history_board[self.step])
            self.current_move = self.history_move[self.step]

    def restart(self):
        """重开"""
        self.step = 0
        self.chessboard = np.copy(self.history_board[self.step])
        self.current_move = self.history_move[self.step]

    def set_MCTSscore_board(self, mcts_score_board):
        """设置MCTS得分板"""
        self.mcts_score_board = mcts_score_board

    def get_MCTSscore_board(self):
        """获取MCTS得分板"""
        return self.mcts_score_board

    def set_current_win_rate(self, win_rate: float = 0.0):
        """设置当前玩家的胜率"""
        self.current_win_rate = win_rate

    def get_current_win_rate(self):
        """获取当前玩家的胜率"""
        return self.current_win_rate
    
    def get_current_move(self):
        """获取当前玩家的移动"""
        current_move = self.current_move
        return (int(current_move[0]), int(current_move[1]))

    def get_score(self):
        """计算棋盘上所有非无穷大格子的总分数"""
        total_score = calculate_no_inf(self.chessboard)
        return total_score - self.balance_num
    
    def get_balance_num(self):
        """计算平衡数"""
        balance_num = self.power * (self.power + 1) * (2 * self.power + 1) / 12
        return balance_num

    def get_all_moves(self):
        """获取所有合法移动"""
        return get_all_zero_index(self.chessboard, self.board_range)
    
    def get_random_move(self):
        """获取随机合法移动"""
        return get_random_zero_index(self.chessboard, self.board_range)

    def get_perfect_moves(self):
        """
        获取所有最好的合法移动
        仅在深度较浅时有用
        """
        filtered_moves = []
        for row in range(0, self.board_range[0] - self.power + 1):
            for col in range(self.power - 1, self.board_range[1] - self.power + 1):
                if abs(self.chessboard[row][col][0]) != 0:
                    continue
                filtered_moves.append((row, col))

        if not filtered_moves:
            filtered_moves = get_all_zero_index(self.chessboard, self.board_range)

        return filtered_moves
        
    def get_board_key(self):
        '''
        获取棋盘的哈希值
        '''
        # 将数组转换为字节数据
        arr_bytes = self.chessboard.tobytes()
        
        # 计算 SHA-256 哈希值
        sha256_hash = hashlib.sha256(arr_bytes).hexdigest()
        
        return sha256_hash
    
    def get_board_value(self):
        '''获取棋盘的值'''
        return get_first_channel(self.chessboard, self.board_range)
    
    def get_color(self):
        '''获取当前玩家的颜色的颜色'''
        return self.current_color
    
    def get_format_board(self):
        '''获取棋盘的格式化'''
        return self.format_matrix(self.chessboard)
    
    def get_format_board_value(self):
        '''获取棋盘数值的格式化'''
        return self.format_simple_matrix(self.get_board_value())

    def is_game_over(self):
        '''
        判断游戏是否结束
        :return: 游戏是否结束
        '''
        return optimized_not_exist_zero_index(self.chessboard)
    
    def is_move_valid(self, row, col):
        '''
        判断移动是否有效
        :param move: 移动
        :return: 移动是否有效
        '''
        row_len, col_len = self.board_range
        if self.chessboard[row, col, 0] != 0:
            # raise ValueError(f"须在值为0处落子, ({row},{col})为{self.chessboard[row, col, 0]}")
            return False
        elif row >= row_len or row < 0 or col >= col_len or col < 0:
            # raise ValueError(f"超出棋盘范围, ({row},{col})")
            return False
        else:
            return True
    
    def who_is_winner(self, color = None):
        '''
        判断当前玩家是否获胜
        :return: 1 蓝方获胜，-1 红方获胜，0 平局
        '''
        score = self.get_score()
        balance_num = self.balance_num
        color = color if color is not None else self.get_color()
        comparison = score + color * balance_num # color: 1 蓝方，-1 红方

        return (comparison > 0) - (comparison < 0)
    
    def simulate_by_random(self):
        '''
        随机模拟游戏
        :return: 1 蓝方获胜，-1 红方获胜，0 平局
        '''
        return simulate_to_over_by_random(self.chessboard, self.board_range, self.current_color, 
                                          self.power, self.threshold, self.balance_num)
        current_color = self.current_color
        while not self.is_game_over():
            random_move = self.get_random_move()
            self.update_chessboard(*random_move, current_color)
            current_color *= -1
        
        winner = self.who_is_winner(current_color)
        return winner

    def show_chessboard(self):
        '''打印棋盘'''
        print(self.get_format_board())

    def show_chessboard_value(self):
        '''打印棋盘数值部分'''
        print(self.get_format_board_value())

    def format_matrix(self, matrix):
        '''
        格式化矩阵
        :param matrix: 矩阵
        :return: 格式化后的字符串
        '''
        # 确定每个元素的最大宽度
        max_width = max(len(str(item)) for row in matrix for sublist in row for item in sublist)
        
        formatted_rows = []
        for row in matrix:
            formatted_row = "  ["
            for sublist in row:
                formatted_row += "[" + ", ".join(f"{item:>{max_width}}" for item in sublist) + "], "
            formatted_row = formatted_row.rstrip(", ") + "]"
            formatted_rows.append(formatted_row)
        
        formatted_string = "[\n" + ",\n".join(formatted_rows) + "\n]"
        return formatted_string
    
    def format_simple_matrix(self, matrix):
        '''
        格式化简单矩阵
        :param matrix: 矩阵
        :return: 格式化后的字符串
        '''
        # 确定每个元素的最大宽度
        max_width = max(len(str(item)) for row in matrix for item in row)
        
        formatted_rows = []
        for row in matrix:
            formatted_row = "  [" + ", ".join(f"{item:>{max_width}}" for item in row) + "]"
            formatted_rows.append(formatted_row)
        
        formatted_string = "[\n" + ",\n".join(formatted_rows) + "\n]"
        return formatted_string


