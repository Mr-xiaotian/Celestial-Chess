"""
Author: 晓天
Vision: 1.3
"""
import hashlib
import numpy as np
from collections import deque
from tools.tools_func import *

class ChessGame:
    BLACK_HOLE = np.inf

    def __init__(self, board_range=(5, 5), power=2) -> None:
        self.chessboard = np.zeros((board_range[0], board_range[1], 2), dtype=float)
        self.power = power
        self.board_range = board_range
        self.threshold = self.power * 2 + 1

        self.history_board = {0: np.copy(self.chessboard)}
        self.history_move = {0: None}
        self.step = 0
        self.current_win_rate: float = 0.0

        init_board = np.zeros((board_range[0], board_range[1], 2), dtype=float)

        optimized_not_exist_zero_index(init_board)
        get_zero_index(init_board, board_range)
        get_first_channel(init_board)
        update_by_bfs(init_board, 0, 0, 0, power, board_range)
        
    def update_chessboard(self, row, col, color):
        """
        根据新规则更新棋盘状态，并考虑黑洞点的影响。
        """
        assert (
            0 <= row < self.board_range[0] and 0 <= col < self.board_range[1]
        ), f"落子点({row},{col})超出棋盘范围"
        assert (
            self.chessboard[row][col][0] == 0
        ), f"须在值为0处落子, ({row},{col})为{self.chessboard[row][col][0]}"

        # 更新落子点及周围点的值
        visited = self.update_adjacent_cells(row, col, color)

        # 检查并标记黑洞
        self.mark_black_holes(visited)

        self.step += 1
        self.history_board[self.step] = np.copy(self.chessboard)
        self.history_move[self.step] = (row, col)

    def update_adjacent_cells(self, row, col, color):
        """更新落子点周围的格子，考虑黑洞点对路径的阻挡作用，同时只影响下方的格子"""
        visited = update_by_bfs(self.chessboard, row, col, color, self.power, self.board_range)

    def mark_black_holes(self, visited: set):
        """标记黑洞区域"""
        while visited:
            row, col = visited.pop()
            if self.chessboard[row][col][1] >= self.threshold:  # 检查是否超过极限值
                # 标记为黑洞区域
                self.mark_adjacent_black_holes(row, col)

    def mark_adjacent_black_holes(self, row, col):
        """标记黑洞周围的格子为黑洞区域"""
        black_power = self.power - 1
        queue = deque([(row + black_power - 1, col, 0)])
        visited = set()

        while queue:
            r, c, distance = queue.popleft()
            if (r, c) in visited:
                continue
            elif distance >= black_power:
                continue
            
            try:
                self.chessboard[r][c][0] = self.BLACK_HOLE
            except IndexError as e:
                pass
            visited.add((r, c))
            
            # 向四个方向扩展
            for dr, dc in [(0, 1), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                queue.append((nr, nc, distance + 1))

    def undo(self):
        """悔棋"""
        self.step -= 1 if self.step >= 1 else 0
        self.chessboard = np.copy(self.history_board[self.step])

    def redo(self):
        """重悔"""
        if self.step + 1 in self.history_board.keys():
            self.step += 1
            self.chessboard = np.copy(self.history_board[self.step])

    def restart(self):
        """重开"""
        self.step = 0
        self.chessboard = np.copy(self.history_board[self.step])

    def set_current_win_rate(self, win_rate: float = 0.0):
        """设置当前玩家的胜率"""
        self.current_win_rate = win_rate

    def get_current_win_rate(self):
        """获取当前玩家的胜率"""
        return self.current_win_rate
    
    def get_current_move(self):
        """获取当前玩家的移动"""
        return self.history_move[self.step]

    def get_score(self):
        """计算棋盘上所有非无穷大格子的总分数"""
        total_score = sum(
            cell[0]
            for row in self.chessboard
            for cell in row
            if cell[0] != self.BLACK_HOLE
        )
        return total_score - self.get_balance_num()
    
    def get_balance_num(self):
        """计算平衡数"""
        balance_num = self.power * (self.power + 1) * (2 * self.power + 1) / 12
        return balance_num

    def get_all_moves(self):
        """获取所有合法移动"""
        return get_zero_index(self.chessboard, self.board_range)

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
            filtered_moves = self.get_all_moves()

        return filtered_moves

        # perfect_score = float('-inf') if color == 1 else float('inf')
        # perfect_moves = []

        # for move in filtered_moves:
        #     row, col = move
        #     self.update_chessboard(row, col, color)  # 更新棋盘状态
        #     score = self.get_score()  # 计算得分
        #     self.undo()  # 回溯棋盘状态

        #     if (color == 1 and score > perfect_score) or (color == -1 and score < perfect_score):
        #         perfect_score = score
        #         perfect_moves = [move]
        #     elif score == perfect_score:
        #         perfect_moves.append(move)
            
        # return perfect_moves
        
    def get_board_key(self):
        '''获取棋盘的哈希值'''
        data_str = str(self.chessboard) # 将数据转换为字符串形式
        hash_object = hashlib.sha256(data_str.encode()) # 使用 hashlib 库中的 sha256 函数来生成哈希值
        hash_hex = hash_object.hexdigest() # 获取十六进制表示的哈希值
        return hash_hex
    
    def get_board_value(self):
        '''获取棋盘的值'''
        return get_first_channel(self.chessboard)
    
    def get_color(self):
        '''获取当前玩家的颜色的颜色'''
        color = 1 if self.step % 2 == 0 else -1
        return color
    
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
        optimized_not_exist_zero_index(self.chessboard)
    
    def who_is_winner(self):
        '''
        判断当前玩家是否获胜
        :return: 1 蓝方获胜，-1 红方获胜，0 平局
        '''
        if not self.is_game_over():
            return None

        score = self.get_score()
        balance_num = self.get_balance_num()
        if self.get_color() == -1:
            if score > balance_num:
                return 1
            elif score == balance_num:
                return 0
            else:
                return -1
        else:
            if score < -1 * balance_num:
                return -1
            elif score == -1 * balance_num:
                return 0
            else:
                return 1

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

    

if __name__ == "__main__":
    game = ChessGame((5,5), power=2)
    game.update_chessboard(2, 2, 1)
    game.update_chessboard(2,1,-1)
    # # game.undo()
    # # print(game.step, game.history_board)
    # game.show_chessboard()
    # print(game.get_all_moves())
    # game.find_best_move(1, 1)
    print(game.format_matrix(game.chessboard))

