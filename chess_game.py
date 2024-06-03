"""
Author: 晓天
Vision: 1.1
"""
from time import strftime, localtime
import pickle, hashlib
from copy import deepcopy
from collections import deque
from pprint import pprint
from loguru import logger

logger.remove()  # remove the default handler
now_time = strftime("%Y-%m-%d", localtime())
logger.add(f"logs/chess_manager({now_time}).log", format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}")

class ChessGame:
    def __init__(self, board_range=(5, 5), power=2) -> None:
        self.chessboard = [
            [[0, 0] for _ in range(board_range[1])] for _ in range(board_range[0])
        ]
        self.power = power
        self.threshold = self.power * 2 + 1

        self.member_dict = {0: deepcopy(self.chessboard)}
        self.step = 0
        self.transposition_file = f"transposition_table/transposition_table({power}&{board_range[0]}_{board_range[1]})(sha256).pickle"
        self.transposition_table = dict()
        self.transposition_table_change = False

    def load_transposition_table(self):
        """
        加载transposition table
        """
        try:
            with open(self.transposition_file, "rb") as file:
                self.transposition_table = pickle.load(file)
                logger.info(f"load transposition table from {self.transposition_file}")
        except (FileNotFoundError, EOFError):
            with open(self.transposition_file, "wb") as file:
                pickle.dump(self.transposition_table, file)

    def update_transposition_table(self, key, value):
        """
        更新transposition table
        """
        if (
            key not in self.transposition_table
            or self.transposition_table[key]["depth"] < value["depth"]
        ):
            old_value = self.transposition_table.get(key, None)
            self.transposition_table[key] = value
            self.transposition_table_change = True
            logger.info(f"update transposition table: {old_value} -> {value}\n{self.format_matrix(self.chessboard):>20}")
            
    def save_transposition_table(self):
        """
        保存transposition table到文件
        """
        if not self.transposition_table_change:
            self.transposition_table = dict()
            self.transposition_table_change = False
            return
        with open(self.transposition_file, "wb") as file:
            pickle.dump(self.transposition_table, file)
            self.transposition_table = dict()
            self.transposition_table_change = False
            logger.info(f"save transposition table to {self.transposition_file}")

    def update_chessboard(self, row, col, color):
        """
        根据新规则更新棋盘状态，并考虑黑洞点的影响。
        """
        assert (
            self.chessboard[row][col][0] == 0
        ), f"须在值为0处落子, ({row},{col})为{self.chessboard[row][col][0]}"

        # 落子点设置为3，并更新周围点的值
        self.chessboard[row][col][0] += self.power * color
        self.chessboard[row][col][1] += self.power
        self.update_adjacent_cells(row, col, color)

        # 检查并标记黑洞
        self.mark_black_holes()

        self.step += 1
        self.member_dict[self.step] = deepcopy(self.chessboard)

    def update_adjacent_cells(self, row, col, color):
        """更新落子点周围的格子，考虑黑洞点对路径的阻挡作用，同时只影响右侧的格子"""
        visited = set()
        queue = deque([(row, col, 0)])  # 使用队列进行BFS搜索

        while queue:
            r, c, distance = queue.popleft()
            if (r, c) != (row, col) and (r, c) not in visited:
                if self.chessboard[r][c][0] is not float("inf"):  # 检查是否为黑洞点
                    # print(distance, r, c)
                    change = max(self.power - distance, 0) * color
                    self.chessboard[r][c][0] += change
                    self.chessboard[r][c][1] += change * color
                visited.add((r, c))

            if distance < self.power - 1:
                # 只向右侧、上方和下方扩展搜索区域
                for dr, dc in [(0, 1), (-1, 0), (1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < len(self.chessboard)
                        and 0 <= nc < len(self.chessboard[0])
                        and (nr, nc) not in visited
                        and self.chessboard[nr][nc][0] != float("inf")
                    ):
                        queue.append((nr, nc, distance + 1))

    def mark_black_holes(self):
        """标记黑洞区域"""
        for row in range(len(self.chessboard[0])):
            for col in range(len(self.chessboard[1])):
                if self.chessboard[row][col][1] >= self.threshold:  # 检查是否超过极限值
                    # 标记为黑洞区域
                    self.mark_adjacent_black_holes(row, col)

    def mark_adjacent_black_holes(self, row, col):
        """标记黑洞周围的格子为黑洞区域"""
        black_power = self.power - 2
        for dr in range(-1 * black_power, black_power + 1):
            for dc in range(-black_power, 1):
                r, c = row + dr, col + dc + black_power
                distance = abs(dr) + abs(dc)
                if (
                    0 <= r < len(self.chessboard)
                    and 0 <= c < len(self.chessboard[0])
                    and distance < black_power + 1
                ):
                    self.chessboard[r][c][0] = float("inf")

    def undo(self):
        """悔棋"""
        self.step -= 1 if self.step >= 1 else 0
        self.chessboard = deepcopy(self.member_dict[self.step])

    def redo(self):
        """重悔"""
        if self.step + 1 in self.member_dict.keys():
            self.step += 1
            self.chessboard = deepcopy(self.member_dict[self.step])

    def restart(self):
        """重开"""
        self.step = 0
        self.chessboard = deepcopy(self.member_dict[self.step])

    def get_score(self):
        """计算棋盘上所有非无穷大格子的总分数"""
        total_score = sum(
            cell[0]
            for row in self.chessboard
            for cell in row
            if cell[0] != float("inf")
        )
        return total_score - self.get_balance_num()
    
    def get_balance_num(self):
        """计算平衡数"""
        balance_num = self.power * (self.power + 1) * (2 * self.power + 1) / 12
        return balance_num

    def get_all_moves(self):
        """创建一个空列表，用于存放棋盘上所有格子的坐标"""
        move_list = []
        for row_idx, row in enumerate(self.chessboard):
            for col_idx, cell in enumerate(row):
                if abs(cell[0]) == 0:
                    move_list.append((row_idx, col_idx))
        return move_list
    
    def get_board_key(self):
        '''获取棋盘的哈希值'''
        data_str = str(self.chessboard) # 将数据转换为字符串形式
        hash_object = hashlib.sha256(data_str.encode()) # 使用 hashlib 库中的 sha256 函数来生成哈希值
        hash_hex = hash_object.hexdigest() # 获取十六进制表示的哈希值
        return hash_hex
    
    def get_board_value(self):
        '''获取棋盘的值'''
        return [[cell[0] for cell in row] for row in self.chessboard]
    
    def get_color(self):
        '''获取当前玩家的颜色的颜色'''
        color = 1 if self.step % 2 == 0 else -1
        return color

    def is_game_over(self):
        '''
        判断游戏是否结束
        :return: 游戏是否结束
        '''
        for row_value in self.get_board_value():
            for cell_value in row_value:
                if cell_value == 0:
                    return False
        return True
    
    def who_is_winner(self):
        '''
        判断当前玩家是否获胜
        :return: 1 蓝方获胜，-1 红方获胜，0 平局
        '''
        if not self.is_game_over():
            return None
        if self.get_color() == -1:
            if self.get_score() > self.get_balance_num():
                return 1
            elif self.get_score() == self.get_balance_num():
                return 0
            else:
                return -1
        else:
            if self.get_score() < -1 * self.get_balance_num():
                return -1
            elif self.get_score() == -1 * self.get_balance_num():
                return 0
            else:
                return 1


    def show_chessboard(self):
        '''打印棋盘'''
        format_board_value = self.format_simple_matrix(self.get_board_value())
        print(format_board_value)

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
    # # print(game.step, game.member_dict)
    # game.show_chessboard()
    # print(game.get_all_moves())
    # game.find_best_move(1, 1)
    print(game.format_matrix(game.chessboard))

