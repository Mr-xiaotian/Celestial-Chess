"""
Author: 晓天
Vision: 1.1
"""
import pickle, hashlib
from time import time
from copy import deepcopy
from collections import deque
from pprint import pprint


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
        self.transposition_table = {}
        self.transposition_table_change = False

        try:
            with open(self.transposition_file, "rb") as file:
                self.transposition_table = pickle.load(file)
        except (FileNotFoundError, EOFError):
            with open(self.transposition_file, "wb") as file:
                pickle.dump(self.transposition_table, file)

    def update_transposition_table(self, key, value):
        # 检查是否存在更深层次的搜索结果
        if (
            key not in self.transposition_table
            or self.transposition_table[key]["depth"] < value["depth"]
        ):
            self.transposition_table[key] = value
            self.transposition_table_change = True

    def save_transposition_table(self):
        if not self.transposition_table_change:
            return
        with open(self.transposition_file, "wb") as file:
            pickle.dump(self.transposition_table, file)

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
        for row in range(len(self.chessboard)):
            for col in range(len(self.chessboard[0])):
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
        # 计算棋盘上所有非无穷大格子的总分数
        total_score = sum(
            cell[0]
            for row in self.chessboard
            for cell in row
            if cell[0] != float("inf")
        )
        balance_num = self.power * (self.power + 1) * (2 * self.power + 1) / 12
        return total_score - balance_num

    def get_all_moves(self):
        # 创建一个空列表，用于存放棋盘上所有格子的坐标
        move_list = []
        for row_idx, row in enumerate(self.chessboard):
            for col_idx, cell in enumerate(row):
                if abs(cell[0]) == 0:
                    move_list.append((row_idx, col_idx))
        return move_list
    

    def get_board_key(self):
        # 将数据转换为字符串形式
        data_str = str(self.chessboard)
        # 使用 hashlib 库中的 sha256 函数来生成哈希值
        hash_object = hashlib.sha256(data_str.encode())
        # 获取十六进制表示的哈希值
        hash_hex = hash_object.hexdigest()
        return hash_hex

    def is_game_over(self):
        '''
        判断游戏是否结束
        :return: 游戏是否结束
        '''
        for row in self.chessboard:
            for cell in row:
                if cell[0] == 0:
                    return False
        return True

    def minimax(self, depth, color, alpha, beta):
        board_key = self.get_board_key()
        if board_key in self.transposition_table and \
            self.transposition_table[board_key]['depth'] >= depth:
            return self.transposition_table[board_key]['score']
        
        if depth == 0 or self.is_game_over():
            score = self.get_score()
            self.update_transposition_table(
                board_key,  {'score': score, 'depth': depth})
            return score

        if color == 1:
            max_eval = float("-inf")
            for move in self.get_all_moves():
                self.update_chessboard(*move, color)
                eval = self.minimax(depth - 1, -1, alpha, beta)
                self.undo()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            self.update_transposition_table(
                board_key,  {'score': max_eval, 'depth': depth})
            return max_eval
        elif color == -1:
            min_eval = float("inf")
            for move in self.get_all_moves():
                self.update_chessboard(*move, color)
                eval = self.minimax(depth - 1, 1, alpha, beta)
                self.undo()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            self.update_transposition_table(
                board_key,  {'score': min_eval, 'depth': depth})
            return min_eval

    def find_best_move(self, color, depth=3):
        '''
        定义函数find_best_move，用于在搜索树中搜索最佳移动
        参数color表示玩家颜色
        参数depth表示搜索深度
        回最佳移动
        '''
        best_move = None

        if color == 1:
            best_score = float("-inf")
            for move in self.get_all_moves():
                self.update_chessboard(*move, color)
                score = self.minimax(depth, color * -1, float("-inf"), float("inf"))
                self.undo()
                if score > best_score:
                    best_score = score
                    best_move = move

        elif color == -1:
            best_score = float("inf")
            for move in self.get_all_moves():
                self.update_chessboard(*move, color)
                score = self.minimax(depth, color * -1, float("-inf"), float("inf"))
                self.undo()
                if score < best_score:
                    best_score = score
                    best_move = move

        self.save_transposition_table()

        return best_move

    def show_chessboard(self):
        '''打印棋盘'''
        pprint([[cell[0] for cell in row] for row in self.chessboard])

    def test_time(self, depth = 3):
        self.restart()

        first_time = time()
        while True:
            last_time = time()
            color = 1 if self.step%2==0 else -1
            move = self.find_best_move(color, depth)
            print(time()-last_time)
            self.update_chessboard(*move, color)
            self.show_chessboard()
            print()
            # sleep(0)
            if self.is_game_over():
                print(f'总分数:{self.get_score()}\n用时:{time()-first_time}')
                break

if __name__ == "__main__":
    game = ChessGame((5,5), power=2)
    # game.update_chessboard(2, 2, 1)
    # # game.update_chessboard(2,1,-1)
    # # game.undo()
    # # print(game.step, game.member_dict)
    # game.show_chessboard()
    # print(game.get_all_moves())
    # game.find_best_move(1, 1)

    game.test_time(5)
    
    print(len(game.transposition_table.keys()))
