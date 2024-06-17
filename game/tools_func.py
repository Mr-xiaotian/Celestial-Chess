from numba import njit
from collections import deque

@njit
def get_zero_index(chessboard):
    move_list = []
    for row_idx in range(chessboard.shape[0]):
        for col_idx in range(chessboard.shape[1]):
            if chessboard[row_idx, col_idx, 0] == 0:
                move_list.append((row_idx, col_idx))
    return move_list

@njit
def get_first_channel(chessboard):
    return [[cell[0] for cell in row] for row in chessboard]


def update_adjacent_cells(self, row, col, color):
        """更新落子点周围的格子，考虑黑洞点对路径的阻挡作用，同时只影响下方的格子"""
        visited = set()
        queue = deque([(row, col, 0)])  # 使用队列进行BFS搜索

        while queue:
            r, c, distance = queue.popleft()
            if (r, c) in visited:
                continue
            elif distance >= self.power:
                continue
            elif self.chessboard[r][c][0] is self.BLACK_HOLE:  # 检查是否为黑洞点
                continue

            visited.add((r, c))

            change = max(self.power - distance, 0)
            self.chessboard[r][c][0] += change * color
            self.chessboard[r][c][1] += change
            
            # 只向左侧、右侧和下方扩展搜索区域
            for dr, dc in [(0, 1), (0, -1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.board_range[0]
                    and 0 <= nc < self.board_range[1]
                ):
                    queue.append((nr, nc, distance + 1))

        return visited