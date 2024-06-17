import numpy as np
from numba import njit
from collections import deque

@njit
def get_zero_index(chessboard, board_range):
    move_list = []
    for row_idx in range(board_range[0]):
        for col_idx in range(board_range[1]):
            if chessboard[row_idx, col_idx, 0] == 0:
                move_list.append((row_idx, col_idx))
    return move_list

@njit
def get_first_channel(chessboard):
    return [[cell[0] for cell in row] for row in chessboard]

@njit
def update_by_bfs(chessboard, row, col, color, power, board_range):
        visited = set()
        queue = deque([(row, col, 0)])  # 使用队列进行BFS搜索

        while queue:
            r, c, distance = queue.popleft()
            if (r, c) in visited:
                continue
            elif distance >= power:
                continue
            elif chessboard[r][c][0] is np.inf:  # 检查是否为黑洞点
                continue

            visited.add((r, c))

            change = max(power - distance, 0)
            chessboard[r][c][0] += change * color
            chessboard[r][c][1] += change
            
            # 只向左侧、右侧和下方扩展搜索区域
            for dr, dc in [(0, 1), (0, -1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < board_range[0]
                    and 0 <= nc < board_range[1]
                ):
                    queue.append((nr, nc, distance + 1))

        return visited