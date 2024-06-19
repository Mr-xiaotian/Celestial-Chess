import numpy as np
from numba import njit, types


@njit(types.boolean(types.float64[:, :, :]))
def optimized_not_exist_zero_index(chessboard):
    return not np.any(chessboard[:, :, 0] == 0)

@njit(types.int32[:, :](types.float64[:, :, :], types.UniTuple(types.int32, 2)))
def get_zero_index(chessboard, board_range):
    move_list = np.empty((board_range[0] * board_range[1], 2), dtype=np.int32)
    count = 0
    for row_idx in range(board_range[0]):
        for col_idx in range(board_range[1]):
            if chessboard[row_idx, col_idx, 0] == 0:
                move_list[count] = (row_idx, col_idx)
                count += 1
    return move_list[:count]  # 截取有效的部分

@njit(types.int32(types.float64[:, :, :]))
def calculate_no_inf(chessboard):
    total_score = 0
    for row in chessboard:
        for cell in row:
            if cell[0] != np.inf:
                total_score += cell[0]
    return total_score

@njit(types.float64[:, :](types.float64[:, :, :], types.UniTuple(types.int32, 2)))
def get_first_channel(chessboard, board_range):
    rows, cols = board_range
    first_channel = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            first_channel[i, j] = chessboard[i, j, 0]
    return first_channel

@njit
def expand_by_bfs_and_threshold(chessboard, board_range, row, col, color, threshold, power):
    visited = np.zeros((board_range[0], board_range[1]), dtype=np.bool_)

    expand_queue_r = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    expand_queue_c = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    expand_queue_d = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    head, tail = 0, 1
    expand_queue_r[0] = row
    expand_queue_c[0] = col
    expand_queue_d[0] = power

    over_threshold_queue_r = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    over_threshold_queue_c = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    over_threshold_queue_d = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    over_threshold_len = 0
    black_power = power - 1

    while head < tail:
        """更新落子点周围的格子，考虑黑洞点对路径的阻挡作用，同时只影响下方的格子"""
        r, c, distance = expand_queue_r[head], expand_queue_c[head], expand_queue_d[head]
        head += 1

        if visited[r, c] or distance <= 0 or chessboard[r, c, 0] == np.inf:
            continue

        visited[r, c] = True

        chessboard[r, c, 0] += distance * color
        chessboard[r, c, 1] += distance

        if chessboard[r, c, 1] >= threshold:
            over_threshold_queue_r[over_threshold_len] = r + black_power - 1
            over_threshold_queue_c[over_threshold_len] = c
            over_threshold_queue_d[over_threshold_len] = black_power
            over_threshold_len += 1

        for dr, dc in [(0, 1), (0, -1), (1, 0)]:
            nr, nc = r + dr, c + dc
            if nr >= board_range[0]:
                continue
            expand_queue_r[tail] = nr
            expand_queue_c[tail] = nc % board_range[1]
            expand_queue_d[tail] = distance - 1
            tail += 1

    head = 0
    visited_expansion = np.zeros((board_range[0], board_range[1]), dtype=np.bool_)

    while head < over_threshold_len:
        """扩张黑洞区域"""
        r, c, distance = over_threshold_queue_r[head], over_threshold_queue_c[head], over_threshold_queue_d[head]
        head += 1

        if visited_expansion[r, c] or distance <= 0:
            continue
        
        if r < board_range[0]:
            chessboard[r, c, 0] = np.inf
        visited_expansion[r, c] = True
        
        for dr, dc in [(0, 1), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            over_threshold_queue_r[over_threshold_len] = nr
            over_threshold_queue_c[over_threshold_len] = nc % board_range[1]
            over_threshold_queue_d[over_threshold_len] = distance - 1
            over_threshold_len += 1

