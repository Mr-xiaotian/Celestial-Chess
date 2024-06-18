import numpy as np
from numba import njit


@njit
def optimized_not_exist_zero_index(chessboard):
    return not np.any(chessboard[:, :, 0] == 0)

@njit
def get_zero_index(chessboard, board_range):
    move_list = np.empty((board_range[0] * board_range[1], 2), dtype=np.int32)
    count = 0
    for row_idx in range(board_range[0]):
        for col_idx in range(board_range[1]):
            if chessboard[row_idx, col_idx, 0] == 0:
                move_list[count] = (row_idx, col_idx)
                count += 1
    return move_list[:count]  # 截取有效的部分

@njit
def get_first_channel(chessboard):
    return [[cell[0] for cell in row] for row in chessboard]

@njit
def update_by_bfs(chessboard, row, col, color, power, board_range):
    visited = np.zeros((board_range[0], board_range[1]), dtype=np.bool_)
    queue_r = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    queue_c = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    queue_d = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    head, tail = 0, 1
    queue_r[0] = row
    queue_c[0] = col
    queue_d[0] = power

    while head < tail:
        r, c, distance = queue_r[head], queue_c[head], queue_d[head]
        head += 1

        if visited[r, c] or distance <= 0 or chessboard[r, c, 0] == np.inf:
            continue

        visited[r, c] = True

        chessboard[r, c, 0] += distance * color
        chessboard[r, c, 1] += distance

        for dr, dc in [(0, 1), (0, -1), (1, 0)]:
            nr, nc = r + dr, c + dc
            if nr >= board_range[0]:
                continue
            queue_r[tail] = nr
            queue_c[tail] = nc%board_range[1]
            queue_d[tail] = distance - 1
            tail += 1

    return visited

@njit
def mark_and_expand_over_threshold(chessboard, visited, board_range, threshold, power):
    queue_r = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    queue_c = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    queue_d = np.empty(board_range[0] * board_range[1], dtype=np.int32)
    queue_len = 0
    black_power = power - 1

    for row_idx in range(board_range[0]):
        for col_idx in range(board_range[1]):
            if not visited[row_idx, col_idx]:
                continue
            if chessboard[row_idx, col_idx, 1] >= threshold:
                queue_r[queue_len] = row_idx + black_power - 1
                queue_c[queue_len] = col_idx
                queue_d[queue_len] = black_power
                queue_len += 1

    visited_expansion = np.zeros((board_range[0], board_range[1]), dtype=np.bool_)

    head = 0
    while head < queue_len:
        r, c, distance = queue_r[head], queue_c[head], queue_d[head]
        head += 1

        if visited_expansion[r, c] or distance <= 0:
            continue
        
        if not r >= board_range[0]:
            chessboard[r, c, 0] = np.inf
        visited_expansion[r, c] = True
        
        for dr, dc in [(0, 1), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            queue_r[queue_len] = nr
            queue_c[queue_len] = nc%board_range[1]
            queue_d[queue_len] = distance - 1
            queue_len += 1
