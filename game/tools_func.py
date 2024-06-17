import numpy as np
from numba import njit
from collections import deque

@njit
def get_zero_index(chessboard, board_range, execute=True):
    if not execute:
        return
    move_list = []
    for row_idx in range(board_range[0]):
        for col_idx in range(board_range[1]):
            if chessboard[row_idx, col_idx, 0] == 0:
                move_list.append((row_idx, col_idx))
    return move_list

@njit
def get_first_channel(chessboard, execute=True):
    if not execute:
        return
    return [[cell[0] for cell in row] for row in chessboard]

@njit
def update_by_bfs(chessboard, row, col, color, power, board_range, execute=True):
    if not execute:
        return
    visited = np.zeros((board_range[0], board_range[1]), dtype=np.bool_)
    queue = np.empty((board_range[0] * board_range[1], 3), dtype=np.int32)
    head, tail = 0, 1
    queue[head] = (row, col, power)

    while head < tail:
        r, c, distance = queue[head]
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
            queue[tail] = (nr, nc%board_range[1], distance - 1)
            tail += 1

    return visited