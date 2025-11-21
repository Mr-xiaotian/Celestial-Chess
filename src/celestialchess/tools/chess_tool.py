import random
import numpy as np
from numba import njit, types


@njit(types.boolean(types.float64[:, :, :], types.UniTuple(types.int32, 2)), cache=True)
def optimized_not_exist_zero_index(chessboard, board_range):
    row_len, col_len = board_range
    for row_idx in range(row_len):
        for col_idx in range(col_len):
            if chessboard[row_idx, col_idx, 0] == 0.0:
                return False
    return True


@njit(
    types.int32[:, :](types.float64[:, :, :], types.UniTuple(types.int32, 2)),
    cache=True,
)
def get_all_zero_index(chessboard, board_range):
    row_len, col_len = board_range
    move_list = np.empty((row_len * col_len, 2), dtype=np.int32)
    count = 0
    for row_idx in range(row_len):
        for col_idx in range(col_len):
            if chessboard[row_idx, col_idx, 0] == 0.0:
                # 直接赋值，而不是创建临时的元组
                move_list[count, 0] = row_idx
                move_list[count, 1] = col_idx
                count += 1
    return move_list[:count]  # 截取有效的部分


@njit(
    types.UniTuple(types.int32, 2)(
        types.float64[:, :, :], types.UniTuple(types.int32, 2)
    ),
    cache=True,
)
def get_random_zero_index(chessboard, board_range):
    row_len, col_len = board_range
    chosen_row, chosen_col = -1, -1
    count = 0

    for row_idx in range(row_len):
        for col_idx in range(col_len):
            if chessboard[row_idx, col_idx, 0] == 0.0:
                count += 1
                if random.randint(0, count - 1) == 0:
                    chosen_row, chosen_col = row_idx, col_idx

    return chosen_row, chosen_col


@njit(types.int32(types.float64[:, :, :], types.UniTuple(types.int32, 2)), cache=True)
def calculate_no_inf(chessboard, board_range):
    row_len, col_len = board_range
    total_score = 0
    for row_idx in range(row_len):
        for col_idx in range(col_len):
            score = chessboard[row_idx, col_idx, 0]
            if score != np.inf:
                total_score += score
    return total_score


@njit(
    types.float64[:, :](types.float64[:, :, :], types.UniTuple(types.int32, 2)),
    cache=True,
)
def get_first_channel(chessboard, board_range):
    rows, cols = board_range
    first_channel = np.empty((rows, cols), dtype=np.float64)
    for row_idx in range(rows):
        for col_idx in range(cols):
            first_channel[row_idx, col_idx] = chessboard[row_idx, col_idx, 0]
    return first_channel


@njit(cache=True)
def bfs_expand_with_power_threshold(
    chessboard, board_range, row, col, color, power, threshold
):
    row_len, col_len = board_range
    board_size = row_len * col_len
    max_size = board_size * 2

    # 第一层存储power_expand的visit信息, 第二层存储threshold_expand的visit信息
    virtual_rows = row_len + power - 2  # 允许中心落在下面几层, 此处数值经过计算
    visited = np.zeros((virtual_rows, col_len, 2), dtype=np.bool_)
    visited[row, col, 0] = True

    # 前board_size存储power_expand内容, 后board_size存储threshold_expand内容
    expand_queue_r = np.empty(max_size, dtype=np.int32)
    expand_queue_c = np.empty(max_size, dtype=np.int32)
    expand_queue_d = np.empty(max_size, dtype=np.int32)
    expand_queue_r[0] = row
    expand_queue_c[0] = col
    expand_queue_d[0] = power

    head, tail = 0, 1
    black_hole_power = power - 1
    over_threshold_max_index = board_size

    while head < tail:
        """更新落子点周围的格子，考虑黑洞点对路径的阻挡作用，同时只影响下方的格子"""
        r, c, distance = (
            expand_queue_r[head],
            expand_queue_c[head],
            expand_queue_d[head],
        )
        head += 1

        if distance <= 0 or chessboard[r, c, 0] == np.inf:
            continue

        visited[r, c, 0] = True

        chessboard[r, c, 0] += distance * color
        chessboard[r, c, 1] += distance

        if chessboard[r, c, 1] >= threshold:
            seed_r = r + black_hole_power - 1

            visited[seed_r, c, 1] = True
            expand_queue_r[over_threshold_max_index] = seed_r
            expand_queue_c[over_threshold_max_index] = c
            expand_queue_d[over_threshold_max_index] = black_hole_power
            over_threshold_max_index += 1

        for dr, dc in [(0, 1), (0, -1), (1, 0)]:
            nr, nc = r + dr, (c + dc) % col_len
            if nr >= row_len or visited[nr, nc, 0]:
                continue
            visited[nr, nc, 0] = True
            expand_queue_r[tail] = nr
            expand_queue_c[tail] = nc
            expand_queue_d[tail] = distance - 1
            tail += 1

    head = board_size

    while head < over_threshold_max_index:
        """扩张黑洞区域"""
        r, c, distance = (
            expand_queue_r[head],
            expand_queue_c[head],
            expand_queue_d[head],
        )
        head += 1

        if distance <= 0:
            continue

        visited[r, c, 1] = True
        if r < row_len:
            chessboard[r, c, 0] = np.inf

        for dr, dc in [(0, 1), (0, -1), (-1, 0)]:
            nr, nc = r + dr, (c + dc) % col_len
            if visited[nr, nc, 1]:
                continue
            visited[nr, nc, 1] = True
            expand_queue_r[over_threshold_max_index] = nr
            expand_queue_c[over_threshold_max_index] = nc
            expand_queue_d[over_threshold_max_index] = distance - 1
            over_threshold_max_index += 1


@njit(cache=True)
def go_random_simulate(
    chessboard, board_range, current_color, power, threshold, balance_num
):
    row_len, col_len = board_range
    board_size = row_len * col_len
    max_size = board_size * 2
    black_hole_power = power - 1

    # 前board_size存储power_expand内容, 后board_size存储threshold_expand内容
    expand_queue_r = np.empty(max_size, dtype=np.int32)
    expand_queue_c = np.empty(max_size, dtype=np.int32)
    expand_queue_d = np.empty(max_size, dtype=np.int32)

    count = 0
    for row_idx in range(row_len):
        for col_idx in range(col_len):
            if chessboard[row_idx, col_idx, 0] == 0.0:
                count += 1
                if random.randint(0, count - 1) == 0:
                    chosen_row, chosen_col = row_idx, col_idx
    while count != 0:
        # 第一层存储power_expand的visit信息, 第二层存储threshold_expand的visit信息
        virtual_rows = row_len + power - 2  # 允许中心落在下面几层, 此处数值经过计算
        visited = np.zeros((virtual_rows, col_len, 2), dtype=np.bool_)
        visited[chosen_row, chosen_col, 0] = True

        expand_queue_r[0] = chosen_row
        expand_queue_c[0] = chosen_col
        expand_queue_d[0] = power

        head, tail = 0, 1
        over_threshold_max_index = board_size

        while head < tail:
            """更新落子点周围的格子，考虑黑洞点对路径的阻挡作用，同时只影响下方的格子"""
            r, c, distance = (
                expand_queue_r[head],
                expand_queue_c[head],
                expand_queue_d[head],
            )
            head += 1

            if distance <= 0 or chessboard[r, c, 0] == np.inf:
                continue

            visited[r, c, 0] = True

            chessboard[r, c, 0] += distance * current_color
            chessboard[r, c, 1] += distance

            if chessboard[r, c, 1] >= threshold:
                seed_r = r + black_hole_power - 1

                visited[seed_r, c, 1] = True
                expand_queue_r[over_threshold_max_index] = seed_r
                expand_queue_c[over_threshold_max_index] = c
                expand_queue_d[over_threshold_max_index] = black_hole_power
                over_threshold_max_index += 1

            for dr, dc in [(0, 1), (0, -1), (1, 0)]:
                nr, nc = r + dr, (c + dc) % board_range[1]
                if nr >= board_range[0] or visited[nr, nc, 0]:
                    continue
                visited[nr, nc, 0] = True
                expand_queue_r[tail] = nr
                expand_queue_c[tail] = nc
                expand_queue_d[tail] = distance - 1
                tail += 1

        head = board_size

        while head < over_threshold_max_index:
            """扩张黑洞区域"""
            r, c, distance = (
                expand_queue_r[head],
                expand_queue_c[head],
                expand_queue_d[head],
            )
            head += 1

            if distance <= 0:
                continue

            visited[r, c, 1] = True
            if r < board_range[0]:
                chessboard[r, c, 0] = np.inf

            for dr, dc in [(0, 1), (0, -1), (-1, 0)]:
                nr, nc = r + dr, (c + dc) % board_range[1]
                if visited[nr, nc, 1]:
                    continue
                visited[nr, nc, 1] = True
                expand_queue_r[over_threshold_max_index] = nr
                expand_queue_c[over_threshold_max_index] = nc
                expand_queue_d[over_threshold_max_index] = distance - 1
                over_threshold_max_index += 1

        current_color *= -1

        count = 0
        for row_idx in range(row_len):
            for col_idx in range(col_len):
                if chessboard[row_idx, col_idx, 0] == 0.0:
                    count += 1
                    if random.randint(0, count - 1) == 0:
                        chosen_row, chosen_col = row_idx, col_idx

    comparison_score = current_color * balance_num - balance_num
    for row_index in range(row_len):
        for col_index in range(col_len):
            score = chessboard[row_index, col_index, 0]
            if score != np.inf:
                comparison_score += score

    if comparison_score > 0:
        return 1
    elif comparison_score < 0:
        return -1
    else:
        return 0


def format_matrix(matrix, decimal_places=(0, 0)):
    """
    格式化矩阵
    :param matrix: 矩阵
    :param decimal_places: 保留的小数点位数 (第一个数的位数, 第二个数的位数)
    :return: 格式化后的字符串
    """
    # 确定每个元素的最大宽度
    max_widths = [
        max(
            len(f"{sublist[idx]:.{decimal_place}f}")
            for row in matrix
            for sublist in row
        )
        for idx, decimal_place in enumerate(decimal_places)
    ]

    formatted_rows = []
    for row in matrix:
        formatted_row = "  ["
        for sublist in row:
            formatted_row += (
                "["
                + ", ".join(
                    f"{item:>{max_widths[item_idx]}.{decimal_places[item_idx]}f}"
                    for item_idx, item in enumerate(sublist)
                )
                + "], "
            )
        formatted_row = formatted_row.rstrip(", ") + "]"
        formatted_rows.append(formatted_row)

    formatted_string = "[\n" + ",\n".join(formatted_rows) + "\n]"
    return formatted_string


def format_simple_matrix(matrix):
    """
    格式化简单矩阵
    :param matrix: 矩阵
    :return: 格式化后的字符串
    """
    # 确定每个元素的最大宽度
    max_width = max(len(str(item)) for row in matrix for item in row)

    formatted_rows = []
    for row in matrix:
        formatted_row = (
            "  [" + ", ".join(f"{item:>{max_width}}" for item in row) + "]"
        )
        formatted_rows.append(formatted_row)

    formatted_string = "[\n" + ",\n".join(formatted_rows) + "\n]"
    return formatted_string