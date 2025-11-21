import numpy as np


def process_board(chess_board: np.ndarray, color: int) -> np.ndarray:
    """
    构造训练用棋盘输入张量（5×5×4）
    通道含义：
    0: value（inf 替换为 0）
    1: load
    2: color（当前行动方）
    3: blackhole_flag（1=黑洞）
    """
    rows, cols, channels = chess_board.shape

    # 当前行动方颜色
    color_channel = np.full((rows, cols), color, dtype=float)

    # 处理 value 通道：将 inf 替换为 0
    value_channel = chess_board[:, :, 0].copy()
    value_channel[np.isinf(value_channel)] = 0.0

    # load 通道保持不变
    load_channel = chess_board[:, :, 1].copy()

    # 黑洞通道：inf → 1，其他 → 0
    blackhole_channel = np.isinf(chess_board[:, :, 0]).astype(float)

    # 组合成最终 4 通道输入
    processed_board = np.stack(
        [value_channel, load_channel, color_channel, blackhole_channel],
        axis=2
    )

    return processed_board