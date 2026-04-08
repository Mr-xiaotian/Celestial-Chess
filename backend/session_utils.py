from typing import List


def parse_options(tokens: List[str]):
    """
    解析命令行风格参数列表。

    参数格式示例:
        --row 3 --col 4 --color 1

    解析结果示例:
        {"row": "3", "col": "4", "color": "1"}

    规则:
        - 以 `--` 开头的 token 被视为键；
        - 键后若跟随非 `--` token，则作为该键的值；
        - 若键后无值，则该键映射为 True（布尔开关参数）。
    """
    opts = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                opts[key] = tokens[i + 1]
                i += 2
            else:
                opts[key] = True
                i += 1
        else:
            i += 1
    return opts


def convert_inf_to_string(value):
    """
    将 Python 的正无穷转换为可序列化字符串。

    参数:
        value: 任意值。

    返回:
        - value 为 float("inf") 时返回 "inf"
        - 其他情况原样返回
    """
    if value == float("inf"):
        return "inf"
    else:
        return value


def prepare_board_for_json(board):
    """
    将棋盘二维结构转换为 JSON 友好格式。

    处理内容:
        - 遍历每个格子 `[value, load]`
        - 将 value 中的无穷值转换为字符串 "inf"
    """
    return [
        [[convert_inf_to_string(cell[0]), cell[1]] for cell in row]
        for row in board
    ]
