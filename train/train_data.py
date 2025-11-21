import json
from pathlib import Path
from time import strftime, localtime
from celestialflow import TaskManager

from celestialchess import ChessGame, MCTSAI, ai_battle, process_board


class TrainDataThread(TaskManager):
    def set_ai(self, ai_0, ai_1):
        self.ai_0 = ai_0
        self.ai_1 = ai_1

    def get_args(self, obj: object):
        train_game = ChessGame((5, 5), 2)
        train_game.init_cfunc()
        train_game.init_history()
        return (self.ai_0, self.ai_1, train_game, False)

    def process_result_dict(self):
        all_training_data = []
        success_dict = self.get_success_dict()
        for over_game in success_dict.values():
            history_board = over_game.history_board
            history_move = over_game.history_move
            for step in range(over_game.max_step - 1):
                color = 1 if step % 2 == 0 else -1
                board = process_board(history_board[step], color)
                all_training_data.append((board, history_move[step + 1]))
        return all_training_data


def start_train_data(
    train_num: int, mcts_iter: int = 1000, execution_mode: str = "serial"
):
    train_data_threader = TrainDataThread(
        ai_battle,
        execution_mode=execution_mode,
        worker_limit = 5,
        enable_result_cache=True,
        progress_desc="TrainDataProcess",
        show_progress=True,
    )

    mcts_ai = MCTSAI(mcts_iter, enable_cps=False)
    train_data_threader.set_ai(mcts_ai, mcts_ai)

    train_data_threader.start(range(train_num))
    all_training_data = train_data_threader.process_result_dict()

    return all_training_data


def convert_training_data_to_json_format(data):
    """
    将训练数据转换为 JSON 可序列化格式
    data: list of (np.array(5,5,4), (row, col))
    """
    json_list = []
    for board, move in data:
        json_list.append({
            "board": board.tolist(),   # numpy -> python list
            "move": [int(move[0]), int(move[1])]   # int32 -> int
        })
    return json_list


def save_train_data(data, train_num, mcts_iter):
    """
    以 JSON 格式保存训练数据
    """
    # 先转换格式
    json_data = convert_training_data_to_json_format(data)
    
    now_date = strftime("%Y-%m-%d", localtime())
    now_time = strftime("%H-%M", localtime())

    parent_path = Path(f"./data/train_data/{now_date}")
    parent_path.mkdir(parents=True, exist_ok=True)

    mcts_iter_str = f"MCTS{mcts_iter/1000}k"
    train_num_str = (
        f"Games{train_num/1000}k" if train_num > 1000 else f"Games{train_num}"
    )
    data_size_str = f"Length{len(data)}"

    file_path = (
        f"{parent_path}/{mcts_iter_str}_{train_num_str}_{data_size_str}({now_time}).json"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)

    return file_path


def load_train_data(file_path):
    """
    从 JSON 文件中加载训练数据，恢复为 numpy 格式
    """
    import numpy as np

    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    data = []
    for sample in json_data:
        board_np = np.array(sample["board"], dtype=float)
        move = tuple(sample["move"])
        data.append((board_np, move))

    return data

