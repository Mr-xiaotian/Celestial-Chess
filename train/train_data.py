import numpy as np
import pickle
from pathlib import Path
from time import strftime, localtime
from celestialflow import TaskManager

from celestialchess import ChessGame, MCTSAI, ai_battle


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
                board = self.process_board(history_board[step], step)
                all_training_data.append((board, history_move[step + 1]))
        return all_training_data
    
    def process_board(self, chess_board, step):
        color = 1 if step % 2 == 0 else -1
        color_channel = np.full((5, 5, 1), color)
        processed_board = np.concatenate((chess_board, color_channel), axis=2)
        
        for row in processed_board:
            for cell in row:
                if cell[0] == float("inf"):
                    cell[0] = 5
        return processed_board


def start_train_data(train_num: int, mcts_iter: int=1000, execution_mode: str='serial'):
    train_data_threader = TrainDataThread(
            ai_battle,
            execution_mode = execution_mode,
            progress_desc='TrainDataProcess',
            show_progress=True
            )
    
    mcts_ai = MCTSAI(mcts_iter, complete_mode=False)
    train_data_threader.set_ai(mcts_ai, mcts_ai)

    train_data_threader.start(range(train_num))
    all_training_data = train_data_threader.process_result_dict()
    
    return all_training_data

def save_train_data(data, train_num, mcts_iter):
    """
    保存训练数据到指定路径
    """
    data_size = len(data)
    now_data = strftime("%Y-%m-%d", localtime())
    now_time = strftime("%H-%M", localtime())
    
    parent_path = Path(f'./data/train_data/{now_data}')
    parent_path.mkdir(parents=True, exist_ok=True)

    mcts_iter_str = f'MCTS{mcts_iter // 1000}k'
    train_num_str = f'Games{train_num // 1000}k' if train_num > 1000 else f'Games{train_num}'
    data_size_str = f'Length{data_size}'

    train_data_path = f"{parent_path}/{mcts_iter_str}_{train_num_str}_{data_size_str}({now_time}).pkl"
    pickle.dump(data, open(train_data_path, "wb"))

    return train_data_path

def load_train_data(file_path):
    """
    从指定路径加载训练数据
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data
