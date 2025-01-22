import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle, re, json
import torch
import numpy as np
from pathlib import Path
from ai import ai_battle, MinimaxAI, MCTSAI
from game.chess_game import ChessGame
from CelestialVault.instances.inst_task import ExampleTaskManager
from time import strftime, localtime


class TrainDataThread(ExampleTaskManager):
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
        result_dict = self.get_result_dict()
        for over_game in result_dict.values():
            history_board = over_game.history_board
            history_move = over_game.history_move
            for step in range(over_game.max_step-1):
                board = self.process_board(history_board[step], step)
                # if (board, history_move[step+1]) in all_training_data: # 这样效果并不好
                #     continue
                all_training_data.append((board, history_move[step+1]))
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

def load_train_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def save_train_data(data, train_num, mcts_iter):
    data_size = len(data)
    now_data = strftime("%Y-%m-%d", localtime())
    now_time = strftime("%H-%M", localtime())
    
    parent_path = Path(f'train_data/{now_data}')
    parent_path.mkdir(parents=True, exist_ok=True)

    mcts_iter_str = f'MCTS{mcts_iter//1000}k'
    train_num_str = f'Games{train_num//1000}k' if train_num > 1000 else f'Games{train_num}'
    data_size_str = f'Length{data_size}'

    pickle.dump(data, open(f"{parent_path}/{mcts_iter_str}_{train_num_str}_{data_size_str}({now_time}).pkl", "wb"))

def start_train_data(train_num: int, mcts_iter: int=1000, execution_mode: str='serial'):
    train_data_threader = TrainDataThread(
            ai_battle,
            execution_mode = execution_mode,
            progress_desc='trainDataProcess',
            show_progress=True
            )
    
    mcts_ai = MCTSAI(mcts_iter, complete_mode=False)
    train_data_threader.set_ai(mcts_ai, mcts_ai)

    train_data_threader.start(range(train_num))
    all_training_data = train_data_threader.process_result_dict()

    save_train_data(all_training_data, train_num, mcts_iter)
    
    return all_training_data


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ai.deeplearning import ChessPolicyModel

class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, move = self.data[idx]
        board_state = torch.tensor(board_state, dtype=torch.float32)
        move = move[0] * 5 + move[1]
        return board_state, move
    
class ModelTrainer:
    def __init__(self):
        # 设置CuDNN选项
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, train_data, num_epochs = 10):
        model = ChessPolicyModel().to(self.device) # 初始化模型，并将其移动到GPU上
        criterion = nn.CrossEntropyLoss() # 定义交叉熵损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001) # 定义Adam优化器
        dataset = ChessDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        train_log_text = []
        # 训练循环
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                # 调整输入的维度，并将其移动到GPU上
                # inputs 的原始形状是 (batch_size, height, width, channels)，也就是 (32, 5, 5, 3)
                # inputs.permute(0, 3, 1, 2) 会将 inputs 的维度从 (32, 5, 5, 3) 转换为 (32, 3, 5, 5)
                inputs = inputs.permute(0, 3, 1, 2).to(self.device)  # (batch_size, channels, height, width)
                labels = labels.to(self.device).to(torch.int64)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)
                # 计算损失
                loss = criterion(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新参数
                optimizer.step()

                # 累积损失
                running_loss += loss.item()
                if i % 100 == 99:  # 每100个batch打印一次loss
                    log_text = f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}'
                    print(log_text)
                    train_log_text.append(log_text)
                    running_loss = 0.0

        data_size = len(train_data)
        model_path = self.save_model(model, data_size)
        self.save_model_loss(train_log_text, data_size)

        return model_path, model

    def save_model(self, model, data_size):
        now_data = strftime("%Y-%m-%d", localtime())
        now_time = strftime("%H-%M", localtime())

        parent_path = Path(f'models/{now_data}')
        parent_path.mkdir(parents=True, exist_ok=True)

        model_path = f'{parent_path}/dl_model({now_time})({data_size}).pth'
        torch.save(model.state_dict(), model_path)

        return model_path

    def save_model_loss(self, train_log_text, data_size):
        now_data = strftime("%Y-%m-%d", localtime())
        now_time = strftime("%H-%M", localtime())

        parent_path = Path(f'models_loss/{now_data}')
        parent_path.mkdir(parents=True, exist_ok=True)

        model_loss_path = f'{parent_path}/dl_model({now_time})({data_size}).txt'
        with open(model_loss_path, 'w') as f:
            f.write('\n'.join(train_log_text))


def get_model_info_dict(path, train_data_path, model):
    model_info_dict = {}
    model_info_dict["path"] = path
    model_info_dict["train_data_path"] = train_data_path

    model_str = str(model)
    model_lines = re.split(r'\n', model_str)
    layer_dict = {}
    for line in model_lines[1:-1]:  # 跳过开头和结尾的行
        re_ = re.compile(r"\((.*?)\): (.*)")
        layer_name = re_.search(line).group(1)
        layer_args = re_.search(line).group(2)
        
        layer_dict[layer_name] = layer_args
    model_info_dict["layers"] = layer_dict

    return model_info_dict

def save_model_info_dict(info_dict, model_type):
    with open('model_score.json', 'r') as f:
        model_score = json.load(f)

    model_score[model_type].append(info_dict)

    with open('model_score.json', 'w') as f:
        json.dump(model_score, f, indent=2)