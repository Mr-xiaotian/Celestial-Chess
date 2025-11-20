import json, re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from time import strftime, localtime

from celestialchess.ai import get_model_score_by_mcts
from celestialchess.ai.deeplearning import ChessPolicyModel, DeepLearningAI


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, move = self.data[idx]
        board_state = torch.tensor(board_state, dtype=torch.float32)
        move = move[0] * 5 + move[1]  # 将 (row, col) 转换为索引
        return board_state, move


class ModelTrainer:
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(self, train_data, num_epochs=10):
        model = ChessPolicyModel().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        dataset = ChessDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        train_log_text = []
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.permute(0, 3, 1, 2).to(self.device)
                labels = labels.to(self.device).to(torch.int64)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    log_text = f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}"
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

        parent_path = Path(f"./data/models/{now_data}")
        parent_path.mkdir(parents=True, exist_ok=True)

        model_path = f"{parent_path}/dl_model({now_time})({data_size}).pth"
        torch.save(model.state_dict(), model_path)
        return model_path

    def save_model_loss(self, train_log_text, data_size):
        now_data = strftime("%Y-%m-%d", localtime())
        now_time = strftime("%H-%M", localtime())

        parent_path = Path(f"./data/models_loss/{now_data}")
        parent_path.mkdir(parents=True, exist_ok=True)

        model_loss_path = f"{parent_path}/dl_model_loss({now_time})({data_size}).txt"
        with open(model_loss_path, "w") as f:
            f.write("\n".join(train_log_text))


def get_model_info_dict(model_path, train_data_path, model):
    model_info_dict = {}
    model_info_dict["model_path"] = model_path
    model_info_dict["train_data_path"] = train_data_path

    model_str = str(model)
    model_lines = re.split(r"\n", model_str)
    layer_dict = {}
    for line in model_lines[1:-1]:  # 跳过开头和结尾的行
        re_ = re.compile(r"\((.*?)\): (.*)")
        layer_name = re_.search(line).group(1)
        layer_args = re_.search(line).group(2)

        layer_dict[layer_name] = layer_args
    model_info_dict["layers"] = layer_dict

    game_state = ((5, 5), 2)
    dl_model = DeepLearningAI(model_path, complete_mode=False)
    score, score_dict = get_model_score_by_mcts(dl_model, game_state)
    now_data = strftime("%Y-%m-%d", localtime())
    test_dict = {
        now_data: {"complete_mode": False, "score": score, "score_dict": score_dict}
    }
    model_info_dict["tests"] = test_dict

    return model_info_dict


def save_model_info_dict(info_dict, model_type):
    with open(".model_score.json", "r") as f:
        model_score = json.load(f)

    model_score[model_type].append(info_dict)

    with open(".model_score.json", "w") as f:
        json.dump(model_score, f, indent=2)
