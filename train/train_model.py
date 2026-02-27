import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from pathlib import Path
from time import strftime, localtime

from celestialchess import get_model_score_by_mcts, ChessPolicyModel, DeepLearningAI


class ChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board_state, move = self.data[idx]
        board_state = torch.as_tensor(board_state, dtype=torch.float32)
        col_len = board_state.shape[1]
        move = move[0] * col_len + move[1]
        return board_state, move


class ModelTrainer:
    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 0.001,
        log_interval: int = 100,
        num_workers: int = 0,
        pin_memory: bool = None,
    ):
        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

    def train_model(self, train_data, num_epochs=10):
        model = ChessPolicyModel().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        dataset = ChessDataset(train_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

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
                if self.log_interval > 0 and i % self.log_interval == self.log_interval - 1:
                    avg_loss = running_loss / self.log_interval
                    log_text = f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss:.3f}"
                    print(log_text)
                    train_log_text.append(log_text)
                    running_loss = 0.0

        data_size = len(train_data)
        model_path = self.save_model(model, data_size)
        self.save_model_loss(train_log_text, data_size)

        sample_board = train_data[0][0]
        channels = sample_board.shape[2]
        rows = sample_board.shape[0]
        cols = sample_board.shape[1]
        summary(model, input_size=(channels, rows, cols))
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


def get_model_info_dict(model_path, train_data_path, model, chess_state=None):
    model_info_dict = {}
    model_info_dict["model_path"] = model_path
    model_info_dict["train_data_path"] = train_data_path

    model_str = str(model)
    model_lines = model_str.split("\n")

    layer_dict = {}
    pattern = re.compile(r"\((.*?)\): (.*)")

    for line in model_lines:
        line = line.strip()
        if "): " not in line:
            continue

        m = pattern.match(line)
        if not m:
            continue

        layer_name, layer_args = m.groups()
        layer_dict[layer_name] = layer_args

    model_info_dict["layers"] = layer_dict

    game_state = chess_state or ((5, 5), 2)
    dl_model = DeepLearningAI(model_path)
    score, score_dict = get_model_score_by_mcts(dl_model, game_state)
    now_data = strftime("%Y-%m-%d", localtime())
    test_dict = {
        now_data: {"score": score, "score_dict": score_dict}
    }
    model_info_dict["tests"] = test_dict

    return model_info_dict


SCORE_FILES = {
    "MCTSAI": Path(__file__).resolve().parent / "model_score_mcts.json",
    "MinimaxAI": Path(__file__).resolve().parent / "model_score_minimax.json",
    "DeepLearningAI": Path(__file__).resolve().parent / "model_score_dl.json",
}
LEGACY_SCORE_FILE = Path(__file__).resolve().parent / "model_score.json"


def load_model_score(model_type):
    score_path = SCORE_FILES.get(model_type)
    if score_path is None:
        raise ValueError(f"Unknown model type: {model_type}")

    if score_path.exists():
        with open(score_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if LEGACY_SCORE_FILE.exists():
        with open(LEGACY_SCORE_FILE, "r", encoding="utf-8") as f:
            legacy_scores = json.load(f)
        for key, path in SCORE_FILES.items():
            if not path.exists():
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(legacy_scores.get(key, []), f, indent=2, ensure_ascii=False)
        return legacy_scores.get(model_type, [])

    return []


def save_model_score(model_type, model_score):
    score_path = SCORE_FILES.get(model_type)
    if score_path is None:
        raise ValueError(f"Unknown model type: {model_type}")
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump(model_score, f, indent=2, ensure_ascii=False)


def save_model_info_dict(info_dict, model_type):
    model_score = load_model_score(model_type)
    model_score.append(info_dict)
    save_model_score(model_type, model_score)
