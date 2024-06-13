from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .ai_algorithm import AIAlgorithm, logger
from game.chess_game import ChessGame


class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        # 第一个卷积层，输入通道数为2（棋盘的当前值和总负载值），输出通道数为32，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        # 第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为3x3，填充为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 第一个全连接层，输入大小为64 * 5 * 5，输出大小为128
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # 第二个全连接层，输入大小为128，输出大小为25（棋盘的5x5个可能的移动位置）
        self.fc2 = nn.Linear(128, 25)

    def forward(self, x):
        # 通过第一个卷积层，然后进行ReLU激活
        x = F.relu(self.conv1(x))
        # 通过第二个卷积层，然后进行ReLU激活
        x = F.relu(self.conv2(x))
        # 将特征图展平为一维向量
        x = x.reshape(-1, 64 * 5 * 5)
        # 通过第一个全连接层，然后进行ReLU激活
        x = F.relu(self.fc1(x))
        # 通过第二个全连接层，得到输出
        x = self.fc2(x)
        return x
    

class DeepLearningAI(AIAlgorithm):
    def __init__(self, model_path):
        self.model = ChessModel().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def find_best_move(self, game: ChessGame):
        board_state = np.array(game.chessboard).reshape(1, 5, 5, 2)
        board_state = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        with torch.no_grad():
            outputs = self.model(board_state)
            print(outputs)
            move_index = torch.argmax(outputs).item()
            move = (move_index // 5, move_index % 5)
        return move