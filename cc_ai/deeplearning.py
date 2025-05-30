from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cc_game.chess_game import ChessGame
from .ai_algorithm import AIAlgorithm, logger


class ChessPolicyModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(ChessPolicyModel, self).__init__()
        # 卷积层，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(3, 30, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 60, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(60)
        self.conv3 = nn.Conv2d(60, 120, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(120)
        self.conv4 = nn.Conv2d(120, 240, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(240)

        # 全连接层，输出大小为25（棋盘的5x5个可能的移动位置）
        self.fc1 = nn.Linear(240 * 5 * 5, 512)
        # self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 25)

    def forward(self, x):
        # 通过卷积层，然后进行ReLU激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # 将特征图展平为一维向量
        x = x.reshape(-1, 240 * 5 * 5)
        # 通过第一个全连接层，然后进行ReLU激活
        x = F.relu(self.fc1(x))
        # 加入Dropout层
        # x = self.dropout(x)
        # 通过第二个全连接层，得到输出
        x = self.fc2(x)
        # # 将分数转换为概率分布
        # x = F.softmax(x, dim=1)
        return x
    

class DeepLearningAI(AIAlgorithm):
    def __init__(self, model_path, complete_mode=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessPolicyModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.complete_mode = complete_mode

    def process_output(self, output, board_state):
        """
        将已经有落子的点在输出中屏蔽
        :param output: 模型输出，形状为 (batch_size, 25)
        :param board_state: 当前棋盘状态，形状为 (batch_size, 3, 5, 5)
        :return: (row, col)
        """
        # print(output.view(5, 5))
        batch_size = output.size(0)
        output = output.view(batch_size, 5, 5)
        
        # 找到已经有落子的点
        occupied = torch.any(board_state[:, :1, :, :] != 0, dim=1)  # (batch_size, 5, 5)
        # print(occupied)
        
        # 将这些点的输出设为极小值
        output[occupied] = -float('inf')
        
        return output.view(batch_size, -1)
    
    def process_board(self, game: ChessGame):
        """
        将棋盘状态转换为模型输入的格式
        :param game: 当前游戏状态
        :return: 模型输入，形状为 (batch_size, 3, 5, 5)
        """
        color = game.get_color()
        color_channel = np.full((5, 5, 1), color)
        processed_board = np.concatenate((game.chessboard, color_channel), axis=2)

        for row in processed_board:
            for cell in row:
                if cell[0] == float("inf"):
                    cell[0] = 5
        return processed_board
    
    def trans_softmax(self, outputs):
        """
        将输出转换为概率分布
        :param outputs: 模型输出，形状为 (batch_size, 25)
        :return: 归一化概率分布，形状为 (batch_size, 25)
        """
        # 将负无穷大部分屏蔽掉
        mask = outputs != float('-inf')
        
        # 对屏蔽后的部分进行softmax计算
        masked_tensor = outputs.clone()
        masked_tensor[~mask] = float('-inf')  # 使用负无穷大来忽略这些值
        softmax_tensor = F.softmax(masked_tensor, dim=-1)
        
        # 保持负无穷大部分不变
        softmax_tensor[~mask] = float('-inf')
        print(softmax_tensor.reshape(5,5))

        return softmax_tensor

    def get_move_probs(self, game: ChessGame):
        chessboard = self.process_board(game)

        board_state = np.array(chessboard).reshape(1, 5, 5, 3)
        board_state = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            outputs = self.model(board_state)
            masked_outputs = self.process_output(outputs, board_state)

        return masked_outputs.view(5, 5)
    
    def find_best_move(self, game: ChessGame):
        chessboard = self.process_board(game)

        board_state = np.array(chessboard).reshape(1, 5, 5, 3)
        board_state = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            outputs = self.model(board_state)
            masked_outputs = self.process_output(outputs, board_state)

            move_index = torch.argmax(masked_outputs).item()
            move = (move_index // 5, move_index % 5)

        if self.complete_mode:
            self.trans_softmax(masked_outputs)
            game.set_current_win_rate()
        return move
    
    def end_game(self):
        pass

    def end_model(self):
        pass