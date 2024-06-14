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
        # 第一个卷积层，输入通道数为3（棋盘的当前值和总负载值），输出通道数为32，卷积核大小为3x3，填充为1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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

    def process_output(self, output, board_state):
        """
        将已经有落子的点在输出中屏蔽, 并输出最大数值所在的行列。
        :param output: 模型输出，形状为 (batch_size, 25)
        :param board_state: 当前棋盘状态，形状为 (batch_size, 3, 5, 5)
        :return: (row, col)
        """
        batch_size = output.size(0)
        output = output.view(batch_size, 5, 5)
        
        # 找到已经有落子的点
        occupied = torch.any(board_state[:, :1, :, :] != 0, dim=1)  # (batch_size, 5, 5)
        # print(occupied)
        
        # 将这些点的输出设为极小值
        output[occupied] = -float('inf')
        print(output)
        
        return output.view(batch_size, -1)

    def calculate_win_probability(self, outputs):
        """
        计算当前局面的获胜概率
        :param outputs: 模型输出，形状为 (batch_size, 25)
        :return: 获胜概率，标量
        """
        # 将 -inf 替换为一个非常小的有限值
        outputs[outputs == -float('inf')] = -1e10

        # 使用Softmax将得分转换为概率分布
        probabilities = F.softmax(outputs, dim=1)
        
        # 取概率的最大值作为获胜概率
        win_prob = torch.max(probabilities)[0].item()
        
        return win_prob
    
    def process_board(self, game: ChessGame):
        color = game.get_color()
        
        processed_board = []
        for row in game.chessboard:
            processed_row = []
            for cell in row:
                processed_cell = cell + [color]
                if processed_cell[0] == float("inf"):
                    processed_cell[0] = 5
                processed_row.append(processed_cell)
            processed_board.append(processed_row)
        return processed_board

    def find_best_move(self, game: ChessGame):
        chessboard = self.process_board(game)

        board_state = np.array(chessboard).reshape(1, 5, 5, 3)
        board_state = torch.tensor(board_state, dtype=torch.float32).permute(0, 3, 1, 2).cuda()
        with torch.no_grad():
            outputs = self.model(board_state)
            masked_outputs = self.process_output(outputs, board_state)

            move_index = torch.argmax(masked_outputs).item()
            move = (move_index // 5, move_index % 5)

        win_rate = self.calculate_win_probability(masked_outputs)
        game.set_current_win_rate(win_rate)
        return move