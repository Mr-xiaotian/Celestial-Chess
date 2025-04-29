import gymnasium as gym
from gymnasium import spaces
import numpy as np
from cc_game.chess_game import ChessGame

class ChessGameEnv(gym.Env):
    def __init__(self):
        super(ChessGameEnv, self).__init__()
        self.game = ChessGame()
        self.action_space = spaces.Discrete(self.game.board_range[0] * self.game.board_range[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.game.board_range[0], self.game.board_range[1], 2), 
                                            dtype=np.float32)
        self._max_episode_steps = 100  # 设置每个 episode 的最大步数，可以根据需要调整

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.game = ChessGame()
        return self.game.chessboard.flatten(), {}

    def step(self, action):
        row = action // self.game.board_range[1]
        col = action % self.game.board_range[1]
        
        if not self.game.is_move_valid(row, col):
            return self.game.chessboard.flatten(), -10, True, {}  # Invalid move results in a penalty
        
        self.game.update_chessboard(row, col, self.game.current_color)
        reward = self.game.get_score()
        done = self.game.is_game_over()
        
        return self.game.chessboard.flatten(), reward, done, {}

    def render(self, mode='human'):
        self.game.show_chessboard()


