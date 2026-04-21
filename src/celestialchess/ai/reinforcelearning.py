"""
强化学习 AI —— PPO (Proximal Policy Optimization) 风格的 Actor-Critic 网络

网络结构：
  输入: (batch, 4, rows, cols) —— 与 DeepLearningAI 相同的 4 通道棋盘表示
  Actor (policy head): 输出 logits，形状 (batch, rows*cols)
  Critic (value head): 输出 state value，形状 (batch, 1)

训练方式：自我对弈 (self-play) + PPO 更新
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..chess_game import ChessGame
from ..tools.dl_tool import process_board
from .base_ai import BaseAI, logger


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------

class RLActorCriticModel(nn.Module):
    """共享卷积特征提取器 + Actor head + Critic head"""

    def __init__(self, rows: int, cols: int, in_channels: int = 4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        flat = rows * cols

        # 共享卷积主干
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 全连接共享层
        self.shared_fc = nn.Sequential(
            nn.Linear(128 * rows * cols, 512),
            nn.ReLU(),
        )

        # Actor head：输出每个位置的 logit
        self.actor_head = nn.Linear(512, flat)

        # Critic head：输出棋局价值估计
        self.critic_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 4, rows, cols)
        returns:
            logits: (B, rows*cols)
            value:  (B, 1)
        """
        feat = self.backbone(x)
        feat = feat.reshape(feat.size(0), -1)
        shared = self.shared_fc(feat)
        logits = self.actor_head(shared)
        value = self.critic_head(shared)
        return logits, value

    def get_action_and_value(
        self, x: torch.Tensor, valid_mask: torch.Tensor, action: torch.Tensor | None = None
    ):
        """
        推断动作、log_prob、熵、状态价值。

        valid_mask: (B, rows*cols)  True = 可落子
        """
        logits, value = self.forward(x)

        # 将非法格屏蔽
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=masked_logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


# ---------------------------------------------------------------------------
# 辅助：棋盘 → tensor
# ---------------------------------------------------------------------------

def _board_to_tensor(game: ChessGame, device: torch.device) -> torch.Tensor:
    """将 ChessGame 当前状态转换成 (1, 4, rows, cols) tensor"""
    processed = process_board(game.chessboard, game.get_color())   # (rows, cols, 4)
    arr = np.array(processed, dtype=np.float32)                    # (rows, cols, 4)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)        # (1, 4, rows, cols)
    return t.to(device)


def _valid_mask(game: ChessGame, device: torch.device) -> torch.Tensor:
    """返回合法落子掩码 (1, rows*cols)，合法=True"""
    rows, cols = game.board_range
    flat = rows * cols
    mask = torch.zeros(flat, dtype=torch.bool)
    moves = game.get_all_moves()
    for r, c in moves:
        mask[r * cols + c] = True
    return mask.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# 动作选取（带温度采样 / deterministic）
# ---------------------------------------------------------------------------

def _pick_action(game: ChessGame, model: RLActorCriticModel, device: torch.device, greedy: bool = True):
    rows, cols = game.board_range
    state = _board_to_tensor(game, device)
    vmask = _valid_mask(game, device)

    with torch.no_grad():
        action, log_prob, _, value = model.get_action_and_value(state, vmask)

    idx = action.item()
    move = (idx // cols, idx % cols)
    return move, log_prob.item(), value.item()


# ---------------------------------------------------------------------------
# PPO 训练循环（自我对弈）
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    PPO 训练器：
    - 使用 self-play 生成 rollout
    - 用 PPO-clip 更新策略
    """

    def __init__(
        self,
        model: RLActorCriticModel,
        device: torch.device,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 64,
    ):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- rollout 收集 --------

    def collect_rollout(self, game_state: tuple, n_games: int = 10):
        """
        自我对弈 n_games 局，收集 (state, action, log_prob, reward, done, value, valid_mask) 元组。

        game_state: 形如 ((rows, cols), power)
        """
        rows_cols, power = game_state
        rows, cols = rows_cols

        states, actions, log_probs, rewards, dones, values, vmasks = [], [], [], [], [], [], []

        for _ in range(n_games):
            game = ChessGame(rows_cols, power)
            ep_states, ep_actions, ep_log_probs, ep_rewards, ep_dones, ep_values, ep_vmasks = (
                [], [], [], [], [], [], []
            )

            while not game.is_game_over():
                state_t = _board_to_tensor(game, self.device)
                vmask_t = _valid_mask(game, self.device)

                with torch.no_grad():
                    action_t, lp_t, _, val_t = self.model.get_action_and_value(state_t, vmask_t)

                ep_states.append(state_t)
                ep_vmasks.append(vmask_t)
                ep_actions.append(action_t)
                ep_log_probs.append(lp_t)
                ep_values.append(val_t)

                idx = action_t.item()
                move = (idx // cols, idx % cols)
                color = game.get_color()
                game.update_chessboard(*move, color)

                done = game.is_game_over()
                # 奖励：当前棋盘分（相对于落子前的变化）
                score = float(game.get_raw_score())
                ep_rewards.append(torch.tensor([score], dtype=torch.float32, device=self.device))
                ep_dones.append(torch.tensor([float(done)], dtype=torch.float32, device=self.device))

            states.extend(ep_states)
            actions.extend(ep_actions)
            log_probs.extend(ep_log_probs)
            rewards.extend(ep_rewards)
            dones.extend(ep_dones)
            values.extend(ep_values)
            vmasks.extend(ep_vmasks)

        # 堆叠，并 detach —— rollout 只保留数据，不保留计算图
        states_t    = torch.cat(states, dim=0).detach()        # (T, 4, rows, cols)
        actions_t   = torch.cat(actions, dim=0).detach()       # (T,)
        log_probs_t = torch.cat(log_probs, dim=0).detach()     # (T,)
        rewards_t   = torch.cat(rewards, dim=0).detach()       # (T,)
        dones_t     = torch.cat(dones, dim=0).detach()         # (T,)
        values_t    = torch.cat(values, dim=0).detach()        # (T,)
        vmasks_t    = torch.cat(vmasks, dim=0).detach()        # (T, rows*cols)

        # --- GAE advantages ---
        advantages = self._compute_gae(rewards_t, dones_t, values_t)
        returns = advantages + values_t

        return states_t, actions_t, log_probs_t, advantages, returns, vmasks_t

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1].item() if t + 1 < T else 0.0
            delta = rewards[t].item() + self.gamma * next_val * (1 - dones[t].item()) - values[t].item()
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].item()) * gae
            advantages[t] = gae
        return advantages

    # -------- PPO 更新 --------

    def update(self, states_t, actions_t, log_probs_t, advantages, returns, vmasks_t):
        T = states_t.size(0)
        idxs = list(range(T))

        total_loss = 0.0
        for _ in range(self.ppo_epochs):
            random.shuffle(idxs)
            for start in range(0, T, self.mini_batch_size):
                mb = idxs[start: start + self.mini_batch_size]
                mb_states  = states_t[mb]
                mb_actions = actions_t[mb]
                mb_old_lp  = log_probs_t[mb]
                mb_adv     = advantages[mb]
                mb_ret     = returns[mb]
                mb_vmask   = vmasks_t[mb]

                _, new_lp, entropy, new_val = self.model.get_action_and_value(
                    mb_states, mb_vmask, mb_actions
                )

                # ratio = exp(new_log_prob - old_log_prob)
                ratio = torch.exp(new_lp - mb_old_lp)

                # normalize advantage
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss (PPO-clip)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                v_loss = F.mse_loss(new_val, mb_ret)

                # entropy bonus
                ent_loss = -entropy.mean()

                loss = pg_loss + self.value_loss_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss


# ---------------------------------------------------------------------------
# RL Agent（对外接口，继承 BaseAI）
# ---------------------------------------------------------------------------

class RLAI(BaseAI):
    """
    基于 PPO 的强化学习 AI。

    使用:
        agent = RLAI((5, 5), power=2)
        # 先训练（可选）
        agent.train(n_iterations=50, games_per_iter=20)
        # 保存/加载
        agent.save("rl_model.pth")
        agent.load("rl_model.pth")
        # 推断
        move = agent.find_best_move(game)
    """

    name = "rl"

    def __init__(
        self,
        board_range: tuple = (5, 5),
        power: int = 2,
        model_path: str | None = None,
    ):
        self.board_range = board_range
        rows, cols = board_range
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RLActorCriticModel(rows, cols).to(self.device)
        self._msg = "强化学习网络正在感知棋局……"

        if model_path:
            self.load(model_path)

    # -------- BaseAI 接口 --------

    def find_best_move(self, game: ChessGame):
        self.model.eval()
        move, _, value = _pick_action(game, self.model, self.device, greedy=True)
        self._msg = self._build_msg(value)
        return move

    @property
    def msg(self):
        return self._msg

    def end_game(self):
        pass

    def end_model(self):
        pass

    # -------- 训练 --------

    def train(
        self,
        game_state: tuple | None = None,
        n_iterations: int = 100,
        games_per_iter: int = 10,
        lr: float = 3e-4,
        ppo_epochs: int = 4,
        save_path: str | None = None,
        log_interval: int = 10,
    ):
        """
        PPO 自我对弈训练。

        game_state: 形如 ((rows, cols), power)，默认使用构造时的参数
        n_iterations: 训练迭代轮数（每轮收集 games_per_iter 局）
        save_path: 每 log_interval 轮保存模型
        """
        if game_state is None:
            game_state = (self.board_range, 2)

        trainer = PPOTrainer(
            self.model, self.device,
            lr=lr, ppo_epochs=ppo_epochs
        )

        logger.info(f"[RLAI] 开始训练: {n_iterations} 轮, 每轮 {games_per_iter} 局")

        for it in range(1, n_iterations + 1):
            self.model.train()
            rollout = trainer.collect_rollout(game_state, n_games=games_per_iter)
            loss = trainer.update(*rollout)

            if it % log_interval == 0:
                logger.info(f"[RLAI] iter {it}/{n_iterations}  loss={loss:.4f}")
                print(f"[RLAI] iter {it}/{n_iterations}  loss={loss:.4f}")

                if save_path:
                    self.save(save_path)

        if save_path:
            self.save(save_path)
            logger.info(f"[RLAI] 训练完成，模型已保存至 {save_path}")

    # -------- 保存 / 加载 --------

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info(f"[RLAI] 模型已加载: {path}")

    # -------- 人格消息 --------

    def _build_msg(self, value: float) -> str:
        sentiments = [
            "策略网络已收敛，奖励函数低声歌唱。",
            "价值函数估计此局形势……略显混沌。",
            "梯度指引着手，优势函数微微倾斜。",
            "探索与利用之间，我选择了此格。",
            "累积折扣回报在这一步达到局部极大。",
        ]
        hint = np.random.choice(sentiments)
        return f"当前局面价值估计: {value:.3f}\n{hint}"
