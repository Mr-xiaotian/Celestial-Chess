import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from cc_game.chess_game_env import ChessGameEnv


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


class REINFORCE(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = (
            64  # The number of neurons in hidden layers of the neural network
        )
        self.lr = 4e-4  # learning rate
        self.GAMMA = 0.99  # discount factor
        self.episode_s, self.episode_a, self.episode_r = [], [], []

        self.policy = Policy(state_dim, action_dim, self.hidden_width)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = (
            self.policy(s).detach().numpy().flatten()
        )  # probability distribution(numpy)
        if deterministic:  # We use the deterministic policy during the evaluating
            a = np.argmax(
                prob_weights
            )  # Select the action with the highest probability
            return a
        else:  # We use the stochastic policy during the training
            a = np.random.choice(
                range(self.action_dim), p=prob_weights
            )  # Sample the action according to the probability distribution
            return a

    def store(self, s, a, r):
        self.episode_s.append(s)
        self.episode_a.append(a)
        self.episode_r.append(r)

    def learn(
        self,
    ):
        G = []
        g = 0
        for r in reversed(self.episode_r):  # calculate the return G reversely
            g = self.GAMMA * g + r
            G.insert(0, g)

        for t in range(len(self.episode_r)):
            s = torch.unsqueeze(torch.tensor(self.episode_s[t], dtype=torch.float), 0)
            a = self.episode_a[t]
            g = G[t]

            a_prob = self.policy(s).flatten()
            policy_loss = -pow(self.GAMMA, t) * g * torch.log(a_prob[a])
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Clean the buffer
        self.episode_s, self.episode_a, self.episode_r = [], [], []


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(
                s, deterministic=True
            )  # We use the deterministic policy during the evaluating
            s_, r, done, truncated, _ = env.step(a)
            done = done or truncated
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == "__main__":
    env = ChessGameEnv()
    env_evaluate = ChessGameEnv()  # 评估时需要重新构建环境
    number = 1
    seed = 500

    # 设置随机种子
    env.reset(seed=seed)
    env_evaluate.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # 每个episode的最大步数
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = REINFORCE(state_dim, action_dim)
    writer = SummaryWriter(
        log_dir="tests/runs/REINFORCE/REINFORCE_env_{}_number_{}_seed_{}".format(
            "ChessGameEnv", number, seed
        )
    )  # 构建tensorboard

    max_train_steps = 1e5  # 最大训练步数
    evaluate_freq = 1e3  # 每隔evaluate_freq步评估一次策略
    evaluate_num = 0  # 记录评估次数
    evaluate_rewards = []  # 记录评估奖励
    total_steps = 0  # 记录训练的总步数

    frames = []

    while total_steps < max_train_steps:
        episode_steps = 0
        s, _ = env.reset()
        done = False
        while not done:
            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, truncated, _ = env.step(a)
            done = done or truncated
            agent.store(s, a, r)
            s = s_

            # 每隔evaluate_freq步评估一次策略
            if (total_steps + 1) % evaluate_freq == 0:
                env_evaluate.render()
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)
                print(
                    f"evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward} \t"
                )
                writer.add_scalar(
                    f'step_rewards_{"ChessGameEnv"}',
                    evaluate_reward,
                    global_step=total_steps,
                )
                if evaluate_num % 10 == 0:
                    np.save(
                        f'tests/data_train/REINFORCE_env_{"ChessGameEnv"}_number_{number}_seed_{seed}.npy',
                        np.array(evaluate_rewards),
                    )

            total_steps += 1

        # 一个episode结束后进行更新
        agent.learn()
