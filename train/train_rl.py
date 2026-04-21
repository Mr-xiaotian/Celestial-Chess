"""
RL Agent 独立训练脚本
用法（在项目根目录执行）：
    python train/train_rl.py                        # 用默认参数训练 5x5 棋盘
    python train/train_rl.py --rows 5 --cols 5 --power 2 --iters 200 --games 20
"""
import argparse
import sys
from pathlib import Path

# 让脚本可以从项目根目录执行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from celestialchess.ai.reinforcelearning import RLAI


def parse_args():
    p = argparse.ArgumentParser(description="Train RLAI via PPO self-play")
    p.add_argument("--rows",   type=int, default=5,   help="Board rows (default: 5)")
    p.add_argument("--cols",   type=int, default=5,   help="Board cols (default: 5)")
    p.add_argument("--power",  type=int, default=2,   help="Game power parameter (default: 2)")
    p.add_argument("--iters",  type=int, default=200, help="Training iterations (default: 200)")
    p.add_argument("--games",  type=int, default=10,  help="Games per iteration (default: 10)")
    p.add_argument("--lr",     type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    p.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update (default: 4)")
    p.add_argument("--log-interval", type=int, default=10, help="Log every N iters (default: 10)")
    p.add_argument(
        "--save", type=str,
        default="data/models/rl_agent.pth",
        help="Model save path (default: data/models/rl_agent.pth)"
    )
    p.add_argument(
        "--load", type=str, default=None,
        help="Load existing model before training (optional)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    board_range = (args.rows, args.cols)
    game_state  = (board_range, args.power)

    # 确保保存目录存在
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[train_rl] board: {board_range}, power={args.power}")
    print(f"[train_rl] training {args.iters} iters, {args.games} games/iter")
    print(f"[train_rl] save to: {args.save}")

    agent = RLAI(board_range=board_range, power=args.power)

    if args.load:
        print(f"[train_rl] loading existing model: {args.load}")
        agent.load(args.load)

    agent.train(
        game_state=game_state,
        n_iterations=args.iters,
        games_per_iter=args.games,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        save_path=str(save_path),
        log_interval=args.log_interval,
    )

    print(f"[train_rl] Training done! Model saved to {save_path}")


if __name__ == "__main__":
    main()
