import csv

import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window=50):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start : i + 1]))
    return out


def save_results(rows, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{name}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rewards = [r["total_reward"] for r in rows]
    areas = [r["covered_area"] for r in rows]
    success = [r["success"] for r in rows]

    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(moving_average(rewards))
    plt.ylabel("reward")
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(moving_average(areas))
    plt.ylabel("covered")
    plt.grid(alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(moving_average(success))
    plt.ylabel("success")
    plt.xlabel("episode")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"{name}_plot.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()

    print("saved:", csv_path)
    print("saved:", fig_path)


def save_combined_plot(all_rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    for algo in sorted(set(r["algo"] for r in all_rows)):
        rows = [r for r in all_rows if r["algo"] == algo]
        rewards = [r["total_reward"] for r in rows]
        plt.plot(moving_average(rewards), label=algo)

    plt.xlabel("episode")
    plt.ylabel("moving average reward")
    plt.title("DQN vs PPO")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / "dqn_vs_ppo_reward.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print("saved:", fig_path)


def save_aggregated_plots(out_dir, seeds):
    import pandas as pd

    algos = ["dqn", "ppo"]
    plt.figure(figsize=(10, 6))

    for algo in algos:
        all_seed_rewards = []
        episodes = None

        for seed in seeds:
            csv_path = out_dir / f"seed_{seed}" / algo / f"{algo}_metrics.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            rewards = df["total_reward"].values
            mv_rewards = moving_average(rewards)
            all_seed_rewards.append(mv_rewards)
            if episodes is None:
                episodes = df["episode"].values

        if not all_seed_rewards:
            continue

        min_len = min(len(r) for r in all_seed_rewards)
        all_seed_rewards = [r[:min_len] for r in all_seed_rewards]
        episodes = episodes[:min_len]

        all_seed_rewards = np.array(all_seed_rewards)
        mean_rewards = np.mean(all_seed_rewards, axis=0)
        std_rewards = np.std(all_seed_rewards, axis=0)

        plt.plot(episodes, mean_rewards, label=algo)
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.15)

    plt.xlabel("episode")
    plt.ylabel("moving average reward")
    plt.title("DQN vs PPO (Aggregated over 5 Seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / "dqn_vs_ppo_reward_aggregated.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print("saved aggregated plot:", fig_path)

