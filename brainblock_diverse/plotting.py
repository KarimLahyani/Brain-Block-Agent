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
    unique = [r["unique_solutions"] for r in rows]

    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(moving_average(rewards))
    plt.ylabel("reward")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 2)
    plt.plot(moving_average(areas))
    plt.ylabel("covered")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 3)
    plt.plot(moving_average(success))
    plt.ylabel("success")
    plt.grid(alpha=0.3)

    plt.subplot(4, 1, 4)
    plt.plot(unique)
    plt.ylabel("unique")
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

    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    for algo in sorted(set(r["algo"] for r in all_rows)):
        rows = [r for r in all_rows if r["algo"] == algo]
        rewards = [r["total_reward"] for r in rows]
        plt.plot(moving_average(rewards), label=algo)
    plt.ylabel("moving avg reward")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    for algo in sorted(set(r["algo"] for r in all_rows)):
        rows = [r for r in all_rows if r["algo"] == algo]
        unique = [r["unique_solutions"] for r in rows]
        plt.plot(unique, label=algo)
    plt.xlabel("episode")
    plt.ylabel("unique solutions")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    fig_path = out_dir / "dqn_vs_ppo_diversity.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print("saved:", fig_path)
