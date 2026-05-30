import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

RUN_DIR = Path("results/brainblock_diverse")
OUTPUT_PATH = RUN_DIR / "solution_frequencies_aggregated.png"
SEEDS = [42, 100, 123, 456, 789]


def parse_solutions(path):
    if not path.exists() or path.stat().st_size == 0:
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    solutions = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Solution"):
            title = line
            board = []
            i += 1

            while i < len(lines) and lines[i].strip():
                board.append(lines[i].split())
                i += 1

            if board:
                solutions.append((title, board))
        i += 1

    return solutions


def aggregate_frequencies(seeds, algo):
    combined_counts = Counter()
    for seed in seeds:
        path = RUN_DIR / f"seed_{seed}" / algo / "discovered_solutions.txt"
        if not path.exists():
            continue

        solutions = parse_solutions(path)
        for title, board in solutions:
            match = re.search(r"seen\s+(\d+)\s+time", title)
            if not match:
                continue
            count = int(match.group(1))
            board_str = "\n".join(" ".join(row) for row in board)
            combined_counts[board_str] += count

    return [c for _, c in combined_counts.most_common()]


def main(seeds=None):
    if seeds is None:
        seeds = SEEDS
    dqn_freqs = aggregate_frequencies(seeds, "dqn")
    ppo_freqs = aggregate_frequencies(seeds, "ppo")

    if not dqn_freqs and not ppo_freqs:
        print("No solution files found to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if dqn_freqs:
        x_dqn = np.arange(1, len(dqn_freqs) + 1)
        axes[0].bar(x_dqn, dqn_freqs, color="#3b82f6", edgecolor="#1e3a8a", alpha=0.85)
        axes[0].set_title("DQN Discovered Solution Frequencies", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Unique Solution Index", fontsize=10)
        axes[0].set_ylabel("Times Solved (Log Scale)", fontsize=10)
        axes[0].set_yscale("log")
        axes[0].set_xticks(x_dqn)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        for x, y in zip(x_dqn, dqn_freqs):
            axes[0].text(x, y * 1.1, f"{y}", ha="center", va="bottom", fontsize=8, rotation=30)

    if ppo_freqs:
        x_ppo = np.arange(1, len(ppo_freqs) + 1)
        axes[1].bar(x_ppo, ppo_freqs, color="#10b981", edgecolor="#064e3b", alpha=0.85)
        axes[1].set_title("PPO Discovered Solution Frequencies", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Unique Solution Index", fontsize=10)
        axes[1].set_ylabel("Times Solved", fontsize=10)
        axes[1].set_ylim(0, max(ppo_freqs) + 1)
        axes[1].set_xticks(x_ppo)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        for x, y in zip(x_ppo, ppo_freqs):
            axes[1].text(x, y + 0.05, f"{y}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Frequency of Discovered Solutions: DQN vs. PPO (Aggregated over 5 Seeds)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Aggregated solution frequency comparison plot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
