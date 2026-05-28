import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path("runs/brainblock_diverse")
OUTPUT_PATH = RUN_DIR / "solution_frequencies.png"

def parse_frequencies(path):
    if not path.exists():
        return []
    
    text = path.read_text(encoding="utf-8")
    # Matches patterns like "Solution X | seen Y time(s)"
    matches = re.findall(r"Solution\s+(\d+)\s*\|\s*seen\s+(\d+)\s+time", text)
    
    frequencies = []
    for num, count in matches:
        frequencies.append((int(num), int(count)))
    
    # Sort by solution number
    frequencies.sort(key=lambda x: x[0])
    return [f[1] for f in frequencies]

def main():
    dqn_freqs = parse_frequencies(RUN_DIR / "dqn" / "discovered_solutions.txt")
    ppo_freqs = parse_frequencies(RUN_DIR / "ppo" / "discovered_solutions.txt")
    
    if not dqn_freqs and not ppo_freqs:
        print("No solution files found to plot.")
        return

    # Use a clean, scientific styling
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot DQN
    if dqn_freqs:
        x_dqn = np.arange(1, len(dqn_freqs) + 1)
        axes[0].bar(x_dqn, dqn_freqs, color="#3b82f6", edgecolor="#1e3a8a", alpha=0.85)
        axes[0].set_title("DQN Solution Frequencies (Log Scale)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Solution ID", fontsize=10)
        axes[0].set_ylabel("Times Solved", fontsize=10)
        axes[0].set_yscale("log")  # Using log scale due to massive variation (3 vs 22,993)
        axes[0].set_xticks(x_dqn)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)
        
        # Add labels on top of the bars
        for x, y in zip(x_dqn, dqn_freqs):
            axes[0].text(x, y * 1.1, f"{y}", ha="center", va="bottom", fontsize=8, rotation=30)
            
    # Plot PPO
    if ppo_freqs:
        x_ppo = np.arange(1, len(ppo_freqs) + 1)
        axes[1].bar(x_ppo, ppo_freqs, color="#10b981", edgecolor="#064e3b", alpha=0.85)
        axes[1].set_title("PPO Solution Frequencies", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Solution ID", fontsize=10)
        axes[1].set_ylabel("Times Solved", fontsize=10)
        axes[1].set_ylim(0, 2)
        axes[1].set_xticks(x_ppo)
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)
        
        for x, y in zip(x_ppo, ppo_freqs):
            axes[1].text(x, y + 0.05, f"{y}", ha="center", va="bottom", fontsize=9)
            
    plt.suptitle("Frequency of Discovered Solutions: DQN vs. PPO", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    # Save the figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    plt.close()
    
    print(f"Solution frequency comparison plot saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
