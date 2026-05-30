import sys
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainblock_diverse.dqn_agent import train_dqn
from brainblock_diverse.ppo_agent import train_ppo
from brainblock_diverse.make_solution_images import main as make_solution_images
from brainblock_diverse.make_frequency_plots import main as make_frequency_plots


ALGO = "both"  # choose "dqn", "ppo", or "both"
EPISODES = 100_000
SEEDS = [42, 100, 123, 456, 789]
OUT_DIR = Path("results/brainblock_diverse")


def run_dqn_proc(episodes, seed, out_dir, conn):
    rows = train_dqn(episodes, seed, out_dir)
    conn.send(rows)
    conn.close()


def run_ppo_proc(episodes, seed, out_dir, conn):
    rows = train_ppo(episodes, seed, out_dir)
    conn.send(rows)
    conn.close()


def print_summary_statistics(out_dir, seeds, algos):
    import pandas as pd
    import numpy as np

    print("\n" + "="*50)
    print("DIVERSE EVALUATION PROTOCOL SUMMARY STATISTICS (Across 5 Seeds)")
    print("="*50)

    for algo in algos:
        success_rates = []
        returns_means = []
        lengths_means = []
        invalid_rates = []

        for seed in seeds:
            csv_path = out_dir / f"seed_{seed}" / algo / f"{algo}_diverse_metrics.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)

            success_rates.append(df["success"].mean())
            returns_means.append(df["total_reward"].mean())
            lengths_means.append(df["steps"].mean())
            
            invalid_count = (df["terminal_reason"] == "invalid").sum()
            invalid_rates.append(invalid_count / len(df))

        if not success_rates:
            continue

        print(f"\nAlgorithm: {algo.upper()}_DIVERSE")
        print(f"Success Rate:      {np.mean(success_rates)*100:.2f}% ± {np.std(success_rates)*100:.2f}%")
        print(f"Episodic Return:   {np.mean(returns_means):.3f} ± {np.std(returns_means):.3f}")
        print(f"Episode Length:    {np.mean(lengths_means):.2f} ± {np.std(lengths_means):.2f}")
        print(f"Invalid-action:    {np.mean(invalid_rates)*100:.2f}% ± {np.std(invalid_rates)*100:.2f}%")
    print("="*50 + "\n")


def main(seeds=None):
    if seeds is None:
        seeds = SEEDS

    if ALGO == "both":
        for seed in seeds:
            print(f"Starting DQN and PPO in parallel (spawn mode) for seed {seed} with {EPISODES} episodes...")
            seed_out_dir = OUT_DIR / f"seed_{seed}"
            
            dqn_parent_conn, dqn_child_conn = multiprocessing.Pipe()
            ppo_parent_conn, ppo_child_conn = multiprocessing.Pipe()
            
            dqn_process = multiprocessing.Process(
                target=run_dqn_proc, 
                args=(EPISODES, seed, seed_out_dir / "dqn", dqn_child_conn)
            )
            ppo_process = multiprocessing.Process(
                target=run_ppo_proc, 
                args=(EPISODES, seed, seed_out_dir / "ppo", ppo_child_conn)
            )
            
            dqn_process.start()
            ppo_process.start()
            
            dqn_parent_conn.recv()
            ppo_parent_conn.recv()
            
            dqn_process.join()
            ppo_process.join()
            
        from brainblock_diverse.plotting import save_aggregated_plots
        save_aggregated_plots(OUT_DIR, seeds)
        print_summary_statistics(OUT_DIR, seeds, ["dqn", "ppo"])
        
    else:
        for seed in seeds:
            print(f"Running {ALGO} sequentially for seed {seed} with {EPISODES} episodes...")
            seed_out_dir = OUT_DIR / f"seed_{seed}"
            if ALGO == "dqn":
                train_dqn(EPISODES, seed, seed_out_dir / "dqn")
            elif ALGO == "ppo":
                train_ppo(EPISODES, seed, seed_out_dir / "ppo")
        
        from brainblock_diverse.plotting import save_aggregated_plots
        save_aggregated_plots(OUT_DIR, seeds)
        print_summary_statistics(OUT_DIR, seeds, [ALGO])

    make_solution_images(seeds)
    make_frequency_plots(seeds)



if __name__ == "__main__":
    main()
