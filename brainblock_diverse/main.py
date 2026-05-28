import sys
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainblock_diverse.dqn_agent import train_dqn
from brainblock_diverse.plotting import save_combined_plot
from brainblock_diverse.ppo_agent import train_ppo
from brainblock_diverse.make_solution_images import main as make_solution_images
from brainblock_diverse.make_frequency_plots import main as make_frequency_plots


ALGO = "both"  # choose "dqn", "ppo", or "both"
EPISODES = 100_000
SEED = 42
OUT_DIR = Path("runs/brainblock_diverse")


def run_dqn_proc(episodes, seed, out_dir, conn):
    rows = train_dqn(episodes, seed, out_dir)
    conn.send(rows)
    conn.close()


def run_ppo_proc(episodes, seed, out_dir, conn):
    rows = train_ppo(episodes, seed, out_dir)
    conn.send(rows)
    conn.close()


def main():
    if ALGO == "both":
        print(f"Starting DQN and PPO in parallel (spawn mode) with {EPISODES} episodes...")
        
        # Create Pipes for process communication
        dqn_parent_conn, dqn_child_conn = multiprocessing.Pipe()
        ppo_parent_conn, ppo_child_conn = multiprocessing.Pipe()
        
        # Create separate processes
        dqn_process = multiprocessing.Process(
            target=run_dqn_proc, 
            args=(EPISODES, SEED, OUT_DIR / "dqn", dqn_child_conn)
        )
        ppo_process = multiprocessing.Process(
            target=run_ppo_proc, 
            args=(EPISODES, SEED, OUT_DIR / "ppo", ppo_child_conn)
        )
        
        # Start processes
        dqn_process.start()
        ppo_process.start()
        
        # Retrieve results
        dqn_rows = dqn_parent_conn.recv()
        ppo_rows = ppo_parent_conn.recv()
        
        # Wait for both processes to finish
        dqn_process.join()
        ppo_process.join()
        
        # Combine results and plot
        all_rows = dqn_rows + ppo_rows
        save_combined_plot(all_rows, OUT_DIR)
        print("Parallel training complete! Comparison plot saved.")
        
    else:
        all_rows = []
        if ALGO == "dqn":
            print(f"Running DQN sequentially with {EPISODES} episodes...")
            rows = train_dqn(EPISODES, SEED, OUT_DIR / "dqn")
            all_rows += rows
        elif ALGO == "ppo":
            print(f"Running PPO sequentially with {EPISODES} episodes...")
            rows = train_ppo(EPISODES, SEED, OUT_DIR / "ppo")
            all_rows += rows
        else:
            print(f"Unknown ALGO value: {ALGO}")

    make_solution_images()
    make_frequency_plots()


if __name__ == "__main__":
    main()
