import sys
import multiprocessing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainblock_standard.dqn_agent import train_dqn
from brainblock_standard.plotting import save_combined_plot
from brainblock_standard.ppo_agent import train_ppo


ALGO = "both"  # choose "dqn", "ppo", or "both"
EPISODES = 20_000
SEED = 42
OUT_DIR = Path("runs/brainblock_standard")


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


if __name__ == "__main__":
    main()
