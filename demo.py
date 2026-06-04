import os
import sys
import time
from pathlib import Path
import torch
import numpy as np

# Ensure path resolution
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Color codes for high-quality terminal graphics
COLORS = {
    "I": "\033[1;36m",   # Bold Cyan
    "O": "\033[1;33m",   # Bold Yellow
    "L": "\033[1;34m",   # Bold Blue
    "Z": "\033[1;32m",   # Bold Green
    "T": "\033[1;35m",   # Bold Magenta
    ".": "\033[90m",     # Dark Gray
    "RESET": "\033[0m"
}

def print_colored_board(board, id_to_piece):
    print("┌" + "──" * 8 + "─┐")
    for row in board:
        row_str = []
        for cell in row:
            char = id_to_piece.get(cell, ".")
            color = COLORS.get(char, COLORS["."])
            if char == ".":
                row_str.append(f"{color}·{COLORS['RESET']}")
            else:
                row_str.append(f"{color}{char}{COLORS['RESET']}")
        print("│ " + " ".join(row_str) + " │")
    print("└" + "──" * 8 + "─┘")

def clear_screen():
    # Attempt to clear terminal screen and home cursor
    if os.name == 'nt':
        os.system('cls')
    else:
        print("\033[H\033[J", end="")

def choose_action(algo, model, state, action_mask, temperature=1.0):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        if algo == "ppo":
            logits, _ = model(state_t)
            mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
            logits[~mask_t] = -1e9
            if temperature != 1.0:
                logits = logits / temperature
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return int(action.item())
        else:
            q_values = model(state_t).squeeze(0)
            mask = torch.full_like(q_values, -1e9)
            mask[action_mask] = 0
            q_values = q_values + mask
            return int(torch.argmax(q_values).item())

def run_demo(mode, algo, seed, delay=0.6, temperature=1.2):
    # Set up imports depending on mode
    if mode == "standard":
        from brainblock_standard.env import BrainBlockGymEnv
        from brainblock_standard.pieces import ID_TO_PIECE
        from brainblock_standard.ppo_agent import ActorCritic as PPOModel
        from brainblock_standard.dqn_agent import DQN as DQNModel
        
        env = BrainBlockGymEnv()
        model_class = PPOModel if algo == "ppo" else DQNModel
        model_name = "ppo_model.pt" if algo == "ppo" else "dqn_model.pt"
        model_path = ROOT / "training_results" / "brainblock_standard" / f"seed_{seed}" / algo / model_name
    else:
        from brainblock_diverse.env import BrainBlockDiverseEnv
        from brainblock_diverse.pieces import ID_TO_PIECE
        from brainblock_diverse.ppo_agent import ActorCritic as PPOModel
        from brainblock_diverse.dqn_agent import DQN as DQNModel
        from brainblock_diverse.diversity import SolutionMemory
        
        env = BrainBlockDiverseEnv(solution_memory=SolutionMemory())
        model_class = PPOModel if algo == "ppo" else DQNModel
        model_name = "ppo_diverse_model.pt" if algo == "ppo" else "dqn_diverse_model.pt"
        model_path = ROOT / "training_results" / "brainblock_diverse" / f"seed_{seed}" / algo / model_name

    # Check path
    if not model_path.exists():
        print(f"\nError: Pretrained weights not found at: {model_path}")
        print("Please check that your model is trained and saved there.")
        return False

    # Load model weights
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Reset environment
    state, info = env.reset(seed=seed)
    done = False
    step = 0
    total_reward = 0

    clear_screen()
    print("=" * 50)
    print(f"  LIVE DEMO: {mode.upper()} MODE - {algo.upper()} AGENT (Seed {seed})")
    print("=" * 50)
    print("\nInitial Empty Board:")
    print_colored_board(info["board"], ID_TO_PIECE)
    print(f"Queue remaining: {list(env.queue)}")
    print(f"Next piece to place: {env.current_piece}")
    time.sleep(delay * 1.5)

    while not done:
        current_p = env.current_piece
        action = choose_action(algo, model, state, info["action_mask"], temperature=temperature)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        total_reward += reward
        
        clear_screen()
        print("=" * 50)
        print(f"  LIVE DEMO: {mode.upper()} MODE - {algo.upper()} AGENT (Seed {seed})")
        print("=" * 50)
        print(f"\nStep {step}/10 | Placed piece: {current_p}")
        print_colored_board(info["board"], ID_TO_PIECE)
        print(f"Step Reward: {reward:.3f} | Total Reward: {total_reward:.3f}")
        
        if not done:
            print(f"Queue remaining: {list(env.queue)}")
            print(f"Next piece to place: {env.current_piece}")
        
        time.sleep(delay)

    print("\n" + "=" * 50)
    if info.get("terminal_reason") == "solved":
        print("\033[1;32mSUCCESS: The agent solved the puzzle perfectly! \033[0m")
    else:
        print(f"\033[1;31mFAILED: Episode ended. Reason: {info.get('terminal_reason')} \033[0m")
    print("=" * 50)
    return True

def main():
    # Provide simple terminal prompt
    print("=" * 50)
    print("      BRAINBLOCK RL AGENT VISUAL DEMO SUITE      ")
    print("=" * 50)
    
    try:
        mode_input = input("Select mode (1: diverse [default], 2: standard): ").strip()
        mode = "standard" if mode_input == "2" else "diverse"
        
        algo_input = input("Select algorithm (1: ppo [default], 2: dqn): ").strip()
        algo = "dqn" if algo_input == "2" else "ppo"
        
        seed_input = input("Select seed (0, 1, 2, 3, 4, 5, 42 [default]): ").strip()
        if not seed_input:
            seed = 42
        else:
            seed = int(seed_input)
            
        delay_input = input("Step delay in seconds (default 0.6): ").strip()
        delay = float(delay_input) if delay_input else 0.6
        
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        sys.exit(0)
    except ValueError:
        print("Invalid input. Using defaults.")
        mode = "diverse"
        algo = "ppo"
        seed = 0
        delay = 0.6

    print(f"\nLaunching {mode} {algo.upper()} agent using seed {seed}...")
    time.sleep(1)
    run_demo(mode, algo, seed, delay)

if __name__ == "__main__":
    main()
