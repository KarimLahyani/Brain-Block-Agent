import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from brainblock_standard.dqn_agent import DQN, select_dqn_action
from brainblock_standard.env import BrainBlockGymEnv
from brainblock_standard.pieces import print_board
from brainblock_standard.ppo_agent import ActorCritic


def choose_ppo_greedy(model, state, action_mask):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(state_t)
        logits[~mask_t] = -1e9
        action = torch.argmax(logits, dim=1)

    return int(action.item())


def run_episode(algo, model, seed=0, show_board=False):
    env = BrainBlockGymEnv()
    state, info = env.reset(seed=seed)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        if algo == "dqn":
            action = select_dqn_action(model, state, info["action_mask"], epsilon=0.0)
        else:
            action = choose_ppo_greedy(model, state, info["action_mask"])

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if show_board:
            print("step:", steps, "reward:", reward)
            print_board(info["board"])

    return {
        "total_reward": total_reward,
        "covered_area": info["covered_area"],
        "steps": steps,
        "success": int(info.get("terminal_reason") == "solved"),
        "terminal_reason": info.get("terminal_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-board", action="store_true")
    args = parser.parse_args()

    if args.algo == "dqn":
        model = DQN()
    else:
        model = ActorCritic()

    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    results = []
    for episode in range(args.episodes):
        result = run_episode(args.algo, model, seed=args.seed + episode, show_board=args.show_board)
        results.append(result)
        print("episode", episode, result)

    success_rate = np.mean([r["success"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_area = np.mean([r["covered_area"] for r in results])

    print()
    print("summary")
    print("success rate:", round(float(success_rate), 3))
    print("average reward:", round(float(avg_reward), 3))
    print("average covered area:", round(float(avg_area), 3))


if __name__ == "__main__":
    main()
