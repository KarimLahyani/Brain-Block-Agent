import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from brainblock_diverse.diversity import SolutionMemory
from brainblock_diverse.env import BrainBlockDiverseEnv
from brainblock_diverse.pieces import N_ACTIONS
from brainblock_diverse.plotting import save_results


ROLLOUT_EPISODES = 32
PPO_EPOCHS = 6
ENTROPY_COEF = 0.02
VALUE_COEF = 0.5


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value


def ppo_action(model, state, action_mask):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(state_t)
        logits[~mask_t] = -1e9
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

    return int(action.item()), float(log_prob.item()), float(value.item())


def discounted_returns(rewards, gamma=0.95):
    returns = []
    g = 0
    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)
    return returns


def ppo_update(model, optimizer, trajectory, gamma=0.95, clip_eps=0.2):
    if len(trajectory["states"]) == 0:
        return

    states = torch.tensor(np.array(trajectory["states"]), dtype=torch.float32)
    masks = torch.tensor(np.array(trajectory["masks"]), dtype=torch.bool)
    actions = torch.tensor(trajectory["actions"], dtype=torch.long)
    old_log_probs = torch.tensor(trajectory["log_probs"], dtype=torch.float32)
    old_values = torch.tensor(trajectory["values"], dtype=torch.float32)
    returns = torch.tensor(discounted_returns(trajectory["rewards"], gamma), dtype=torch.float32)

    advantages = returns - old_values
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        logits, values = model(states)
        logits[~masks] = -1e9
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_ppo(episodes, seed, out_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    memory = SolutionMemory()
    env = BrainBlockDiverseEnv(solution_memory=memory)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    rows = []
    rollout = {"states": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": []}

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed * 100000 + episode)
        done = False
        total_reward = 0
        steps = 0
        diversity_penalty = 0
        trajectory = {"states": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": []}

        while not done:
            action, log_prob, value = ppo_action(model, state, info["action_mask"])
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            trajectory["states"].append(state)
            trajectory["masks"].append(info["action_mask"])
            trajectory["actions"].append(action)
            trajectory["log_probs"].append(log_prob)
            trajectory["values"].append(value)
            trajectory["rewards"].append(reward)

            state = next_state
            info = next_info
            total_reward += reward
            diversity_penalty += info.get("diversity_penalty", 0)
            steps += 1

        for key in rollout:
            rollout[key].extend(trajectory[key])

        if episode % ROLLOUT_EPISODES == 0:
            ppo_update(model, optimizer, rollout)
            rollout = {"states": [], "masks": [], "actions": [], "log_probs": [], "values": [], "rewards": []}

        rows.append(
            {
                "episode": episode,
                "algo": "ppo_diverse",
                "reward": "area_with_diversity_penalty",
                "total_reward": total_reward,
                "covered_area": info["covered_area"],
                "steps": steps,
                "success": int(info.get("terminal_reason") == "solved"),
                "new_solution": int(info.get("is_new_solution", False)),
                "unique_solutions": info.get("unique_solutions", len(memory)),
                "diversity_penalty": diversity_penalty,
                "epsilon": "",
                "terminal_reason": info.get("terminal_reason"),
            }
        )

        if episode % 100 == 0:
            last = rows[-100:]
            avg_reward = np.mean([r["total_reward"] for r in last])
            unique = rows[-1]["unique_solutions"]
            success_count = np.sum([r["success"] for r in last])
            print(f"PPO (seed {seed}) episode {episode} | avg reward {avg_reward:.2f} | success {success_count} | unique {unique}")

    ppo_update(model, optimizer, rollout)

    save_results(rows, out_dir, "ppo_diverse")
    memory.save(out_dir / "discovered_solutions.txt")
    print("saved:", out_dir / "discovered_solutions.txt")
    return rows
