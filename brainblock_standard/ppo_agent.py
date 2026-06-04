import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from brainblock_standard.env import BrainBlockGymEnv
from brainblock_standard.pieces import N_ACTIONS
from brainblock_standard.plotting import save_results


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


def ppo_update(model, optimizer, trajectory, clip_eps=0.2):
    if len(trajectory["states"]) == 0:
        return

    states = torch.tensor(np.array(trajectory["states"]), dtype=torch.float32)
    masks = torch.tensor(np.array(trajectory["masks"]), dtype=torch.bool)
    actions = torch.tensor(trajectory["actions"], dtype=torch.long)
    old_log_probs = torch.tensor(trajectory["log_probs"], dtype=torch.float32)
    old_values = torch.tensor(trajectory["values"], dtype=torch.float32)
    returns = torch.tensor(trajectory["returns"], dtype=torch.float32)

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

    env = BrainBlockGymEnv()
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    rows = []
    rollout = {"states": [], "masks": [], "actions": [], "log_probs": [], "values": [], "returns": []}

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed * 100000 + episode)
        done = False
        total_reward = 0
        steps = 0

        trajectory = {
            "states": [],
            "masks": [],
            "actions": [],
            "log_probs": [],
            "values": [],
            "rewards": [],
        }

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
            steps += 1

        returns = discounted_returns(trajectory["rewards"], gamma=0.95)
        
        rollout["states"].extend(trajectory["states"])
        rollout["masks"].extend(trajectory["masks"])
        rollout["actions"].extend(trajectory["actions"])
        rollout["log_probs"].extend(trajectory["log_probs"])
        rollout["values"].extend(trajectory["values"])
        rollout["returns"].extend(returns)

        if episode % ROLLOUT_EPISODES == 0:
            ppo_update(model, optimizer, rollout)
            rollout = {"states": [], "masks": [], "actions": [], "log_probs": [], "values": [], "returns": []}

        rows.append(
            {
                "episode": episode,
                "algo": "ppo",
                "reward": "area",
                "total_reward": total_reward,
                "covered_area": info["covered_area"],
                "steps": steps,
                "success": int(info.get("terminal_reason") == "solved"),
                "epsilon": "",
                "terminal_reason": info.get("terminal_reason"),
            }
        )

        if episode % 100 == 0:
            last = rows[-100:]
            avg_reward = np.mean([r["total_reward"] for r in last])
            avg_area = np.mean([r["covered_area"] for r in last])
            success_count = np.sum([r["success"] for r in last])
            print(f"PPO (seed {seed}) episode {episode} | avg reward {avg_reward:.2f} | avg area {avg_area:.2f} | success {success_count}")

    ppo_update(model, optimizer, rollout)

    save_results(rows, out_dir, "ppo")
    model_path = out_dir / "ppo_model.pt"
    torch.save(model.state_dict(), model_path)
    print("saved:", model_path)
    return rows
