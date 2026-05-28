import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from brainblock_diverse.diversity import SolutionMemory
from brainblock_diverse.env import BrainBlockDiverseEnv
from brainblock_diverse.pieces import N_ACTIONS
from brainblock_diverse.plotting import save_results


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def select_dqn_action(model, state, action_mask, epsilon):
    valid_actions = np.flatnonzero(action_mask)
    if len(valid_actions) == 0:
        return random.randint(0, N_ACTIONS - 1)

    if random.random() < epsilon:
        return int(random.choice(valid_actions))

    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_t).squeeze(0)
        mask = torch.full((N_ACTIONS,), -1e9)
        mask[action_mask] = 0
        q_values = q_values + mask
        return int(torch.argmax(q_values).item())


def train_dqn_step(model, target_model, optimizer, buffer, batch_size=32, gamma=0.95):
    if len(buffer) < batch_size:
        return None

    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, dones, next_masks = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)
    next_masks = torch.tensor(np.array(next_masks), dtype=torch.bool)

    q_values = model(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_model(next_states)
        next_q[~next_masks] = -1e9
        max_next_q = next_q.max(1)[0]
        target = rewards + gamma * max_next_q * (1 - dones)

    loss = F.mse_loss(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_dqn(episodes, seed, out_dir):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    memory = SolutionMemory()
    env = BrainBlockDiverseEnv(solution_memory=memory)
    model = DQN()
    target_model = DQN()
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer()
    epsilon = 0.9
    rows = []

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed * 100000 + episode)
        done = False
        total_reward = 0
        steps = 0
        diversity_penalty = 0

        while not done:
            action = select_dqn_action(model, state, info["action_mask"], epsilon)
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            buffer.push((state, action, reward, next_state, done, next_info["action_mask"]))
            train_dqn_step(model, target_model, optimizer, buffer)

            state = next_state
            info = next_info
            total_reward += reward
            diversity_penalty += info.get("diversity_penalty", 0)
            steps += 1

        if episode % 100 == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(0.05, epsilon * 0.995)

        rows.append(
            {
                "episode": episode,
                "algo": "dqn_diverse",
                "reward": "area_with_diversity_penalty",
                "total_reward": total_reward,
                "covered_area": info["covered_area"],
                "steps": steps,
                "success": int(info.get("terminal_reason") == "solved"),
                "new_solution": int(info.get("is_new_solution", False)),
                "unique_solutions": info.get("unique_solutions", len(memory)),
                "diversity_penalty": diversity_penalty,
                "epsilon": epsilon,
                "terminal_reason": info.get("terminal_reason"),
            }
        )

        if episode % 100 == 0:
            last = rows[-100:]
            avg_reward = np.mean([r["total_reward"] for r in last])
            unique = rows[-1]["unique_solutions"]
            success_count = np.sum([r["success"] for r in last])
            print(f"DQN episode {episode} | avg reward {avg_reward:.2f} | success {success_count} | unique {unique}")

    save_results(rows, out_dir, "dqn_diverse")
    torch.save(model.state_dict(), out_dir / "dqn_diverse_model.pt")
    memory.save(out_dir / "discovered_solutions.txt")
    print("saved:", out_dir / "dqn_diverse_model.pt")
    print("saved:", out_dir / "discovered_solutions.txt")
    return rows
