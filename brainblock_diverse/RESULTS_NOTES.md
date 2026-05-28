# Training Run Analysis & Results Notes

This file details our empirical findings from training both DQN and PPO agents over a budget of 100,000 episodes in the diversity-reward environment.

---

## 1. DQN Diversity Performance

* **Total Episodes**: 100,000
* **Successes (Solves)**: 57,659
* **Unique Solutions Discovered**: 16 unique board configurations
* **Last 1,000 Episodes Success Rate**: 69.8%
* **Last 1,000 Episodes Average Covered Area**: 37.17 / 40 cells

### Analysis
DQN demonstrated exceptional sample efficiency. Due to the reuse of experiences via the replay buffer, DQN quickly learned the placement rules and began completing the board early in training. As it solved the board repeatedly, the diversity penalty successfully pushed it to explore alternative layouts, leading to 16 unique solutions.

---

## 2. PPO Diversity Performance

* **Total Episodes**: 100,000
* **Successes (Solves)**: 12
* **Unique Solutions Discovered**: 12 unique board configurations
* **Uniqueness Rate**: 100% (every single success resulted in a completely new solution layout)

### Analysis
PPO was less sample efficient than DQN, requiring more training steps to achieve its first successes. However, because PPO uses a stochastic policy-gradient formulation (sampling actions from a Categorical probability distribution), it did not get stuck repeating the same solution. In fact, every single time PPO successfully completed the board, it discovered a brand new solution, resulting in 12 distinct packing configurations.

---

## 3. Stability Modifications Made to PPO

Originally, PPO was set to update the network after every single episode. Because BrainBlock episodes are highly sequential and short (at most 10 steps), single-episode updates introduced extreme gradient noise, causing the agent to collapse.

To fix this, we implemented **trajectory rollout batching**:
* **`ROLLOUT_EPISODES = 32`**: The agent collects data from 32 full episodes before computing returns and performing policy updates.
* **`PPO_EPOCHS = 6`**: The actor-critic network updates over 6 epochs per batch.
* **`ENTROPY_COEF = 0.02`**: We tuned the entropy regularization coefficient to maintain exploration and prevent early policy convergence.

This adjustment stabilized PPO training and allowed it to reach its 12 unique solutions.

---

## 4. Key Takeaways for Report/Presentation

1. **Value-Based vs. Policy-Gradient Efficiency**: DQN's experience replay buffer is highly suited for discrete spatial-packing problems with sparse solutions, leading to rapid convergence. PPO, as an on-policy method, requires a higher sample budget but is highly resistant to local minima and excels at diverse exploration.
2. **Diversity Reward Effectiveness**: Labeled board cells (piece IDs 1–5) and cellular similarity checks successfully forced both agents to deviate from previously learned trajectories, proving the validity of our reward shape design.
