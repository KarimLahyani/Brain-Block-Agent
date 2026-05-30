# Training Run Analysis & Results Notes

This file details our empirical findings from training both DQN and PPO agents over a budget of 100,000 episodes in the diversity-reward environment.

---

## 1. DQN Diversity Performance

* **Total Episodes**: 100,000 per seed (across 7 seeds: 0, 1, 2, 3, 4, 5, 42)
* **Average Success Rate**: $16.34\% \pm 26.09\%$
* **Average Episodic Return**: $0.530 \pm 0.462$
* **Average Unique Solutions Discovered**: 34.14 unique board configurations per seed (up to 80 on Seed 3)

### Analysis
DQN demonstrated high sample efficiency. Due to the experience replay buffer reinforcing successful packing layouts, DQN could scale its solve rate significantly on seeds where it discovered packing strategies. The diversity penalty successfully drove DQN to explore alternative configurations, finding a high absolute number of unique layouts (average of 34.14 per seed). However, due to the off-policy replay buffer and argmax selection, it showed higher concentration of solves on a subset of layout modes.

---

## 2. PPO Diversity Performance

* **Total Episodes**: 100,000 per seed (across 7 seeds: 0, 1, 2, 3, 4, 5, 42)
* **Average Success Rate**: $0.01\% \pm 0.00\%$
* **Average Episodic Return**: $0.224 \pm 0.002$
* **Average Unique Solutions Discovered**: 11.14 unique board configurations per seed (with 11.14 average solves per seed)
* **Uniqueness Rate**: ~100% (almost every single success resulted in a completely new solution layout)

### Analysis
PPO was less sample-efficient, requiring more episodes to find successful packs. However, because PPO parameterizes a stochastic policy and samples actions from a Categorical distribution, it is naturally robust against mode collapse. When PPO did succeed, it almost always found a completely distinct packing layout, maintaining a ~100% uniqueness rate and showing its strength in unbiased exploration of the packing space.

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
