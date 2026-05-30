# BrainBlock DRL Project: Code & Design Walkthrough

This document provides a walkthrough of our code structure, design decisions, and core reinforcement learning concepts implemented for the BrainBlock project.

Our repository is organized into two primary implementation folders:
1. **`brainblock_standard/`**: The standard packing pipeline using a binary grid.
2. **`brainblock_diverse/`**: The diversity-reward pipeline that uses labeled grids and solution history tracking to discover multiple distinct solutions.

---

## 1. Environment Representations: Binary vs. Labeled Board

### Standard Version (`brainblock_standard/`)
In the standard version, the board is represented as a binary grid:
* `0` = Empty cell
* `1` = Occupied cell

While this is sufficient to teach the agent the rules of tetromino placement, it makes it impossible to distinguish between different solved configurations. Every solved board contains 40 filled cells and thus looks identical (all ones) to the network.

### Diversity-Reward Version (`brainblock_diverse/`)
To allow the system to recognize and track different solutions, the diverse version grid stores the specific tetromino ID that occupies each cell:
* `0` = Empty
* `1` = I piece
* `2` = O piece
* `3` = L piece
* `4` = Z piece
* `5` = T piece

Dividing this grid representation by 5 yields a normalized board state vector for the neural network, while preserving the identity of the pieces. The final arrangement of these IDs acts as a unique signature for each discovered solution.

---

## 2. Codebase Organization (Diversity-Reward Version)

All files for the diversity-aware agent reside in `brainblock_diverse/`:

* **`pieces.py`**:
  * Defines grid bounds ($8 \times 5$), inventory lists (2 of each piece: I, O, L, Z, T), and piece shapes.
  * Contains orientation rotation and reflection logic.
  * Encodes and decodes the 320 discrete actions.
  * Encodes the 50-dimensional observation state (40 grid values + 5 active piece one-hot features + 5 remaining inventory counts).
* **`diversity.py`**:
  * Implements `SolutionMemory`.
  * Checks if a solved board signature is a duplicate or overlaps heavily (greater than 85% similarity) with previously logged layouts.
  * Calculates the corresponding reward penalty (up to $-4.0$ for exact duplicates).
* **`env.py`**:
  * The Gymnasium-compatible environment that uses the piece ID board.
  * Modifies the final reward step on success by subtracting the diversity penalty.
  * Generates the `action_mask` containing valid action indices.
* **`dqn_agent.py`**:
  * Value-based agent using deep Q-networks, target model updates, and $\epsilon$-greedy exploration.
* **`ppo_agent.py`**:
  * Policy-gradient agent using actor-critic heads, action logits sampling, and rollout trajectory batching.
* **`main.py`**:
  * The entry script that executes the training loops in parallel.
* **`make_solution_images.py`**:
  * Reads the saved solutions from training and renders visual PNG grid sheets.

---

## 3. Comparing DQN and PPO Architectures

### Deep Q-Network (DQN)
* **Objective**: Learns the expected future reward $Q(s, a)$ for taking action $a$ in state $s$.
* **Exploration**: Uses $\epsilon$-greedy exploration. The model selects the action with the maximum estimated Q-value (argmax) most of the time.
* **Why it fits well**: DQN uses an *experience replay buffer*, allowing it to store and reuse transitions. This makes it highly sample-efficient for discrete grid environments with tight constraints.

### Proximal Policy Optimization (PPO)
* **Objective**: Directly learns the probability distribution $\pi(a|s)$ over actions.
* **Exploration**: Samples actions directly from a Categorical probability distribution. This stochasticity makes PPO excellent at exploring different branches of the state space.
* **Tuning for Stability**: Because episodes in BrainBlock are very short (max 10 steps), single-episode updates suffer from high gradient variance. We resolved this by batching rollouts across 32 episodes (`ROLLOUT_EPISODES = 32`) before performing optimization updates.

---

## 4. Summary of Empirical Results

We evaluated DQN and PPO across 7 random seeds ($0, 1, 2, 3, 4, 5, 42$) on both standard and diversity-reward configurations:

* **Standard Environment (20,000 episodes)**:
  * **DQN**: Success Rate of $0.02\% \pm 0.01\%$, Episodic Return of $0.245 \pm 0.004$.
  * **PPO**: Success Rate of $0.01\% \pm 0.01\%$, Episodic Return of $0.214 \pm 0.002$.
* **Diversity-Reward Environment (100,000 episodes)**:
  * **DQN**: Success Rate of $16.34\% \pm 26.09\%$, Episodic Return of $0.530 \pm 0.462$. Discovered an average of **34.14 unique solutions** per seed (up to 80 on Seed 3).
  * **PPO**: Success Rate of $0.01\% \pm 0.00\%$, Episodic Return of $0.224 \pm 0.002$. Discovered an average of **11.14 unique solutions** per seed.
  * **Uniqueness Rate**: PPO achieves near-100% uniqueness in its solves. DQN finds a larger absolute volume of unique solutions but shows higher concentration on specific layout modes.
  * **Diversity Penalty Advantage**: Interestingly, DQN performed significantly better in the diversity-reward environment ($16.34\%$ success rate) than the standard environment ($0.02\%$), as the duplication penalty acted as an intrinsic motivator that drove the agent away from repetitive local dead-ends.
