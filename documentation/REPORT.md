# BrainBlock Packing Environment: Comparing DQN and PPO with Diversity-Aware Rewards

**Course Project Report**  
**CS 445 / 545: Deep Reinforcement Learning**  

---

## Abstract
This report presents our formulation and solution of the BrainBlock tetromino packing puzzle as a finite-horizon Markov Decision Process (MDP). We implement and compare two distinct deep reinforcement learning (DRL) paradigms: a value-based method, Deep Q-Networks (DQN), and an on-policy policy-gradient method, Proximal Policy Optimization (PPO). The environment consists of an $8 \times 5$ board and a fixed inventory of 10 tetrominoes (2 of each type: I, O, L, Z, T), requiring the agent to place all pieces legally from a randomized queue. 

To satisfy the dual goals of solving the packing layout and discovering multiple distinct solutions, we designed two reward functions: a dense area-based reward and a diversity-aware reward that applies a penalty to repeated or highly overlapping solved states. Our experimental results show that DQN is highly sample-efficient, achieving a success rate of 69.8% and discovering 16 unique solutions over 100,000 episodes. PPO, though less sample-efficient, demonstrated robust exploratory capabilities due to its stochastic policy formulation, discovering 12 unique solutions out of 12 successful episodes (a 100% uniqueness rate). We discuss our MDP design choices, algorithm hyperparameter tuning, empirical results, and failure modes.

---

## 1. Introduction
The BrainBlock puzzle is a constrained spatial packing problem played on a grid of width $W = 8$ and height $H = 5$ (40 cells in total). The inventory contains a set of 10 tetrominoes, each covering exactly 4 unit cells. A complete solution occupies all 40 cells with no empty spaces or overlaps. At the start of each episode, the inventory is shuffled into a queue, and the agent must place the piece at the head of the queue on the board.

From an RL perspective, this puzzle is a sequential decision-making task characterized by a high degree of coupling between steps. A placement that is locally legal at step 1 may create isolated empty spaces that make a full solution impossible at step 9. Thus, the environment presents severe challenges in credit assignment, exploration, and delayed rewards. 

We compare a value-based method (DQN) against a policy-gradient method (PPO). Additionally, to satisfy the requirement of finding multiple unique solutions (at least 5), we propose a diversity-reward mechanism that tracks the signature of solved boards and penalizes agents for duplicating known configurations.

---

## 2. MDP Formulation
We model the tetromino packing problem as a finite-horizon MDP defined by the tuple $(S, A, P, R, \gamma)$.

### 2.1 State Space ($S$)
The state representation must provide the agent with the current board state, the piece to be placed, and information about the remaining pieces in the queue. We encode this as a 50-dimensional flat vector:
* **Board Grid (40 elements)**: A flattened representation of the $8 \times 5$ grid.
  * In the **Standard Version**, the board is binary ($0$ for empty, $1$ for occupied).
  * In the **Diversity Version**, the board cells store the specific piece ID ($0$ = empty, $1$ = I, $2$ = O, $3$ = L, $4$ = Z, $5$ = T). The values are normalized by dividing by 5 before feeding them to the network. Labeled grids are necessary to distinguish final solved layouts from one another.
* **Active Piece (5 elements)**: A one-hot encoding of the tetromino type currently at the head of the queue.
* **Remaining Inventory (5 elements)**: Integer counts indicating the number of remaining pieces of each type (I, O, L, Z, T) left in the queue.

### 2.2 Action Space ($A$)
The action space is discrete with a size of 320:
$$|A| = 8 \times 5 \times 8 = 320$$
An action corresponds to the Cartesian product of three components:
$$\text{Action} \in \{\text{orientation}\} \times \{\text{y-coordinate}\} \times \{\text{x-coordinate}\}$$
where:
* $\text{orientation} \in \{0, 1, \dots, 7\}$ representing the 8 rotated/reflected variants of the tetromino.
* $\text{row} \in \{0, 1, \dots, 4\}$ representing the vertical anchor on the board.
* $\text{col} \in \{0, 1, \dots, 7\}$ representing the horizontal anchor.

Redundant orientations due to piece symmetry are mapped back to valid shapes using modulo arithmetic (e.g., `orientation % len(shapes)`). We use action masking to restrict the agent's action selection to valid, non-overlapping, and in-bounds coordinates.

### 2.3 Transition Dynamics ($P$)
* **Initialization ($S_0$)**: The board is cleared, the inventory is shuffled into a queue of length 10, and the first piece is popped as the active piece.
* **Step ($S_t \to S_{t+1}$)**: The agent selects an action index. If the placement is legal, the board state is updated, the active piece is removed, and the next piece in the queue becomes the active piece. If the placement is illegal, the episode terminates.

### 2.4 Terminal Conditions
An episode terminates when:
1. **Solve**: All 10 pieces are placed legally, covering all 40 cells of the board.
2. **Dead-End**: The active piece has no legal placements remaining on the board.
3. **Invalid Action**: The agent chooses an illegal action (in practice, this is prevented by action masking).

---

## 3. Reward Engineering
We evaluate and compare two distinct reward formulations:

### 3.1 Reward Function 1: Base Area Reward
Designed to provide dense intermediate feedback to guide the agent toward placing pieces, defined as:
* **Legal Step**: $+0.1$ (representing the proportion of the board covered: $4 \text{ cells} / 40 \text{ cells} = 0.1$).
* **Solve Bonus**: $+5.0$ (leading to a total reward of $+5.1$ on the final step).
* **Dead-End Penalty**: $-0.4$ for getting stuck.
* **Invalid Action Penalty**: $-1.0$ (used as a fallback penalty).

A solved episode yields a total cumulative return of:
$$\sum R = 9 \times 0.1 + 5.1 = 6.0$$

### 3.2 Reward Function 2: Diversity-Aware Reward
To prevent the agent from repeatedly exploiting the same solution layout, we introduce a `SolutionMemory` buffer. If the agent successfully solves the board, we convert the final labeled grid into a unique signature. The reward for the final step is modified by a diversity penalty:
$$R_{\text{solve}} = 5.1 - \text{penalty}$$
where:
* **Exact Duplicate**: If the final board signature matches an already-discovered layout, a penalty of $-4.0$ is applied (reducing the solve reward to $+1.1$).
* **High Similarity**: We compute the cellular overlap similarity with all previously discovered solutions:
  $$\text{Similarity} = \frac{\text{matching cells}}{\text{total cells}}$$
  If the maximum similarity exceeds a threshold of $0.85$, we subtract a scaled penalty up to $-2.0$:
  $$\text{penalty} = 2.0 \times \frac{\text{Similarity} - 0.85}{1.0 - 0.85}$$
* **New Solution**: If the layout is completely novel, the penalty is $0.0$ and the layout is registered in the memory.

---

## 4. Algorithms

### 4.1 Deep Q-Networks (DQN)
DQN is a value-based method that learns the action-value function $Q(s, a)$.
* **Network Architecture**: Input layer (50) $\to$ Hidden Layer (128, ReLU) $\to$ Hidden Layer (128, ReLU) $\to$ Output Layer (320 linear Q-values).
* **Exploration**: Epsilon-greedy exploration starting at $\epsilon = 0.9$ and decaying by $0.995$ per episode to a minimum of $0.05$.
* **Replay Buffer**: Capacity of 10,000 transitions; updates are performed using mini-batches of size 32.
* **Target Network**: Copied from the main network every 100 episodes to maintain training stability.
* **Action Masking**: Illegal actions are masked by setting their Q-values to $-1\times 10^9$ prior to taking the argmax.

### 4.2 Proximal Policy Optimization (PPO)
PPO is an on-policy policy-gradient method that learns a policy distribution $\pi(a|s)$ and a state-value function $V(s)$.
* **Network Architecture**: Shared representation trunk (50 $\to 128 \to 128$, ReLU) splitting into two heads:
  * **Actor Head**: 320 action logits (masked with $-1\times 10^9$ for illegal actions and sampled via a Categorical distribution).
  * **Critic Head**: 1 scalar value estimating $V(s)$.
* **Rollout Batching**: To reduce policy-gradient variance in this short-horizon environment, PPO collects trajectories over batches of 32 episodes (`ROLLOUT_EPISODES = 32`) before performing optimization.
* **Update Policy**: Uses 6 epochs per batch with a clipping parameter $\epsilon_{\text{clip}} = 0.2$, an entropy coefficient of $0.02$ to encourage exploration, and a value loss coefficient of $0.5$.

---

## 5. Experimental Setup
* **Grid Dimension**: $8 \times 5$.
* **Inventory**: 2 of each piece type (I, O, L, Z, T).
* **Episode Horizon**: Max 10 placement steps.
* **Training Budget**: 100,000 episodes.
* **Verification**: We ran evaluation rollouts over multiple seeds to log average rewards, covered areas, and unique solutions.

---

## 6. Results and Discussion

### 6.1 DQN Performance
* **Successes**: 57,659 solved episodes.
* **Unique Solutions Discovered**: 16 unique layouts.
* **Last 1,000 Success Rate**: 69.8%.
* **Last 1,000 Average Covered Area**: 37.17 cells.

DQN proved to be highly sample-efficient. The experience replay buffer allowed the network to reinforce rare successful placements. As the success rate climbed, the diversity penalty kicked in, forcing DQN to redirect its policy toward unused areas of the action space, ultimately yielding 16 distinct board configurations.

### 6.2 PPO Performance
* **Successes**: 12 solved episodes.
* **Unique Solutions Discovered**: 12 unique layouts.
* **Uniqueness Rate**: 100% (every single solve resulted in a completely novel board layout).

PPO required significantly more episodes to find its first success. However, PPO's stochastic exploration mechanism (sampling from a probability distribution rather than selecting the argmax) made it highly effective at finding diverse solutions. Every single successful episode in PPO's training run resulted in a completely different solution, discovering 12 unique layouts.

### 6.3 Solution Frequency Distribution and Mode Collapse
To analyze the exploratory behavior of both algorithms under the diversity reward, we recorded the frequency distribution of discovered solutions (visualized in `runs/brainblock_diverse/solution_frequencies.png`). 

* **DQN Distribution**: Although DQN discovered 16 unique solutions, it exhibited a significant concentration of successes on only a few specific layouts. For instance, Solution 2 and Solution 3 were solved 17,640 and 22,993 times respectively, whereas several other solutions (e.g., Solutions 1, 5, and 9) were discovered only a handful of times. This is a classic indication of **mode collapse** in value-based RL. Once DQN's Q-network identifies a highly stable trajectory, its deterministic exploitation selection (argmax of Q-values) continues to favor that path, even when penalized, because the penalized reward is still higher than the reward for early dead-ends.
* **PPO Distribution**: In stark contrast, PPO solved the board exactly 12 times and found 12 unique solutions—representing a 100% uniqueness rate where each solution layout was seen exactly once. Because PPO is a stochastic policy-gradient algorithm, it samples actions directly from a Categorical probability distribution parameterized by the actor network's output logits. Combined with entropy regularization, PPO is naturally resistant to mode collapse, yielding high-diversity exploration.

---

## 7. Discovered Solutions Visualization
Both agents succeeded in discovering the required 5 solutions. Below are two sample board layouts extracted from our training runs:

**Solution Example A (DQN)**:
```text
L L L Z Z O O I
L T Z Z L O O I
T T T T L L L I
O O T T T Z Z I
O O I I I I Z Z
```

**Solution Example B (PPO)**:
```text
T I I I I L L L
T T Z Z T Z Z L
T Z Z T T L Z Z
O O O O T L L L
O O O O I I I I
```
The full set of discovered configurations is visualized in `runs/brainblock_diverse/solution_images/`.

---

## 8. Failure Modes & Credit Assignment
* **Dead-Ends**: The primary failure mode was the agent reaching a state where the active piece had no legal placement. This is a classic credit assignment failure: the agent selected locally legal actions early in the episode that left isolated holes (e.g., 2-cell regions) that could not fit any tetromino.
* **Action Masking**: The action mask was highly effective. No invalid-action terminations occurred during exploitation phases, indicating that the agent successfully bypassed the boundary constraint problem to focus entirely on packing strategy.

---

## 9. Conclusions and Future Work
We formulated the BrainBlock puzzle as an MDP and compared DQN and PPO under standard and diversity-aware reward structures. DQN was the more sample-efficient algorithm due to experience replay. PPO had lower sample efficiency but exhibited excellent novelty search, achieving a 100% uniqueness rate on its solved episodes. The diversity-aware reward successfully forced both agents to discover multiple solutions.

Future improvements could include:
1. **Convolutional Grid Encoders**: Replacing the flat MLP input with a 2D CNN to exploit spatial grid features.
2. **Dead-Space Penalties**: Adding an intermediate heuristic penalty if the board contains empty regions whose areas are not divisible by 4, preventing dead-ends.
3. **Double DQN**: Utilizing Double DQN to mitigate Q-value overestimation.
