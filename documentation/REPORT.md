# BrainBlock Packing Environment: Comparing DQN and PPO with Diversity-Aware Rewards

**Course Project Report**  
**CS 445 / 545: Deep Reinforcement Learning**  

---

## Abstract
This report presents our formulation and solution of the BrainBlock tetromino packing puzzle as a finite-horizon Markov Decision Process (MDP). We implement and compare two distinct deep reinforcement learning (DRL) paradigms: a value-based method, Deep Q-Networks (DQN), and an on-policy policy-gradient method, Proximal Policy Optimization (PPO). The environment consists of an $8 \times 5$ board and a fixed inventory of 10 tetrominoes (2 of each type: I, O, L, Z, T), requiring the agent to place all pieces legally from a randomized queue. 

To satisfy the dual goals of solving the packing layout and discovering multiple distinct solutions, we designed two reward functions: a dense area-based reward and a diversity-aware reward that applies a penalty to repeated or highly overlapping solved states. Our experimental results across 7 random seeds show that DQN is highly sample-efficient under the diversity reward, achieving a success rate of $16.34\% \pm 26.09\%$ and discovering an average of $34.14$ unique solutions over 100,000 episodes. PPO, though less sample-efficient in terms of total solves (success rate of $0.01\% \pm 0.00\%$), demonstrated robust exploratory capabilities due to its stochastic policy formulation, discovering an average of $11.14$ unique solutions with a 100% uniqueness rate (every successful solve yielded a distinct layout). We discuss our MDP design choices, algorithm hyperparameter tuning, empirical results, and failure modes.

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

## 5. Evaluation Protocol & Experimental Setup
To rigorously evaluate the algorithms, we ran DQN and PPO across 7 random seeds ($0, 1, 2, 3, 4, 5, 42$) for both the standard and diverse environments. The training budgets were:
- **Standard Environment**: 20,000 episodes per seed.
- **Diverse Environment**: 100,000 episodes per seed.

### 5.1 Metrics Recorded
For each seed, we logged:
- **Success Rate**: The fraction of episodes solved.
- **Episodic Return**: Mean and standard deviation of cumulative reward per episode.
- **Episode Length**: Mean episode steps.
- **Invalid-Action Rate**: Rate of choosing invalid actions (held at 0% due to action masking).

---

## 6. Empirical Results & Discussion

### 6.1 Summary of Performance Metrics
The summary statistics averaged across all 7 seeds are reported in the table below:

| Environment & Algorithm | Success Rate (%) | Episodic Return | Episode Length | Invalid-Action Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard DQN** | $0.02\% \pm 0.01\%$ | $0.245 \pm 0.004$ | $7.44 \pm 0.04$ | $0.00\% \pm 0.00\%$ |
| **Standard PPO** | $0.01\% \pm 0.01\%$ | $0.214 \pm 0.002$ | $7.13 \pm 0.02$ | $0.00\% \pm 0.00\%$ |
| **Diverse DQN** | $16.34\% \pm 26.09\%$ | $0.530 \pm 0.462$ | $7.83 \pm 0.69$ | $0.00\% \pm 0.00\%$ |
| **Diverse PPO** | $0.01\% \pm 0.00\%$ | $0.224 \pm 0.002$ | $7.23 \pm 0.02$ | $0.00\% \pm 0.00\%$ |

### 6.2 Key Insights from Multi-Seed Training
1. **Diversity Incentive Drives Solves**: 
   Interestingly, the diverse environment (incorporating a penalty memory for duplicate layouts) vastly outperformed the standard environment in terms of success rate for DQN ($16.34\%$ vs $0.02\%$). This is because the diversity penalty prevents DQN from committing to local optima (mode collapse onto dead-end states), acting as an intrinsic motivation term that forces the agent to explore alternate packing trajectories.
2. **Exploration Variance**: 
   DQN_DIVERSE shows a high standard deviation in success rate ($16.34\% \pm 26.09\%$). This variation is driven by individual seeds: while some seeds (e.g., Seed 3 and Seed 5) found a massive number of solutions (80 and 74, respectively), others found fewer.
3. **PPO Exploitation vs DQN**: 
   Stochastic on-policy exploration (PPO) was very steady but struggled with high solve rates under a short budget. However, it achieved extremely high diversity relative to its solves (see uniqueness discussion below).

### 6.3 Discovered Solutions & Uniqueness
In the diverse environment, the number of unique solutions discovered across seeds is detailed below:

| Seed | DQN Discovered Solutions | PPO Discovered Solutions |
| :--- | :---: | :---: |
| **Seed 0** | 18 | 14 |
| **Seed 1** | 14 | 12 |
| **Seed 2** | 19 | 13 |
| **Seed 3** | 80 | 15 |
| **Seed 4** | 18 | 3 |
| **Seed 5** | 74 | 9 |
| **Seed 42** | 16 | 12 |
| **Average** | **34.14** | **11.14** |

Across all seeds, PPO's stochastic exploration preserved high uniqueness (almost 100% of PPO solves were distinct), while DQN succeeded in locating a much larger volume of unique packing layouts due to its off-policy replay buffer reinforcing and scaling exploration once the policy hit a solve.

### 6.4 Solution Frequency Distribution and Mode Collapse
To analyze the exploratory behavior of both algorithms under the diversity reward, we recorded the frequency distribution of discovered solutions (visualized in `results/brainblock_diverse/solution_frequencies_aggregated.png`). 

* **DQN Distribution**: Although DQN discovered a large average of 34.14 unique solutions per seed, it exhibited a significant concentration of successes on only a few specific layouts. For instance, once DQN's Q-network identifies a highly stable trajectory, its deterministic exploitation selection (argmax of Q-values) continues to favor that path, even when penalized, because the penalized reward is still higher than the reward for early dead-ends.
* **PPO Distribution**: In stark contrast, PPO solved the board fewer times but found a unique solution almost every single time—representing near 100% uniqueness. Because PPO is a stochastic policy-gradient algorithm, it samples actions directly from a Categorical probability distribution parameterized by the actor network's output logits. Combined with entropy regularization, PPO is naturally resistant to mode collapse, yielding high-diversity exploration.

---

## 7. Discovered Solutions Visualization
Both agents succeeded in discovering a large number of solutions. Below are two sample board layouts extracted from our training runs:

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
The full set of discovered configurations is visualized in `results/brainblock_diverse/solution_images/`.

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

