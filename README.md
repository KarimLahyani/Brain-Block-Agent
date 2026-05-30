# BrainBlock DRL: Tetromino Packing with Deep Reinforcement Learning

This repository contains our implementation and comparative study of Deep Reinforcement Learning (DRL) agents applied to the BrainBlock packing puzzle—an $8 \times 5$ grid-packing problem using tetrominoes. We evaluate a value-based method (DQN) and a policy-gradient method (PPO) under two custom reward schemes, including a diversity-aware reward designed to discover multiple unique packing layouts.

The codebase is organized into two primary implementation directories:
* **`brainblock_standard/`**: The standard version using a binary board representation and an area-shaped reward.
* **`brainblock_diverse/`**: The diversity version which stores piece identities on the board and penalizes repeated/similar solved states to drive the discovery of multiple distinct solutions.

---

## 1. Standard Implementation

Located in the [brainblock_standard/](file:///c:/Users/karim%20lahyani/Downloads/CS%20445%20Codex/brainblock_standard) directory. This runs training for DQN, PPO, or both sequentially (and now in parallel) on the standard environment where the goal is simply to fill the board.

To run:
```powershell
# Running the module directly
python -X utf8 -m brainblock_standard.main
```

The default hyperparameters set in `brainblock_standard/main.py` are:
* `ALGO = "both"` (runs DQN and PPO in parallel)
* `EPISODES = 20_000`
* `SEED = 42`

---

## 2. Diversity Implementation

Located in the [brainblock_diverse/](file:///c:/Users/karim%20lahyani/Downloads/CS%20445%20Codex/brainblock_diverse) directory. This environment uses piece IDs (1-5) on the board rather than a binary grid. It tracks a history of solved configurations, applying a penalty to the final reward if the agent repeats an exact solution or a very similar one.

To run:
```powershell
# Running the module directly
python -X utf8 -m brainblock_diverse.main
```

The default hyperparameters set in `brainblock_diverse/main.py` are:
* `ALGO = "both"` (runs DQN and PPO in parallel)
* `EPISODES = 100_000`
* `SEED = 42`

---

## 3. Discovered Solution Visualizations

After training the diversity model, you can run the visualization script to generate image grids of all discovered packing solutions:
```powershell
python -X utf8 -m brainblock_diverse.make_solution_images
```

The generated PNG grids are saved to:
`results/brainblock_diverse/solution_images/`

---

## 4. Summary of Training Runs and Results

We evaluated DQN and PPO agents across 7 random seeds ($0, 1, 2, 3, 4, 5, 42$) on both the standard (20,000 episodes) and diversity-reward (100,000 episodes) environments. The aggregated results are summarized below:

| Environment & Algorithm | Success Rate (%) | Episodic Return | Episode Length | Invalid-Action Rate (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Standard DQN** | $0.02\% \pm 0.01\%$ | $0.245 \pm 0.004$ | $7.44 \pm 0.04$ | $0.00\% \pm 0.00\%$ |
| **Standard PPO** | $0.01\% \pm 0.01\%$ | $0.214 \pm 0.002$ | $7.13 \pm 0.02$ | $0.00\% \pm 0.00\%$ |
| **Diverse DQN** | $16.34\% \pm 26.09\%$ | $0.530 \pm 0.462$ | $7.83 \pm 0.69$ | $0.00\% \pm 0.00\%$ |
| **Diverse PPO** | $0.01\% \pm 0.00\%$ | $0.224 \pm 0.002$ | $7.23 \pm 0.02$ | $0.00\% \pm 0.00\%$ |

### Key Findings from Multi-Seed Evaluation:
* **Diversity Reward as Exploration Incentive**: The diversity penalty environment dramatically improved DQN's success rate ($16.34\%$ success rate) compared to the standard environment ($0.02\%$). By penalizing duplicate layouts, it prevents the agent from falling into early dead-end local optima (mode collapse).
* **Discovered Unique Solutions**:
  * **Diverse DQN**: Discovered an average of **34.14 unique solutions** per seed, reaching up to 80 unique configurations on Seed 3.
  * **Diverse PPO**: Discovered an average of **11.14 unique solutions** per seed.
* **Exploration Styles**:
  * **DQN** is highly sample-efficient and scales up the number of solves once a pathway is found, but displays a higher concentration of solves on a few modes.
  * **PPO** solves the board less frequently under this budget, but achieves a near-100% uniqueness rate (almost every single success is a completely unique layout) due to its stochastic policy formulation and entropy regularization.

To stabilize PPO training, rollouts are collected over batches of 32 episodes (`ROLLOUT_EPISODES = 32`) to reduce gradient noise.
