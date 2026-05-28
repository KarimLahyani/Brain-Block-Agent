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
`runs/brainblock_diverse/solution_images/`

---

## 4. Summary of Training Runs and Results

We evaluated both agents over a budget of 100,000 episodes on the diversity-reward environment. The results are summarized below:

* **DQN (Deep Q-Network)**:
  * **Successes**: 57,659 solved episodes.
  * **Unique Solutions Discovered**: 16 unique board configurations.
  * DQN demonstrated high sample efficiency, quickly learning a stable policy and starting to solve the puzzle early in training.
* **PPO (Proximal Policy Optimization)**:
  * **Successes**: Stable solves after rollout batching improvements.
  * **Unique Solutions Discovered**: 12 unique board configurations.
  * PPO is naturally more stochastic due to its policy-gradient distribution sampling, which helped it discover a high variety of unique solutions (12 configurations). It required more episodes than DQN to achieve stable coverage, showing standard policy-gradient sample-efficiency characteristics on highly constrained discrete puzzles.

To stabilize PPO training, we update the model using rollouts collected over batches of 32 episodes (`ROLLOUT_EPISODES = 32`) rather than single-episode updates. This drastically reduces gradient noise.
