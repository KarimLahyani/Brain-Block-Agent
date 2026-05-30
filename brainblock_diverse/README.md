# Diversity-Aware BrainBlock packing

This directory contains our implementation of the diversity-reward version of the BrainBlock environment. The core goal of this setup is to prevent the agent from repeatedly converging to the exact same packing solution, driving it instead to discover multiple distinct, valid tetromino arrangements.

## How the Diversity Mechanism Works

To distinguish between solved configurations, the binary grid (used in the standard version) is replaced with a labeled board that stores piece identities:
* `0` = Empty
* `1` = I piece
* `2` = O piece
* `3` = L piece
* `4` = Z piece
* `5` = T piece

Whenever the agent successfully places all 10 pieces and fills the board, the final board layout is converted into a signature tuple. We use a `SolutionMemory` object (implemented in `diversity.py`) to log the signature and calculate dynamic rewards:
1. **New Solution**: If the signature has never been seen, the agent receives the full solve reward (`+5.1`).
2. **Duplicate Solution**: If the signature has already been discovered, the agent is penalized with a repeat penalty (`-4.0`), reducing the final action reward to `+1.1`.
3. **Similar Solution**: If the signature is not identical but overlaps heavily (greater than 85% cellular similarity) with an existing solution, a partial penalty (up to `-2.0`) is subtracted.

This forces the agents to search the MDP state space for new, distinct coordinate mappings.

---

## Code Directory Structure

* **`pieces.py`**: Handles piece shape definitions, piece IDs, coordinate transformation, state encoding, and final board signature conversion.
* **`diversity.py`**: Solution memory class that tracks discovered signature counts and computes overlap penalties.
* **`env.py`**: Gymnasium-compatible environment that uses labeled grids and computes diversity penalties.
* **`dqn_agent.py`**: DQN agent trained under the diversity reward.
* **`ppo_agent.py`**: PPO agent trained under the diversity reward.
* **`plotting.py`**: Logs metrics to CSV and outputs performance plots.
* **`main.py`**: Entry script that trains DQN and PPO in parallel.
* **`make_solution_images.py`**: Generates PNG sheets visualizing the final boards of all unique solutions discovered.

---

## Running the Training

Run the script from the root workspace folder:
```powershell
python -X utf8 -m brainblock_diverse.main
```

The default configuration in `main.py` is:
```python
ALGO = "both"        # choose "dqn", "ppo", or "both"
EPISODES = 100_000   # training budget
SEEDS = [0, 1, 2, 3, 4, 5, 42]  # random seeds
```

All metric CSV files, progress plots, and solution files are saved in `results/brainblock_diverse/`.

---

## Visualizing Discovered Solutions

Once training is complete, you can generate visual grids of all unique layouts discovered by the agents:
```powershell
python -X utf8 -m brainblock_diverse.make_solution_images
```

The output images are saved under `results/brainblock_diverse/solution_images/seed_{seed}/` along with contact sheets.

---

## Result Summary

Averaged across the 7 evaluated seeds:
* **DQN** discovered an average of **34.14 unique solutions** per seed (discovering up to 80 unique layouts on seed 3).
* **PPO** discovered an average of **11.14 unique solutions** per seed.

DQN remains highly sample-efficient and finds a larger volume of unique configurations. PPO's stochastic policy gradient with entropy regularization ensures that almost all of its solves (100% uniqueness rate) represent completely distinct packing layouts, making it highly effective at avoiding mode collapse.

