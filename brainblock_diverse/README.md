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
SEED = 42            # random seed
```

All models, metric CSV files, and progress plots are saved in `runs/brainblock_diverse/`.

---

## Visualizing Discovered Solutions

Once training is complete, you can generate visual grids of all unique layouts discovered by the agents:
```powershell
python -X utf8 -m brainblock_diverse.make_solution_images
```

The output images are saved under `runs/brainblock_diverse/solution_images/` as `dqn_solutions_sheet.png` and `ppo_solutions_sheet.png`.

---

## Result Summary

In our experiments running 100,000 episodes:
* **DQN** discovered **16 unique solutions**.
* **PPO** discovered **12 unique solutions**.

Thanks to rollout batching (`ROLLOUT_EPISODES = 32`), PPO's training signal is highly stable, allowing its natural policy-gradient stochasticity to explore and discover a wide variety of unique configurations (12 configurations). DQN remains highly sample-efficient and finds its unique configurations faster in early training.
