# Standard BrainBlock Packing Pipeline

This directory contains our standard implementation of the BrainBlock environment and training scripts. In this version, the environment is solved as a standard packing puzzle using a binary grid representation, and the goal of the agents is simply to maximize the covered grid area.

## Code Directory Structure

* **`pieces.py`**: Handles piece shape definitions, rotation/reflection transforms, action space mapping (translating action IDs 0-319 to coordinates and orientations), and state vector encoding.
* **`env.py`**: Custom Gymnasium-compatible environment.
* **`dqn_agent.py`**: Deep Q-Network (DQN) implementation using an experience replay buffer and a target network.
* **`ppo_agent.py`**: Proximal Policy Optimization (PPO) implementation using an Actor-Critic architecture and rollout trajectory batching.
* **`plotting.py`**: Metric logger and plot generator.
* **`main.py`**: Entry-point script that launches training (runs both DQN and PPO in parallel by default).

---

## Running the Training

You can run the training directly from the root workspace folder using:
```powershell
python -X utf8 -m brainblock_standard.main
```

Alternatively, you can use the VS Code Run button.

The training parameters are configured at the top of `main.py`:
```python
ALGO = "both"      # choose "dqn", "ppo", or "both"
EPISODES = 20_000  # total training episodes
SEEDS = [0, 1, 2, 3, 4, 5, 42]  # random seeds
```

Training metrics and plots will be saved to `results/brainblock_standard/`.


---

## Environment Design and State Representation

* **State Space**: The environment produces a 50-dimensional observation vector. It contains the flat binary board state (40 elements), a one-hot encoding of the current piece (5 elements), and the remaining piece counts in the randomized queue (5 elements).
* **Action Space**: A discrete space of size 320 (8 orientations × 5 rows × 8 columns).
* **Reward Function**: 
  * `+0.1` for placing a piece legally.
  * `+5.1` for placing the final piece and successfully completing the board.
  * `-0.4` if the agent makes a move that results in an early dead-end (no valid placements remaining).
  * `-1.0` if an invalid placement is attempted (in practice, this is avoided by applying the action mask).
