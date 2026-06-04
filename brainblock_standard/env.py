import gymnasium as gym
from gymnasium import spaces
import numpy as np

from brainblock_standard.pieces import (
    ALL_PIECES,
    COLS,
    INITIAL_INVENTORY,
    N_ACTIONS,
    PIECE_TO_ID,
    ROWS,
    count_filled,
    decode_action,
    encode_action,
    encode_state,
    in_bounds,
)


class BrainBlockGymEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(50,), dtype=np.float32)

        self.board = None
        self.queue = None
        self.current_piece = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = [[0 for _ in range(COLS)] for _ in range(ROWS)]

        self.queue = []
        for piece, count in INITIAL_INVENTORY.items():
            self.queue += [piece] * count

        self.queue = list(self.np_random.permutation(self.queue))
        self.current_piece = self.queue.pop(0)

        return self.get_obs(), self.get_info()

    def get_obs(self):
        return encode_state(self.board, self.current_piece, self.queue)

    def get_info(self):
        return {
            "board": self.board,
            "current_piece": self.current_piece,
            "remaining_pieces": self.queue,
            "valid_actions": self.get_valid_actions(),
            "action_mask": self.get_action_mask(),
            "covered_area": count_filled(self.board),
        }

    def is_valid(self, shape, row, col):
        for dr, dc in shape:
            r = row + dr
            c = col + dc

            if not in_bounds(r, c):
                return False

            if self.board[r][c] != 0:
                return False

        return True

    def place(self, piece, shape, row, col):
        piece_id = PIECE_TO_ID[piece]
        for dr, dc in shape:
            self.board[row + dr][col + dc] = piece_id

    def get_shape_from_action(self, action):
        orientation, row, col = decode_action(action)
        shapes = ALL_PIECES[self.current_piece]
        shape = shapes[orientation % len(shapes)]
        return shape, row, col

    def get_valid_actions(self):
        if self.current_piece is None:
            return []

        valid = []
        shapes = ALL_PIECES[self.current_piece]

        for orientation in range(8):
            shape = shapes[orientation % len(shapes)]
            for row in range(ROWS):
                for col in range(COLS):
                    if self.is_valid(shape, row, col):
                        valid.append(encode_action(orientation, row, col))

        return valid

    def get_action_mask(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        for action in self.get_valid_actions():
            mask[action] = True
        return mask

    def step(self, action):
        piece = self.current_piece
        shape, row, col = self.get_shape_from_action(action)

        if not self.is_valid(shape, row, col):
            reward = -1.0
            terminated = True
            truncated = False
            info = self.get_info()
            info["invalid"] = True
            info["terminal_reason"] = "invalid"
            return self.get_obs(), reward, terminated, truncated, info

        self.place(piece, shape, row, col)

        if len(self.queue) == 0:
            self.current_piece = None
            reward = 5.1
            terminated = True
            truncated = False
            info = self.get_info()
            info["success"] = True
            info["terminal_reason"] = "solved"
            return self.get_obs(), reward, terminated, truncated, info

        self.current_piece = self.queue.pop(0)

        if len(self.get_valid_actions()) == 0:
            reward = -0.4
            terminated = True
            truncated = False
            info = self.get_info()
            info["dead_end"] = True
            info["terminal_reason"] = "dead_end"
            return self.get_obs(), reward, terminated, truncated, info

        reward = 0.1

        terminated = False
        truncated = False
        info = self.get_info()
        info["terminal_reason"] = None
        return self.get_obs(), reward, terminated, truncated, info
