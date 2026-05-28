import numpy as np

ROWS = 5
COLS = 8
N_ACTIONS = 8 * ROWS * COLS

PIECES = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "L": [(0, 0), (1, 0), (2, 0), (2, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
}

PIECE_NAMES = list(PIECES.keys())
INITIAL_INVENTORY = {name: 2 for name in PIECES.keys()}


def print_board(board):
    for row in board:
        print(" ".join(str(cell) for cell in row))
    print()


def in_bounds(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS


def count_filled(board):
    return sum(sum(row) for row in board)


def rotate(shape):
    return [(c, -r) for r, c in shape]


def reflect(shape):
    return [(r, -c) for r, c in shape]


def normalize(shape):
    min_r = min(r for r, c in shape)
    min_c = min(c for r, c in shape)
    return [(r - min_r, c - min_c) for r, c in shape]


def get_orientations(shape):
    orientations = set()
    current = shape

    for _ in range(4):
        current = normalize(current)
        orientations.add(tuple(sorted(current)))

        reflected = normalize(reflect(current))
        orientations.add(tuple(sorted(reflected)))

        current = rotate(current)

    return [list(o) for o in orientations]


ALL_PIECES = {name: get_orientations(shape) for name, shape in PIECES.items()}


def encode_action(orientation, row, col):
    return orientation * (ROWS * COLS) + row * COLS + col


def decode_action(action):
    orientation = action // (ROWS * COLS)
    rem = action % (ROWS * COLS)
    row = rem // COLS
    col = rem % COLS
    return orientation, row, col


def encode_board(board):
    return [cell for row in board for cell in row]


def encode_piece(piece):
    return [1 if p == piece else 0 for p in PIECE_NAMES]


def encode_remaining(queue):
    counts = {p: 0 for p in PIECE_NAMES}
    for p in queue:
        counts[p] += 1
    return [counts[p] for p in PIECE_NAMES]


def encode_state(board, current_piece, queue):
    if current_piece is None:
        current_piece_features = [0 for _ in PIECE_NAMES]
    else:
        current_piece_features = encode_piece(current_piece)

    state = encode_board(board) + current_piece_features + encode_remaining(queue)
    return np.array(state, dtype=np.float32)
