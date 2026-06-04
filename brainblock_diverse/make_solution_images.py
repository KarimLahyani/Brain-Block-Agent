from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


RUN_DIR = Path("training_results/brainblock_diverse")
OUTPUT_DIR = RUN_DIR / "solution_images"

COLORS = {
    ".": "#f8fafc",
    "I": "#3b82f6",
    "O": "#f59e0b",
    "L": "#10b981",
    "Z": "#ef4444",
    "T": "#8b5cf6",
}


def parse_solutions(path):
    if not path.exists() or path.stat().st_size == 0:
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    solutions = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Solution"):
            title = line
            board = []
            i += 1

            while i < len(lines) and lines[i].strip():
                board.append(lines[i].split())
                i += 1

            if board:
                solutions.append((title, board))
        i += 1

    return solutions


def draw_solution(board, title, path):
    rows = len(board)
    cols = len(board[0])

    fig, ax = plt.subplots(figsize=(cols * 0.55, rows * 0.55))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    for r in range(rows):
        for c in range(cols):
            piece = board[r][c]
            ax.add_patch(
                Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=COLORS.get(piece, "#e5e7eb"),
                    edgecolor="#111827",
                    linewidth=1.1,
                )
            )
            ax.text(c + 0.5, r + 0.55, piece, ha="center", va="center", fontsize=11)

    ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_contact_sheet(image_paths, out_path, title):
    if not image_paths:
        return

    images = [plt.imread(path) for path in image_paths]
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = list(axes)
    else:
        axes = list(axes.ravel())

    for ax in axes:
        ax.axis("off")

    for ax, image, path in zip(axes, images, image_paths):
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " "), fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def process_agent(agent_name, seed):
    input_path = RUN_DIR / f"seed_{seed}" / agent_name / "discovered_solutions.txt"
    out_dir = OUTPUT_DIR / f"seed_{seed}" / agent_name
    out_dir.mkdir(parents=True, exist_ok=True)

    solutions = parse_solutions(input_path)
    image_paths = []

    for idx, (title, board) in enumerate(solutions, start=1):
        image_path = out_dir / f"solution_{idx}.png"
        draw_solution(board, title, image_path)
        image_paths.append(image_path)

    sheet_path = OUTPUT_DIR / f"seed_{seed}" / f"{agent_name}_solutions_sheet.png"
    make_contact_sheet(image_paths, sheet_path, f"{agent_name.upper()} seed {seed} discovered solutions")
    print(f"Seed {seed} {agent_name} solutions: {len(solutions)}")
    print("saved to:", out_dir)


def main(seeds=None):
    if seeds is None:
        seeds = [42, 100, 123, 456, 789]
    for seed in seeds:
        process_agent("dqn", seed)
        process_agent("ppo", seed)



if __name__ == "__main__":
    main()

