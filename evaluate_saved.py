import sys
from pathlib import Path
import shutil
import numpy as np
import torch

EVAL_EPISODES = 1000

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def choose_ppo_action(model, state, action_mask, deterministic=False, temperature=1.0):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(state_t)
        logits[~mask_t] = -1e9
        
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            if temperature != 1.0:
                logits = logits / temperature
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

    return int(action.item())

def select_dqn_action(model, state, action_mask):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_t).squeeze(0)
        mask = torch.full_like(q_values, -1e9)
        mask[action_mask] = 0
        q_values = q_values + mask
        return int(torch.argmax(q_values).item())

def run_episode(env, algo, model, print_board_fn, seed=0, show_board=False, temperature=1.0):
    state, info = env.reset(seed=seed)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        if algo == "dqn":
            action = select_dqn_action(model, state, info["action_mask"])
        else:
            action = choose_ppo_action(model, state, info["action_mask"], deterministic=False, temperature=temperature)

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        steps += 1

        if show_board:
            print(f"Step: {steps} | Action Reward: {reward:.3f}")
            print_board_fn(info["board"])

    return {
        "total_reward": total_reward,
        "covered_area": info["covered_area"],
        "steps": steps,
        "success": int(info.get("terminal_reason") == "solved"),
        "terminal_reason": info.get("terminal_reason"),
        "board": info["board"]
    }

def evaluate_single_model(mode, algo, model_path, episodes=10, start_seed=1000, show_board=False, verbose=True, temperature=1.0):
    if mode == "standard":
        from brainblock_standard.env import BrainBlockGymEnv
        from brainblock_standard.pieces import print_board
        from brainblock_standard.dqn_agent import DQN
        from brainblock_standard.ppo_agent import ActorCritic
        
        env = BrainBlockGymEnv()
        print_board_fn = print_board
        if algo == "dqn":
            model = DQN()
        else:
            model = ActorCritic()
    else:
        from brainblock_diverse.env import BrainBlockDiverseEnv
        from brainblock_diverse.pieces import print_board
        from brainblock_diverse.dqn_agent import DQN
        from brainblock_diverse.ppo_agent import ActorCritic
        from brainblock_diverse.diversity import SolutionMemory
        
        memory = SolutionMemory()
        env = BrainBlockDiverseEnv(solution_memory=memory)
        print_board_fn = print_board
        if algo == "dqn":
            model = DQN()
        else:
            model = ActorCritic()

    if verbose:
        print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    results = []
    unique_boards = set()
    for episode in range(episodes):
        if verbose and show_board:
            print(f"\n--- Episode {episode + 1} ---")
        result = run_episode(env, algo, model, print_board_fn, seed=start_seed + episode, show_board=show_board, temperature=temperature)
        results.append(result)
        
        if result["success"]:
            board_flat = tuple(tuple(row) for row in result["board"])
            unique_boards.add(board_flat)
            
        if verbose and show_board:
            print(f"Result: Success={result['success']} | Return={result['total_reward']:.3f} | Steps={result['steps']} | Reason={result['terminal_reason']}")

    success_rate = np.mean([r["success"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_area = np.mean([r["covered_area"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    return {
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_area": avg_area,
        "avg_steps": avg_steps,
        "unique_solutions": len(unique_boards),
        "results": results,
        "memory": memory if mode == "diverse" else None,
        "unique_boards": unique_boards
    }


def moving_average(values, window=50):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def plot_evaluation_run(mode, algo, results, fig_path, seed=None):
    import matplotlib.pyplot as plt
    
    rewards = [r["total_reward"] for r in results]
    areas = [r["covered_area"] for r in results]
    success = [r["success"] for r in results]
    
    # compute cumulative unique solutions count over episodes
    unique_boards = set()
    unique_counts = []
    for r in results:
        if r["success"]:
            board_flat = tuple(tuple(row) for row in r["board"])
            unique_boards.add(board_flat)
        unique_counts.append(len(unique_boards))
        
    plt.figure(figsize=(10, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(moving_average(rewards))
    plt.ylabel("reward")
    plt.grid(alpha=0.3)
    
    title_str = f"Evaluation: {mode.upper()} {algo.upper()}"
    if seed is not None:
        title_str += f" (Seed {seed})"
    plt.title(title_str)
    
    plt.subplot(4, 1, 2)
    plt.plot(moving_average(areas))
    plt.ylabel("covered")
    plt.grid(alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(moving_average(success))
    plt.ylabel("success")
    plt.grid(alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(unique_counts)
    plt.ylabel("unique")
    plt.xlabel("episode")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

def main(episodes=EVAL_EPISODES):
    eval_results_dir = Path("evaluation_results")
    eval_results_dir.mkdir(exist_ok=True)

    training_results_dir = Path("training_results")
    if not training_results_dir.exists():
        print("Error: 'training_results' directory does not exist. Please run training first.")
        sys.exit(1)
        
    pt_files = sorted(list(training_results_dir.rglob("*.pt")))
    if not pt_files:
        print("No saved models (.pt files) found in 'training_results/' directory.")
        sys.exit(0)
        
    print(f"Found {len(pt_files)} model files in '{training_results_dir}'. Parsing configurations...")
    models_to_evaluate = []
    
    for pf in pt_files:
        try:
            parts = pf.relative_to(training_results_dir).parts
            if len(parts) >= 4:
                env_folder = parts[0]
                seed_folder = parts[1]
                algo_folder = parts[2]
                
                mode = "diverse" if "diverse" in env_folder else "standard"
                seed = seed_folder.replace("seed_", "")
                algo = algo_folder
                
                models_to_evaluate.append({
                    "mode": mode,
                    "algo": algo,
                    "seed": seed,
                    "path": pf
                })
        except Exception:
            path_str = str(pf).lower()
            mode = "diverse" if "diverse" in path_str else "standard"
            algo = "ppo" if "ppo" in path_str else "dqn"
            models_to_evaluate.append({
                "mode": mode,
                "algo": algo,
                "seed": "unknown",
                "path": pf
            })

    if not models_to_evaluate:
        print("No models parsed for evaluation.")
        sys.exit(0)

    print(f"\nStarting evaluation of {len(models_to_evaluate)} model(s)...")
    summary_rows = []
    
    for i, m in enumerate(models_to_evaluate):
        p = m["path"]
        print(f"[{i+1}/{len(models_to_evaluate)}] Evaluating {m['mode']} {m['algo'].upper()} (train seed: {m['seed']}) from {p.name}...")
        
        try:
            metrics = evaluate_single_model(
                mode=m["mode"],
                algo=m["algo"],
                model_path=p,
                episodes=episodes,
                start_seed=1000,
                show_board=False,
                verbose=False
            )
            summary_rows.append({
                "mode": m["mode"],
                "algo": m["algo"],
                "train_seed": m["seed"],
                "success_rate": metrics["success_rate"],
                "avg_reward": metrics["avg_reward"],
                "avg_area": metrics["avg_area"],
                "avg_steps": metrics["avg_steps"],
                "unique_solutions": metrics["unique_solutions"]
            })
            
            # Setup partition directory
            partition_dir = eval_results_dir / f"brainblock_{m['mode']}" / f"seed_{m['seed']}" / m["algo"]
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Generate evaluation plot from testing run
            plot_evaluation_run(m["mode"], m["algo"], metrics["results"], partition_dir / "eval_plot.png", seed=m["seed"])
            print(f"  -> Generated testing plot: eval_plot.png")
            
            # 2. Generate solutions files and drawings from testing run
            unique_boards = metrics["unique_boards"]
            if unique_boards:
                sol_txt_file = partition_dir / "solutions.txt"
                if m["mode"] == "diverse":
                    metrics["memory"].save(sol_txt_file)
                else:
                    from brainblock_standard.pieces import ID_TO_PIECE
                    lines = []
                    for idx, board in enumerate(unique_boards, start=1):
                        lines.append(f"Solution {idx}")
                        for row in board:
                            lines.append(" ".join(ID_TO_PIECE.get(cell, ".") for cell in row))
                        lines.append("")
                    sol_txt_file.write_text("\n".join(lines), encoding="utf-8")
                
                from brainblock_diverse.make_solution_images import parse_solutions, draw_solution, make_contact_sheet
                solutions = parse_solutions(sol_txt_file)
                if solutions:
                    images_dir = partition_dir / "images"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    image_paths = []
                    for idx, (title, board) in enumerate(solutions, start=1):
                        image_path = images_dir / f"solution_{idx}.png"
                        draw_solution(board, title, image_path)
                        image_paths.append(image_path)
                    
                    sheet_path = partition_dir / "solutions_sheet.png"
                    make_contact_sheet(image_paths, sheet_path, f"{m['algo'].upper()} Seed {m['seed']} Evaluation Solutions")
                    print(f"  -> Generated {len(solutions)} testing solutions text, layout images, and contact sheet")

        except Exception as e:
            print(f"Failed to evaluate {p}: {e}")

    # Build and print consolidated summary table
    lines = []
    lines.append("="*95)
    lines.append(f"{'CONSOLIDATED EVALUATION RESULTS':^95}")
    lines.append("="*95)
    lines.append(f"{'Mode':<10} | {'Algo':<5} | {'Train Seed':<10} | {'Success Rate':<12} | {'Avg Return':<10} | {'Avg Steps':<10} | {'Avg Area':<8} | {'Unique Solns':<12}")
    lines.append("-"*95)
    for r in summary_rows:
        success_str = f"{r['success_rate']*100:.2f}%"
        lines.append(f"{r['mode']:<10} | {r['algo'].upper():<5} | {r['train_seed']:<10} | {success_str:<12} | {r['avg_reward']:<10.3f} | {r['avg_steps']:<10.2f} | {r['avg_area']:<8.2f} | {r['unique_solutions']}")
    lines.append("="*95)
    
    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # Save global summary to file
    out_file = eval_results_dir / "evaluation_summary.txt"
    try:
        out_file.write_text(summary_text, encoding="utf-8")
        print(f"Saved global evaluation summary table to: {out_file}")
    except Exception as e:
        print(f"Failed to save summary table to file: {e}")


if __name__ == "__main__":
    main()
