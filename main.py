import sys
from pathlib import Path

# Add root directory to sys.path to ensure packages resolve properly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainblock_standard.train import main as run_standard
from brainblock_diverse.train import main as run_diverse

SEEDS = [0, 1, 2, 3, 4, 5, 42]
STANDARD_EPISODES = 200_000
DIVERSE_EPISODES = 200_000
EVAL_EPISODES = 1000


def main(seeds=SEEDS, standard_episodes=STANDARD_EPISODES, diverse_episodes=DIVERSE_EPISODES, eval_episodes=EVAL_EPISODES):
    print("=" * 60)
    print("STARTING COMPLETE EVALUATION SUITE SEQUENTIALLY")
    print("=" * 60)

    print("\n" + "=" * 50)
    print("STEP 1: RUNNING BRAINBLOCK STANDARD EXPERIMENTS...")
    print("=" * 50)
    run_standard(seeds=seeds, episodes=standard_episodes)

    print("\n" + "=" * 50)
    print("STEP 2: RUNNING BRAINBLOCK DIVERSE EXPERIMENTS...")
    print("=" * 50)
    run_diverse(seeds=seeds, episodes=diverse_episodes)


    print("\n" + "=" * 50)
    print("RUNNING AUTOMATIC EVALUATION OF SAVED AGENTS...")
    print("=" * 50)
    from evaluate_saved import main as run_evaluation
    run_evaluation(episodes=eval_episodes)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS AND EVALUATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
