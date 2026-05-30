import sys
from pathlib import Path

# Add root directory to sys.path to ensure packages resolve properly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainblock_standard.main import main as run_standard
from brainblock_diverse.main import main as run_diverse

SEEDS = [0, 1, 2, 3, 4, 5, 42]


def main():
    print("=" * 60)
    print("STARTING COMPLETE EVALUATION SUITE SEQUENTIALLY")
    print("=" * 60)

    print("\n" + "=" * 50)
    print("STEP 1: RUNNING BRAINBLOCK STANDARD EXPERIMENTS...")
    print("=" * 50)
    run_standard(seeds=SEEDS)

    print("\n" + "=" * 50)
    print("STEP 2: RUNNING BRAINBLOCK DIVERSE EXPERIMENTS...")
    print("=" * 50)
    run_diverse(seeds=SEEDS)


    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
