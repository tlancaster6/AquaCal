"""Script entry point for refractive vs non-refractive comparison experiments.

Usage:
    python tests/synthetic/compare_refractive.py --output-dir results/refractive_comparison
    python tests/synthetic/compare_refractive.py --output-dir results/ --experiment 1,2 --seed 123
"""

import sys
from pathlib import Path
import argparse
import time

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tests.synthetic.experiments import (
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    assemble_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run refractive vs non-refractive calibration comparison experiments"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for all experiment outputs",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="1,2,3",
        help="Comma-separated list of experiment numbers to run (default: 1,2,3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Parse experiment numbers
    exp_numbers = [int(x.strip()) for x in args.experiment.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Refractive vs Non-Refractive Calibration Comparison")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Experiments: {exp_numbers}")
    print(f"Base seed: {args.seed}")
    print()

    start_time = time.time()

    # Run experiments
    if 1 in exp_numbers:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Parameter Fidelity")
        print("=" * 70)
        try:
            run_experiment_1(output_dir, seed=args.seed)
        except Exception as e:
            print(f"ERROR: Experiment 1 failed: {e}")
            import traceback

            traceback.print_exc()

    if 2 in exp_numbers:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Depth Generalization")
        print("=" * 70)
        try:
            run_experiment_2(output_dir, seed=args.seed)
        except Exception as e:
            print(f"ERROR: Experiment 2 failed: {e}")
            import traceback

            traceback.print_exc()

    if 3 in exp_numbers:
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: Depth Scaling")
        print("=" * 70)
        try:
            run_experiment_3(output_dir, seed=args.seed)
        except Exception as e:
            print(f"ERROR: Experiment 3 failed: {e}")
            import traceback

            traceback.print_exc()

    # Assemble summary
    print("\n" + "=" * 70)
    print("Assembling summary...")
    print("=" * 70)
    assemble_summary(output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Total runtime: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    print("=" * 70)


if __name__ == "__main__":
    main()
