"""Full TimesFM MT5 pipeline — one command to rule them all.

Runs every step in the correct order:
  1. Export fresh OHLCV data from MetaTrader 5
  2. Generate moon phase features
  3. Generate economic calendar features
  4. Build merged covariate dataset
  5. Fine-tune TimesFM with LoRA adapters
  6. Forecast all instruments
  7. Export forecasts to MT5 for the indicator

Usage:
    python mt5/run_pipeline.py                  # full pipeline
    python mt5/run_pipeline.py --skip-finetune  # reuse existing adapter
    python mt5/run_pipeline.py --from-step 4    # resume from covariates
    python mt5/run_pipeline.py --only 6 7       # forecast + export only

Requires MetaTrader 5 terminal to be running for steps 1 and 7.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

# Pipeline steps in execution order
STEPS = [
    (1, "export_mt5_data.py",              "Export OHLCV from MetaTrader 5"),
    (2, "generate_moon_phases.py",         "Generate moon phase features"),
    (3, "generate_economic_calendar.py",   "Generate economic calendar"),
    (4, "build_covariates.py",             "Build merged covariates"),
    (5, "finetune.py",                     "Fine-tune TimesFM (LoRA)"),
    (6, "forecast.py",                     "Forecast all instruments"),
    (7, "export_to_mt5.py",               "Export forecasts to MT5"),
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full TimesFM → MT5 pipeline"
    )
    p.add_argument(
        "--from-step", type=int, default=1, metavar="N",
        help="Start from step N (1-7, default: 1)",
    )
    p.add_argument(
        "--only", type=int, nargs="+", metavar="N",
        help="Run only these step numbers (e.g. --only 5 6 7)",
    )
    p.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip step 5 (fine-tuning) and reuse existing adapter",
    )
    p.add_argument(
        "--skip-moon", action="store_true",
        help="Skip step 2 (moon phases don't change)",
    )
    return p.parse_args()


def run_step(step_num: int, script: str, description: str) -> bool:
    """Run a single pipeline step. Returns True on success."""
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        print(f"  ERROR: {script_path} not found!")
        return False

    print(f"\n{'='*60}")
    print(f"  Step {step_num}/7: {description}")
    print(f"  Script: {script}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPTS_DIR.parent),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main():
    args = parse_args()

    # Determine which steps to run
    if args.only:
        step_nums = set(args.only)
    else:
        step_nums = {s[0] for s in STEPS if s[0] >= args.from_step}

    if args.skip_finetune:
        step_nums.discard(5)
    if args.skip_moon:
        step_nums.discard(2)

    steps_to_run = [(n, s, d) for n, s, d in STEPS if n in step_nums]

    if not steps_to_run:
        print("No steps selected. Nothing to do.")
        return

    print("=" * 60)
    print("  TimesFM MT5 Pipeline")
    print("=" * 60)
    print(f"\nSteps to run: {', '.join(str(s[0]) for s in steps_to_run)}")
    for n, _, d in steps_to_run:
        print(f"  {n}. {d}")

    t_total = time.time()
    failed = []

    for step_num, script, description in steps_to_run:
        ok = run_step(step_num, script, description)
        if not ok:
            failed.append((step_num, description))
            print(f"\n  Pipeline stopped at step {step_num}.")
            print(f"  Fix the issue and resume with: python mt5/run_pipeline.py --from-step {step_num}")
            break

    total_elapsed = time.time() - t_total

    print(f"\n{'='*60}")
    if failed:
        print(f"  Pipeline FAILED at step {failed[0][0]}: {failed[0][1]}")
    else:
        print(f"  Pipeline completed successfully!")
        print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
        if 7 in step_nums:
            print(f"\n  Press R on your MT5 charts to reload the forecasts.")
    print("=" * 60)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
