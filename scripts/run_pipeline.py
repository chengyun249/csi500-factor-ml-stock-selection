#!/usr/bin/env python
"""Run the reproducible local research pipeline in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_STAGES = [
    "02_build_factor_panel.py",
    "03_factor_ic_analysis.py",
    "05_train_baseline_models.py",
    "06_portfolio_backtest.py",
    "07_robustness_checks.py",
]
FULL_ONLY_STAGES = [
    "08_build_industry_neutral_panel.py",
    "09_train_industry_neutral_models.py",
    "10_backtest_industry_neutral.py",
    "13_add_financial_factors.py",
    "14_train_models_with_finance.py",
    "15_backtest_models_with_finance.py",
    "11_update_benchmark_total_return.py",
    "16_collect_final_results.py",
    "17_generate_readme_and_report.py",
    "18_strategy_acceptance_diagnostics.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["core", "full"], default="core")
    parser.add_argument("--from-stage", help="Start at this numbered script, e.g. 06_portfolio_backtest.py")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = CORE_STAGES + (FULL_ONLY_STAGES if args.profile == "full" else [])
    if args.from_stage:
        if args.from_stage not in stages:
            raise ValueError(f"unknown stage {args.from_stage}; choices={stages}")
        stages = stages[stages.index(args.from_stage) :]

    for stage in stages:
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / stage)]
        print(f"\n>>> {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
