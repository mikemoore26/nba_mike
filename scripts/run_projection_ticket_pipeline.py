from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


PIPELINE_STEPS = [
    ("Predict all stats", "scripts.predict_all"),
    ("Build projection board", "scripts.build_projection_board"),
    ("Rank projection board", "scripts.rank_projection_board"),
    ("Build projection legs", "scripts.build_projection_legs"),
    ("Build scored projection legs", "scripts.build_scored_projection_legs"),
    ("Build projection tickets", "scripts.build_projection_tickets"),
    ("Export ticket report", "scripts.export_ticket_report"),
]


def _run_module(
    module_name: str,
    *,
    python_exe: str,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [python_exe, "-m", module_name]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n[RUN] {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")

    if result.stderr:
        print("[STDERR]")
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )


def _results_dir(run_date: str) -> Path:
    return Path("results") / run_date


def _print_output_summary(run_date: str) -> None:
    results_dir = _results_dir(run_date)
    tickets_dir = results_dir / "tickets"

    print("\n" + "=" * 72)
    print("[PIPELINE COMPLETE]")
    print(f"Results directory: {results_dir}")

    expected_files = [
        results_dir / "pred_all.csv",
        results_dir / "projection_board.csv",
        results_dir / "projection_board_ranked.csv",
        results_dir / "projection_legs.csv",
        results_dir / "projection_legs_scored.csv",
        tickets_dir / "ticket_safe.csv",
        tickets_dir / "ticket_balanced.csv",
        tickets_dir / "ticket_lotto.csv",
    ]

    for path in expected_files:
        status = "FOUND" if path.exists() else "MISSING"
        print(f"[{status}] {path}")

    print("=" * 72)


def run_pipeline(
    *,
    run_date: str | None = None,
    start_at: str | None = None,
    stop_after: str | None = None,
    python_exe: str | None = None,
) -> None:
    if run_date is None:
        run_date = datetime.today().strftime("%Y-%m-%d")

    if python_exe is None:
        python_exe = sys.executable

    step_names = [module for _, module in PIPELINE_STEPS]

    if start_at is not None and start_at not in step_names:
        raise ValueError(
            f"Invalid --start-at '{start_at}'. "
            f"Choose from: {step_names}"
        )

    if stop_after is not None and stop_after not in step_names:
        raise ValueError(
            f"Invalid --stop-after '{stop_after}'. "
            f"Choose from: {step_names}"
        )

    started = start_at is None
    pipeline_start = time.time()

    print("=" * 72)
    print("[NBA PROJECTION TICKET PIPELINE]")
    print(f"Run date   : {run_date}")
    print(f"Python     : {python_exe}")
    print(f"Start at   : {start_at or 'beginning'}")
    print(f"Stop after : {stop_after or 'end'}")
    print("=" * 72)

    for label, module_name in PIPELINE_STEPS:
        if not started:
            if module_name == start_at:
                started = True
            else:
                continue

        step_start = time.time()
        print(f"\n[STEP] {label}")

        try:
            _run_module(module_name, python_exe=python_exe)
        except subprocess.CalledProcessError as exc:
            elapsed = time.time() - step_start
            print(f"\n[FAILED] {label}")
            print(f"[FAILED] Module   : {module_name}")
            print(f"[FAILED] Exit code: {exc.returncode}")
            print(f"[FAILED] Elapsed  : {elapsed:.2f}s")

            if exc.stderr:
                print("\n[FAILED STDERR SUMMARY]")
                print(exc.stderr, end="" if exc.stderr.endswith("\n") else "\n")

            raise

        elapsed = time.time() - step_start
        print(f"[DONE] {label} ({elapsed:.2f}s)")

        if stop_after is not None and module_name == stop_after:
            break

    total_elapsed = time.time() - pipeline_start
    print(f"\n[TOTAL ELAPSED] {total_elapsed:.2f}s")

    _print_output_summary(run_date)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the daily NBA projection-to-ticket pipeline."
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Run date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--start-at",
        type=str,
        default=None,
        help="Module name to start at, e.g. scripts.build_projection_legs",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        help="Module name to stop after, e.g. scripts.build_projection_tickets",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=None,
        help="Python executable to use. Defaults to current interpreter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        run_date=args.run_date,
        start_at=args.start_at,
        stop_after=args.stop_after,
        python_exe=args.python_exe,
    )


if __name__ == "__main__":
    main()