# scripts/backtest_daily_pipeline.py

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from scripts.predict_pts import predict_pts
from scripts.predict_reb import predict_reb
from scripts.predict_ast import predict_ast
from scripts.predict_fg3 import predict_fg3

from ticket.score_legs import build_ranked_pool
from ticket.build_ticket import build_all_tickets


RESULTS_ROOT = Path("results_backtest")


def run_for_date(run_date: str):
    print(f"\n[DATE] {run_date}")

    out_dir = RESULTS_ROOT / run_date
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # STEP 1: PREDICT
    # -------------------------
    print("[STEP] Predicting stats...")

    df_pts = predict_pts(run_date=run_date)
    df_reb = predict_reb(run_date=run_date)
    df_ast = predict_ast(run_date=run_date)
    df_fg3 = predict_fg3(run_date=run_date)

    df = pd.concat([df_pts, df_reb, df_ast, df_fg3], ignore_index=True)

    # -------------------------
    # STEP 2: SCORE + RANK
    # -------------------------
    print("[STEP] Scoring legs...")

    ranked = build_ranked_pool(df)

    ranked_path = out_dir / "ranked_pool.csv"
    ranked.to_csv(ranked_path, index=False)
    print(f"[SAVED] {ranked_path}")

    # -------------------------
    # STEP 3: BUILD TICKETS
    # -------------------------
    print("[STEP] Building tickets...")

    tickets = build_all_tickets(
        ranked,
        output_dir=out_dir / "tickets",
    )

    # -------------------------
    # STEP 4: SAVE LEGS
    # -------------------------
    legs_path = out_dir / "ticket_legs.csv"
    ranked.to_csv(legs_path, index=False)

    print(f"[SAVED] {legs_path}")


def main():
    start = datetime(2025, 1, 1)
    end = datetime(2025, 1, 10)

    d = start
    while d <= end:
        try:
            run_for_date(d.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"[ERROR] Failed on {d.date()}: {e}")
        d += timedelta(days=1)


if __name__ == "__main__":
    main()