from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ticket.build_ticket import build_all_tickets


def main() -> None:
    run_date = datetime.today().strftime("%Y-%m-%d")
    results_dir = Path("results") / run_date

    if not results_dir.exists():
        raise ValueError(f"No results folder for today: {results_dir}")

    print(f"[TICKETS] Using results_dir = {results_dir}")

    files = [
        "pred_pts.csv",
        "pred_reb.csv",
        "pred_ast.csv",
        "pred_fg3.csv",
    ]

    dfs: list[pd.DataFrame] = []

    for f in files:
        path = results_dir / f
        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"[WARN] Empty file: {path}")
            continue

        dfs.append(df)

    if not dfs:
        raise ValueError("No prediction files found to build tickets.")

    # -----------------------------
    # Combine predictions
    # -----------------------------
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Combined predictions: {len(df_all)} rows")

    # -----------------------------
    # Build tickets
    # NOTE:
    # build_all_tickets() already handles:
    #   - schema normalization
    #   - pseudo-line creation if needed
    #   - score_legs(...)
    #   - rank_projection_pool(...)
    #   - safe / balanced / lotto ticket creation
    # So do NOT pre-rank here.
    # -----------------------------
    print("[INFO] Building tickets...")
    tickets = build_all_tickets(df_all)

    tickets_dir = results_dir / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(tickets, dict):
        out_path = tickets_dir / "tickets.csv"
        tickets.to_csv(out_path, index=False)
        print(f"[SAVED] tickets -> {out_path}")
        print("[DONE] Ticket building complete.")
        return

    # Save ranked pool for debugging
    ranked_pool = tickets.get("ranked_pool")
    if ranked_pool is not None and not ranked_pool.empty:
        ranked_path = results_dir / "ranked_pool.csv"
        ranked_pool.to_csv(ranked_path, index=False)
        print(f"[SAVED] ranked_pool -> {ranked_path}")
    else:
        print("[WARN] ranked_pool is empty")

    file_map = {
        "safe": "ticket_safe.csv",
        "balanced": "ticket_balanced.csv",
        "lotto": "ticket_lotto.csv",
        "summary": "ticket_summary.csv",
    }

    for key, filename in file_map.items():
        df = tickets.get(key)
        if df is None or df.empty:
            print(f"[WARN] Empty ticket set: {key}")
            continue

        out_path = tickets_dir / filename
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {key} -> {out_path}")

    print("[DONE] Ticket building complete.")


if __name__ == "__main__":
    main()