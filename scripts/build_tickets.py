# scripts/build_tickets.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ticket.pseudo_legs import expand_to_pseudo_legs
from ticket.score_legs import score_legs
from ticket.pool_postprocess import build_curated_scored_pool
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
    pred_all = pd.concat(dfs, ignore_index=True)
    pred_path = results_dir / "pred_all.csv"
    pred_all.to_csv(pred_path, index=False)
    print(f"[SAVED] pred_all -> {pred_path}")
    print(f"[INFO] Combined predictions: {len(pred_all)} rows")

    # -----------------------------
    # Expand pseudo legs
    # -----------------------------
    print("[INFO] Expanding pseudo legs...")
    pseudo_legs = expand_to_pseudo_legs(
        pred_all,
        line_offsets=(-1.0, 0.0, 1.0),
        min_prob=0.50,
        keep_both_sides=True,
    )

    pseudo_path = results_dir / "pseudo_legs.csv"
    pseudo_legs.to_csv(pseudo_path, index=False)
    print(f"[SAVED] pseudo_legs -> {pseudo_path}")

    if pseudo_legs.empty:
        raise ValueError("No pseudo legs generated from prediction files.")

    # -----------------------------
    # Score legs
    # -----------------------------
    print("[INFO] Scoring legs...")
    scored = score_legs(pseudo_legs)

    scored_raw_path = results_dir / "scored_legs_raw.csv"
    scored.to_csv(scored_raw_path, index=False)
    print(f"[SAVED] scored_legs_raw -> {scored_raw_path}")

    if scored.empty:
        raise ValueError("No scored legs produced.")

    # -----------------------------
    # Curate scored pool
    # -----------------------------
    print("[INFO] Curating scored pool...")
    scored_curated = build_curated_scored_pool(
        scored,
        score_col="score_balanced",
        min_p_hit=0.55,
        min_edge_raw=0.50,
    )

    curated_path = results_dir / "scored_legs_curated.csv"
    scored_curated.to_csv(curated_path, index=False)
    print(f"[SAVED] scored_legs_curated -> {curated_path}")

    if scored_curated.empty:
        raise ValueError("No curated scored legs available for ticket building.")

    # -----------------------------
    # Build tickets
    # -----------------------------
    print("[INFO] Building tickets...")
    tickets = build_all_tickets(scored_curated)

    tickets_dir = results_dir / "tickets"
    tickets_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(tickets, dict):
        out_path = tickets_dir / "tickets.csv"
        tickets.to_csv(out_path, index=False)
        print(f"[SAVED] tickets -> {out_path}")
        print("[DONE] Ticket building complete.")
        return

    ranked_pool = tickets.get("ranked_pool")
    if isinstance(ranked_pool, pd.DataFrame) and not ranked_pool.empty:
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