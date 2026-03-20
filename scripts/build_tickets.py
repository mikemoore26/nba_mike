from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd

from ticket.build_ticket import build_all_tickets
from model_training.common.manual_overrides import load_omit_players, apply_omit_players


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

    dfs = []
    for f in files:
        path = results_dir / f
        if not path.exists():
            print(f"[WARN] Missing {f}")
            continue
        dfs.append(pd.read_csv(path))

    if not dfs:
        raise ValueError("No prediction files found")

    df = pd.concat(dfs, ignore_index=True)

    print(f"[TICKETS] Total rows before omit filter: {len(df)}")
    print(f"[TICKETS] Eligible rows before omit filter: {(df['is_eligible'] == 1).sum()}")

    # -----------------------------
    # APPLY MANUAL PLAYER OMITS
    # -----------------------------
    omit_players = load_omit_players("data/manual/omit_players.csv")
    print(f"[TICKETS] Loaded omit_players list with {len(omit_players)} players")
    df = apply_omit_players(df, omit_players)

    print(f"[TICKETS] Total rows after omit filter: {len(df)}")
    print(f"[TICKETS] Eligible rows after omit filter: {(df['is_eligible'] == 1).sum()}")

    tickets = build_all_tickets(df)

    tickets["ranked_pool"].to_csv(results_dir / "ranked_projection_pool.csv", index=False)
    tickets["safe"].to_csv(results_dir / "ticket_safe.csv", index=False)
    tickets["balanced"].to_csv(results_dir / "ticket_balanced.csv", index=False)
    tickets["lotto"].to_csv(results_dir / "ticket_lotto.csv", index=False)
    tickets["summary"].to_csv(results_dir / "ticket_summary.csv", index=False)

    print("[TICKETS] Saved:")
    print("  ranked_projection_pool.csv")
    print(f"  ticket_safe.csv      ({len(tickets['safe'])} legs)")
    print(f"  ticket_balanced.csv  ({len(tickets['balanced'])} legs)")
    print(f"  ticket_lotto.csv     ({len(tickets['lotto'])} legs)")
    print("  ticket_summary.csv")


if __name__ == "__main__":
    main()