from __future__ import annotations

from pathlib import Path
import pandas as pd

from ticket.build_ticket import build_all_tickets


def main() -> None:
    results_root = Path("results")

    date_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not date_dirs:
        raise ValueError("No results folders found")

    results_dir = date_dirs[-1]

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

    print(f"[TICKETS] Total rows: {len(df)}")
    print(f"[TICKETS] Eligible rows: {(df['is_eligible'] == 1).sum()}")

    tickets = build_all_tickets(df)

    tickets["safe"].to_csv(results_dir / "ticket_safe.csv", index=False)
    tickets["balanced"].to_csv(results_dir / "ticket_balanced.csv", index=False)
    tickets["lotto"].to_csv(results_dir / "ticket_lotto.csv", index=False)
    tickets["summary"].to_csv(results_dir / "ticket_summary.csv", index=False)

    print("[TICKETS] Saved:")
    print(f"  ticket_safe.csv      ({len(tickets['safe'])} legs)")
    print(f"  ticket_balanced.csv  ({len(tickets['balanced'])} legs)")
    print(f"  ticket_lotto.csv     ({len(tickets['lotto'])} legs)")
    print("  ticket_summary.csv")


if __name__ == "__main__":
    main()