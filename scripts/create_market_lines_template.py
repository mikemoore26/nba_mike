from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    results_root = Path("results")
    date_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not date_dirs:
        raise ValueError("No results folders found")

    results_dir = date_dirs[-1]
    print(f"[MARKET TEMPLATE] Using results_dir = {results_dir}")

    pred_files = [
        "pred_pts.csv",
        "pred_reb.csv",
        "pred_ast.csv",
        "pred_fg3.csv",
    ]

    dfs = []
    for f in pred_files:
        path = results_dir / f
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"[MARKET TEMPLATE] Missing prediction file: {f}")

    if not dfs:
        raise ValueError("No prediction files found")

    df = pd.concat(dfs, ignore_index=True)

    # only use eligible players by default
    if "is_eligible" in df.columns:
        df = df[df["is_eligible"] == 1].copy()

    # keep best-looking projection rows first
    df = df.sort_values(["minutes_proj", "pred_mean"], ascending=False).copy()

    # one row per player/stat
    df = df.drop_duplicates(subset=["player", "team", "opp", "stat"]).copy()

    out = df[["player", "team", "opp", "stat"]].copy()

    # blank columns for you to fill
    out["line"] = ""
    out["side"] = ""
    out["american_odds"] = ""
    out["book"] = "fanduel"

    out_path = results_dir / "market_lines_template.csv"
    out.to_csv(out_path, index=False)

    print(f"[MARKET TEMPLATE] Saved -> {out_path}")
    print("Fill in line / side / american_odds, then save a copy as market_lines.csv")


if __name__ == "__main__":
    main()