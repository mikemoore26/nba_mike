from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_training.common.market_math import add_market_edge_columns


def main() -> None:
    results_root = Path("results")
    date_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not date_dirs:
        raise ValueError("No results folders found")

    results_dir = date_dirs[-1]
    print(f"[MARKET] Using results_dir = {results_dir}")

    pred_files = [
        "pred_pts.csv",
        "pred_reb.csv",
        "pred_ast.csv",
        "pred_fg3.csv",
    ]

    pred_dfs = []
    for f in pred_files:
        path = results_dir / f
        if path.exists():
            pred_dfs.append(pd.read_csv(path))
        else:
            print(f"[MARKET] Missing prediction file: {f}")

    if not pred_dfs:
        raise ValueError("No prediction files found")

    pred_df = pd.concat(pred_dfs, ignore_index=True)

    market_path = results_dir / "market_lines.csv"
    if not market_path.exists():
        print(f"[MARKET] No market_lines.csv found at {market_path}")
        print("[MARKET] Skipping EV scoring.")
        return

    market_df = pd.read_csv(market_path)

    required_market_cols = [
        "player",
        "team",
        "opp",
        "stat",
        "line",
        "side",
        "american_odds",
    ]
    missing = [c for c in required_market_cols if c not in market_df.columns]
    if missing:
        raise ValueError(f"market_lines.csv missing required columns: {missing}")

    merge_cols = ["player", "team", "opp", "stat"]

    merged = market_df.merge(
        pred_df[
            [
                "game_date",
                "player",
                "team",
                "opp",
                "stat",
                "pred_mean",
                "baseline_mean",
                "delta_mean",
                "minutes_proj",
                "dist_name",
                "dispersion",
                "is_eligible",
                "eligibility_reason",
                "model_name",
                "model_version",
            ]
        ],
        on=merge_cols,
        how="left",
    )

    missing_preds = merged["pred_mean"].isna().sum()
    if missing_preds > 0:
        print(f"[MARKET] Warning: {missing_preds} market rows did not match prediction rows")

    merged = add_market_edge_columns(merged)

    merged = merged.sort_values(
        ["ev_per_unit", "edge", "model_prob"],
        ascending=False,
    ).reset_index(drop=True)

    out_path = results_dir / "scored_market_legs.csv"
    merged.to_csv(out_path, index=False)

    print(f"[MARKET] Saved -> {out_path}")
    print(
        merged[
            [
                "player",
                "team",
                "opp",
                "stat",
                "line",
                "side",
                "american_odds",
                "pred_mean",
                "model_prob",
                "implied_prob",
                "edge",
                "ev_per_unit",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()