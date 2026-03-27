from __future__ import annotations

import pandas as pd

from scripts.predict_ast import _load_history_df, _build_historical_slate_from_gamelogs
from model_training.common.today_row import (
    _canon_hist_df,
    _latest_pre_date_player_rows,
    build_today_rows_v2,
)


def main(run_date: str = "2025-01-01") -> None:
    history_df = _load_history_df(rebuild_history=False)

    slate_df = _build_historical_slate_from_gamelogs(
        history_df=history_df,
        run_date=run_date,
    )

    history_df_pre = history_df.loc[
        pd.to_datetime(history_df["game_date"], errors="coerce")
        < pd.to_datetime(run_date)
    ].copy()

    print("\n=== RAW HISTORY PRE COLUMNS ===")
    print(sorted(history_df_pre.columns.tolist()))

    raw_cols = [c for c in ["player", "team", "game_date", "mp", "mp_minutes"] if c in history_df_pre.columns]
    print("\n=== RAW SAMPLE ===")
    print(history_df_pre[raw_cols].head(30).to_string(index=False))

    if "mp" in history_df_pre.columns:
        print("\n=== RAW mp VALUE COUNTS (top 25) ===")
        print(history_df_pre["mp"].astype(str).value_counts(dropna=False).head(25).to_string())

    # ---------------------------------------------------------
    # Canonicalized / repaired version
    # ---------------------------------------------------------
    hist_fixed = _canon_hist_df(history_df_pre)

    fixed_cols = [c for c in ["player", "team", "game_date", "mp", "mp_minutes"] if c in hist_fixed.columns]

    print("\n=== FIXED SAMPLE ===")
    print(hist_fixed[fixed_cols].head(30).to_string(index=False))

    if "mp_minutes" in hist_fixed.columns:
        mpm = pd.to_numeric(hist_fixed["mp_minutes"], errors="coerce")
        print("\n=== FIXED mp_minutes SUMMARY ===")
        print(mpm.describe().to_string())

        print("\n=== FIXED mp_minutes > 0 RATE ===")
        print((mpm > 0).mean())

        print("\n=== FIXED mp_minutes VALUE COUNTS (top 25) ===")
        print(mpm.value_counts(dropna=False).head(25).to_string())

    # ---------------------------------------------------------
    # Latest valid rows before date
    # ---------------------------------------------------------
    latest = _latest_pre_date_player_rows(hist_fixed, pd.Timestamp(run_date))

    latest_cols = [c for c in ["player", "team", "game_date", "mp", "mp_minutes"] if c in latest.columns]

    print("\n=== LATEST PRE-DATE VALID SAMPLE ===")
    if latest.empty:
        print("latest is empty")
    else:
        print(latest[latest_cols].head(30).to_string(index=False))

        print("\n=== LATEST COUNTS BY TEAM (top 30) ===")
        print(latest["team"].value_counts().head(30).to_string())

        print("\n=== LATEST mp_minutes SUMMARY ===")
        print(pd.to_numeric(latest["mp_minutes"], errors="coerce").describe().to_string())

    # ---------------------------------------------------------
    # Full today rows build
    # ---------------------------------------------------------
    print("\n=== SLATE SAMPLE ===")
    print(slate_df.head(20).to_string(index=False))

    today_df = build_today_rows_v2(
        df_hist=history_df_pre,
        slate_df=slate_df,
        min_games_required=3,
        active_within_days=21,
        min_minutes_threshold=8.0,
        max_players_per_team=12,
        error_on_empty=True,
    )

    print("\n=== TODAY_DF COLUMNS ===")
    print(sorted(today_df.columns.tolist()))

    today_cols = [c for c in ["player", "team", "opp", "game_date", "mp", "mp_minutes", "minutes_filter_value"] if c in today_df.columns]

    print("\n=== TODAY_DF SAMPLE ===")
    print(today_df[today_cols].head(40).to_string(index=False))

    if "mp_minutes" in today_df.columns:
        print("\n=== TODAY_DF mp_minutes SUMMARY ===")
        print(pd.to_numeric(today_df["mp_minutes"], errors="coerce").describe().to_string())

    print(f"\nrows in today_df = {len(today_df)}")


if __name__ == "__main__":
    main()