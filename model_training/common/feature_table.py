# model_training/common/feature_table.py
from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd


def load_gamelogs(src: Union[pd.DataFrame, str, Path], *, date_col: str = "date") -> pd.DataFrame:
    """
    Loads gamelogs from either:
      - a pandas DataFrame (returned as a cleaned copy)
      - a csv path (str/Path)

    Canonicalizes:
      - creates/ensures `game_date` datetime
      - keeps legacy `date` synced
      - stable sort by player, game_date
    """
    if isinstance(src, pd.DataFrame):
        df = src.copy()
    elif isinstance(src, (str, Path)):
        df = pd.read_csv(Path(src), low_memory=False)
    else:
        raise TypeError(f"load_gamelogs expects DataFrame or path. Got: {type(src)}")

    # Accept either date or game_date
    if "game_date" not in df.columns:
        if date_col in df.columns:
            df["game_date"] = df[date_col]
        else:
            raise ValueError(f"Missing `game_date` and `{date_col}` columns in gamelogs.")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()

    # Keep legacy alias in sync for old code
    df["date"] = df["game_date"]

    # Coerce id columns
    for c in ["player", "team", "opp"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    if "is_home" in df.columns:
        df["is_home"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int)

    # Stable sort (critical for rolling/expanding correctness)
    if "player" in df.columns:
        df = df.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)
    else:
        df = df.sort_values(["game_date"], kind="mergesort").reset_index(drop=True)

    return df


def build_feature_table(
    *,
    history_df: Union[pd.DataFrame, str, Path],
    today_df: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """
    Common pipeline:
      1) load/clean history + today
      2) concat
      3) compute games_played_prior on FULL combined
      4) run feature_builder once (must be no-leak inside)
      5) return X_today aligned to today_df

    Returns: X_today (DataFrame) same row-order/length as today_df.
    """
    history = load_gamelogs(history_df)
    today = load_gamelogs(today_df)

    combined = pd.concat([history, today], ignore_index=True)
    combined = combined.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    # leakage-safe prior game count (for gating)
    combined["games_played_prior"] = combined.groupby("player").cumcount()

    # Build features ONCE (no-leak should be enforced inside builder)
    combined = feature_builder(combined)

    # Slice today rows
    X_today = combined.tail(len(today)).reset_index(drop=True)
    return X_today