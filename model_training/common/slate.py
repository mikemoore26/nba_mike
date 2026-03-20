from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def resolve_slate_date(slate_df: pd.DataFrame) -> str:
    """
    Extract canonical slate date from slate_df.
    Assumes all rows belong to same slate.
    """
    if "game_date" not in slate_df.columns:
        raise ValueError("slate_df missing game_date")

    if slate_df.empty:
        raise ValueError("slate_df is empty")

    slate_date = pd.to_datetime(slate_df["game_date"].iloc[0]).strftime("%Y-%m-%d")
    return slate_date


def make_results_dir(slate_date: str) -> Path:
    """
    Returns results/{slate_date} and ensures it exists.
    """
    out = Path("results") / slate_date
    out.mkdir(parents=True, exist_ok=True)
    return out


def latest_results_dir() -> Tuple[str, Path]:
    """
    Finds latest results folder (used by ticket + report scripts).
    """
    root = Path("results")

    if not root.exists():
        raise ValueError("results directory does not exist")

    dates = sorted([p.name for p in root.iterdir() if p.is_dir()])

    if not dates:
        raise ValueError("No result folders found")

    latest = dates[-1]
    return latest, root / latest