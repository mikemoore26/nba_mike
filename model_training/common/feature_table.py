from __future__ import annotations

from pathlib import Path
import pandas as pd


RAW_DATE_COL = "date"
CANON_DATE_COL = "game_date"

ID_COLS = ["game_date", "season", "player", "team", "opp", "is_home"]

NUMERIC_COLS = [
    "season",
    "mp_minutes",
    "fg", "fga",
    "fg3", "fg3a",
    "ft", "fta",
    "orb", "drb", "trb",
    "ast", "stl", "blk",
    "tov", "pf",
    "pts",
    "usage",
]

def load_gamelogs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize naming
    if RAW_DATE_COL in df.columns and CANON_DATE_COL not in df.columns:
        df = df.rename(columns={RAW_DATE_COL: CANON_DATE_COL})

    # types
    df[CANON_DATE_COL] = pd.to_datetime(df[CANON_DATE_COL], errors="coerce")

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # derived common aliases
    if "reb" not in df.columns and "trb" in df.columns:
        df["reb"] = df["trb"]

    # stable ordering (critical for leakage-safe rolling)
    df = df.sort_values(["player", CANON_DATE_COL], kind="mergesort").reset_index(drop=True)
    return df


def build_feature_table(df_gamelogs: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    """
    Wide table builder (IDs + raw stats now; engineered features later).
    mode: "train" | "predict"
    """
    if mode not in {"train", "predict"}:
        raise ValueError(f"mode must be 'train' or 'predict', got {mode!r}")

    df = df_gamelogs.copy()

    required = {"game_date", "season", "player", "team", "opp", "is_home"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gamelogs missing required cols: {sorted(missing)}")

    # leave all columns for now; trimming happens per-model using feature lists
    return df