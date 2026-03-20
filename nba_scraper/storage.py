# nba_scraper/storage.py
from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pandas as pd


CATEGORICAL_COLS = [
    "team",
    "opp",
    "home_away",
    "result",
    "gs",
]


def normalize_gamelogs_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # enforce season type BEFORE parquet write
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["season"])
        df["season"] = df["season"].astype("int32")

    # (keep your other coercions)
    ...
    return df

def gamelog_dataset_root(gamelog_dir: Path) -> Path:
    """
    Root directory for hive-partitioned parquet dataset:
      gamelogs_parquet/season=2026/part_*.parquet
    """
    return gamelog_dir / "gamelogs_parquet"


def write_gamelog_part(root: Path, season: int, df: pd.DataFrame) -> Path:
    """
    Append-only write: creates a new part file under season partition.
    """
    part_dir = root / f"season={season}"
    part_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"part_{stamp}_{uuid4().hex[:8]}.parquet"
    out_path = part_dir / fname

    df = normalize_gamelogs_df(df)
    df.to_parquet(out_path, index=False, compression="snappy")
    return out_path


def load_seen_keys_from_parquet_dataset(root: Path, season: int) -> set[tuple[str, int, str]]:
    """
    Loads dedupe keys (href, season, date) from existing parquet parts.
    This is much lighter than loading full gamelogs because we read only 3 columns.
    """
    part_dir = root / f"season={season}"
    if not part_dir.exists():
        return set()

    # pandas read_parquet supports directory reads when engine=pyarrow is installed
    df_keys = pd.read_parquet(part_dir, columns=["href", "season", "date"])
    df_keys = df_keys.dropna(subset=["href", "season", "date"])

    out: set[tuple[str, int, str]] = set()
    for href, s, d in df_keys.itertuples(index=False, name=None):
        out.add((str(href), int(s), str(d)))
    return out
