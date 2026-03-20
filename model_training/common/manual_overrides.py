from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_omit_players(path: Path | str = "data/manual/omit_players.csv") -> set[str]:
    path = Path(path)

    if not path.exists():
        return set()

    df = pd.read_csv(path)

    if "player" not in df.columns:
        raise ValueError(f"{path} must contain column: player")

    players = set(df["player"].dropna().astype(str).str.strip())

    return players


def apply_omit_players(df: pd.DataFrame, omit_players: set[str]) -> pd.DataFrame:
    if df.empty or not omit_players:
        return df

    before = len(df)

    df = df[~df["player"].isin(omit_players)].copy()

    after = len(df)

    print(f"[OMIT] Removed {before - after} rows via manual player list")

    return df