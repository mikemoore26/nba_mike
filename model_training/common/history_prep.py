from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def _coalesce_columns(df: pd.DataFrame, candidates: list[str], new_col: str) -> pd.DataFrame:
    """
    Create/overwrite `new_col` by taking the first non-null value from candidate columns.
    """
    out = df.copy()

    existing = [c for c in candidates if c in out.columns]
    if not existing:
        return out

    out[new_col] = out[existing].bfill(axis=1).iloc[:, 0]
    return out


def _to_minutes(series: pd.Series) -> pd.Series:
    """
    Convert minutes-like values into float minutes.

    Handles:
    - numeric minutes already
    - strings like '34:21'
    - messy strings -> NaN
    """
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s = s.astype(str).str.strip()

    # MM:SS format
    has_colon = s.str.contains(":", regex=False, na=False)
    out = pd.to_numeric(s, errors="coerce")

    if has_colon.any():
        parts = s[has_colon].str.split(":", n=1, expand=True)
        mm = pd.to_numeric(parts[0], errors="coerce")
        ss = pd.to_numeric(parts[1], errors="coerce")
        out.loc[has_colon] = mm + (ss / 60.0)

    return out


def _add_season_from_date(df: pd.DataFrame, date_col: str = "game_date") -> pd.DataFrame:
    """
    NBA season convention:
    - Oct/Nov/Dec belong to current year season start
    - Jan..Jun belong to previous year season start

    Example:
    2025-11-10 -> season 2025
    2026-02-14 -> season 2025
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")

    season = np.where(dt.dt.month >= 10, dt.dt.year, dt.dt.year - 1)
    out["season"] = pd.Series(season, index=out.index).astype("Int64")
    return out


def prepare_history_df(
    df: pd.DataFrame,
    *,
    norm_team_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """
    Standardize historical game logs into a canonical modeling frame.

    Output expectations:
    - one row per player-game
    - canonical date column: game_date
    - canonical stat columns:
        pts, reb, ast, fg3a, fg3m, fga, fgm, fta, ftm, tov, mp_minutes
    - canonical context columns:
        player, team, opp, season, is_home

    This function should do only deterministic prep.
    No rolling features. No train/valid logic. No leakage-sensitive aggregates.
    """
    out = df.copy()

    # ---------------------------------------------------------
    # Canonical dates
    # ---------------------------------------------------------
    out = _coalesce_columns(out, ["game_date", "date", "game_dt", "GAME_DATE"], "game_date")
    if "game_date" not in out.columns:
        raise ValueError("prepare_history_df: could not find a date column.")

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()

    # ---------------------------------------------------------
    # Canonical player/team/opponent
    # ---------------------------------------------------------
    out = _coalesce_columns(out, ["player", "player_name", "PLAYER_NAME", "name"], "player")
    out = _coalesce_columns(out, ["team", "team_abbr", "TEAM", "Tm"], "team")
    out = _coalesce_columns(out, ["opp", "opponent", "opp_team", "OPP"], "opp")

    required_id_cols = ["player", "team", "opp"]
    missing_ids = [c for c in required_id_cols if c not in out.columns]
    if missing_ids:
        raise ValueError(f"prepare_history_df: missing required identity columns: {missing_ids}")

    out["player"] = out["player"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["opp"] = out["opp"].astype(str).str.strip().str.upper()

    if norm_team_fn is not None:
        out["team"] = out["team"].map(lambda x: norm_team_fn(x) if pd.notna(x) else x)
        out["opp"] = out["opp"].map(lambda x: norm_team_fn(x) if pd.notna(x) else x)

    # ---------------------------------------------------------
    # Home/away normalization
    # ---------------------------------------------------------
    if "is_home" not in out.columns:
        if "home_game" in out.columns:
            out["is_home"] = pd.to_numeric(out["home_game"], errors="coerce")
        elif "HOME" in out.columns:
            out["is_home"] = pd.to_numeric(out["HOME"], errors="coerce")
        elif "home_away" in out.columns:
            # crude fallback
            tmp = out["home_away"].astype(str).str.upper().str.strip()
            out["is_home"] = np.where(tmp.isin(["HOME", "H", "VS"]), 1, 0)
        elif "matchup" in out.columns:
            # nba_api style: "BOS vs. NYK" home, "BOS @ NYK" away
            tmp = out["matchup"].astype(str)
            out["is_home"] = np.where(tmp.str.contains("vs\\.", case=False, na=False), 1, 0)
        else:
            out["is_home"] = np.nan

    out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce")
    out["is_home"] = out["is_home"].fillna(0).astype(int)

    # ---------------------------------------------------------
    # Canonical stat columns
    # ---------------------------------------------------------
    stat_aliases: dict[str, list[str]] = {
        "pts": ["pts", "PTS"],
        "reb": ["reb", "trb", "REB", "TRB"],
        "ast": ["ast", "AST"],
        "fg3a": ["fg3a", "FG3A", "3PA"],
        "fg3m": ["fg3m", "fg3", "FG3M", "FG3", "3P"],
        "fga": ["fga", "FGA"],
        "fgm": ["fgm", "FGM"],
        "fta": ["fta", "FTA"],
        "ftm": ["ftm", "FT", "FTM"],
        "tov": ["tov", "TOV", "to"],
        "usage": ["usage", "usg", "USG%", "usg_pct"],
        "starter_flag": ["starter_flag", "started", "is_starter"],
    }

    for canon_col, candidates in stat_aliases.items():
        out = _coalesce_columns(out, candidates, canon_col)

    # minutes
    if "mp_minutes" not in out.columns:
        if "mp" in out.columns:
            out["mp_minutes"] = _to_minutes(out["mp"])
        elif "minutes" in out.columns:
            out["mp_minutes"] = _to_minutes(out["minutes"])
        elif "min" in out.columns:
            out["mp_minutes"] = _to_minutes(out["min"])
        elif "MP" in out.columns:
            out["mp_minutes"] = _to_minutes(out["MP"])
        else:
            out["mp_minutes"] = np.nan

    # numeric cast
    numeric_cols = [
        "pts",
        "reb",
        "ast",
        "fg3a",
        "fg3m",
        "fga",
        "fgm",
        "fta",
        "ftm",
        "tov",
        "usage",
        "starter_flag",
        "mp_minutes",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # starter flag default
    if "starter_flag" not in out.columns:
        out["starter_flag"] = 0
    out["starter_flag"] = out["starter_flag"].fillna(0).astype(int)

    # ---------------------------------------------------------
    # Season
    # ---------------------------------------------------------
    if "season" not in out.columns:
        out = _add_season_from_date(out, date_col="game_date")
    else:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
        bad_season = out["season"].isna()
        if bad_season.any():
            out.loc[bad_season, :] = _add_season_from_date(out.loc[bad_season], date_col="game_date")

    # ---------------------------------------------------------
    # Cleaning + dedupe
    # ---------------------------------------------------------
    out = out.dropna(subset=["player", "team", "opp"]).copy()

    # guardrails
    if "mp_minutes" in out.columns:
        out.loc[out["mp_minutes"] < 0, "mp_minutes"] = np.nan
        out.loc[out["mp_minutes"] > 60, "mp_minutes"] = np.nan

    for col in ["pts", "reb", "ast", "fg3a", "fg3m", "fga", "fgm", "fta", "ftm", "tov"]:
        if col in out.columns:
            out.loc[out[col] < 0, col] = np.nan

    # remove exact duplicates first
    out = out.drop_duplicates().copy()

    # remove likely duplicated player-games
    dedupe_subset = [c for c in ["player", "game_date", "team", "opp"] if c in out.columns]
    if dedupe_subset:
        out = (
            out.sort_values(["player", "game_date", "team", "opp"], kind="mergesort")
            .drop_duplicates(subset=dedupe_subset, keep="last")
            .copy()
        )

    # stable ordering
    out = out.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    # legacy alias for compatibility
    out["date"] = out["game_date"]

    return out