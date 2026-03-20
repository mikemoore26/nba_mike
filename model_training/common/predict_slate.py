from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from model_training.common.slate import make_results_dir
from model_training.common.today_row import slate_from_team_pairs
from model_training.utils.team_codes import norm_team


def get_run_date_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def safe_get_games_no_cache(schedule_dt: datetime) -> pd.DataFrame:
    """
    Fetch today's games without trusting stale local cache.
    """
    try:
        from nba_scraper.schedule import get_todays_games_cached

        cache_dir = Path("./data/cache")

        if cache_dir.exists():
            for p in cache_dir.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass

        df = get_todays_games_cached(
            cache_dir=cache_dir,
            game_date=schedule_dt.date(),
        )
        if df is None:
            return pd.DataFrame()
        return df.copy()

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, TimeoutError):
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def matchups_from_games_df(df_games: pd.DataFrame) -> list[tuple[str, str]]:
    if df_games.empty or not {"away_abbrev", "home_abbrev"}.issubset(df_games.columns):
        return []

    matchups: list[tuple[str, str]] = []
    for g in df_games.itertuples(index=False):
        away = norm_team(getattr(g, "away_abbrev", None))
        home = norm_team(getattr(g, "home_abbrev", None))
        if away and home and away != "NAN" and home != "NAN":
            matchups.append((away, home))
    return matchups


def resolve_matchups(
    *,
    schedule_dt: datetime,
    history_df: pd.DataFrame,
    away_team: str | None,
    home_team: str | None,
    feature_date: str | None,
) -> tuple[pd.DataFrame, str, str, Path]:
    """
    Returns:
      slate_df, run_date, history_cutoff_date, results_dir

    IMPORTANT:
    - slate_df.game_date is always run_date (today's prediction slate)
    - history_cutoff_date is metadata only
    """
    run_date = get_run_date_str()

    max_hist_date = pd.to_datetime(history_df["game_date"], errors="coerce").max()
    if pd.isna(max_hist_date):
        raise ValueError("history_df has no valid game_date values")

    history_cutoff_date = feature_date or max_hist_date.strftime("%Y-%m-%d")

    df_games = safe_get_games_no_cache(schedule_dt)
    matchups = matchups_from_games_df(df_games)
    schedule_source = "live"

    if not matchups:
        if not away_team or not home_team:
            raise ValueError(
                f"No valid live schedule found for run_date={run_date}. "
                "Pass away_team/home_team manually."
            )
        matchups = [(norm_team(away_team), norm_team(home_team))]
        schedule_source = "manual"

    # CRITICAL FIX:
    # use run_date for the actual prediction slate date
    slate_df = slate_from_team_pairs(
        game_date=run_date,
        matchups=matchups,
    ).copy()

    slate_df["_run_date"] = run_date
    slate_df["_history_cutoff_date"] = history_cutoff_date
    slate_df["_schedule_source"] = schedule_source

    results_dir = make_results_dir(run_date)
    return slate_df, run_date, history_cutoff_date, results_dir


def print_slate_debug(*, prefix: str, slate_df: pd.DataFrame, run_date: str, history_cutoff_date: str) -> None:
    schedule_source = slate_df["_schedule_source"].iloc[0] if "_schedule_source" in slate_df.columns else "unknown"

    print(f"[{prefix}] Using run_date = {run_date}")
    print(f"[{prefix}] Using history_cutoff_date = {history_cutoff_date}")
    print(f"[{prefix}] Schedule source = {schedule_source}")
    print("[INFO] Matchups used:")

    for _, row in slate_df[["team", "opp", "game_date"]].iterrows():
        print(f"  {row['team']} vs {row['opp']} | {row['game_date']}")