from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from model_training.common.slate import make_results_dir, resolve_slate_date
from model_training.common.today_row import slate_from_team_pairs
from model_training.utils.team_codes import norm_team


def safe_get_games(schedule_dt: datetime) -> pd.DataFrame:
    try:
        from nba_scraper.schedule import get_todays_games_cached

        return get_todays_games_cached(
            cache_dir=Path("./data/cache"),
            game_date=schedule_dt.date(),
        )
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


def load_local_slate_csv(path: Path = Path("./data/cache/todays_games.csv")) -> list[tuple[str, str]]:
    if not path.exists():
        return []

    df = pd.read_csv(path)
    if not {"away_abbrev", "home_abbrev"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: away_abbrev, home_abbrev")

    pairs: list[tuple[str, str]] = []
    for r in df.itertuples(index=False):
        away = norm_team(getattr(r, "away_abbrev"))
        home = norm_team(getattr(r, "home_abbrev"))
        if away and home and away != "NAN" and home != "NAN":
            pairs.append((away, home))
    return pairs


def choose_game_date(
    *,
    history_df: pd.DataFrame,
    schedule_date: str,
    game_date: str | None,
) -> str:
    max_hist_date = pd.to_datetime(history_df["game_date"], errors="coerce").max()
    model_date = (
        schedule_date
        if pd.to_datetime(schedule_date) <= (max_hist_date + pd.Timedelta(days=1))
        else max_hist_date.strftime("%Y-%m-%d")
    )
    return game_date or model_date


def resolve_matchups(
    *,
    schedule_dt: datetime,
    schedule_date: str,
    history_df: pd.DataFrame,
    away_team: str | None,
    home_team: str | None,
    game_date: str | None,
) -> tuple[pd.DataFrame, str, Path]:
    """
    Returns:
      slate_df, slate_date, results_dir
    """
    resolved_game_date = choose_game_date(
        history_df=history_df,
        schedule_date=schedule_date,
        game_date=game_date,
    )

    df_games = safe_get_games(schedule_dt)
    matchups = matchups_from_games_df(df_games)

    if not matchups:
        matchups = load_local_slate_csv()

    if not matchups:
        if not away_team or not home_team:
            raise ValueError("No schedule/local/manual matchup available.")
        matchups = [(norm_team(away_team), norm_team(home_team))]

    slate_df = slate_from_team_pairs(
        game_date=resolved_game_date,
        matchups=matchups,
    )

    slate_date = resolve_slate_date(slate_df)
    results_dir = make_results_dir(slate_date)

    return slate_df, slate_date, results_dir


def print_slate_debug(*, prefix: str, slate_df: pd.DataFrame, slate_date: str) -> None:
    print(f"[{prefix}] Using slate_date = {slate_date}")
    print("[INFO] Matchups used:")
    for _, row in slate_df.iterrows():
        print(f"  {row['team']} vs {row['opp']} | {row['game_date']}")