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


def load_local_slate_csv(path: Path = Path("./data/cache/todays_games.csv")) -> tuple[list[tuple[str, str]], str | None]:
    """
    Returns:
      (matchups, slate_date_from_file)

    Expected columns:
      away_abbrev, home_abbrev
    Optional recommended column:
      game_date
    """
    if not path.exists():
        return [], None

    df = pd.read_csv(path)
    if not {"away_abbrev", "home_abbrev"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: away_abbrev, home_abbrev")

    pairs: list[tuple[str, str]] = []
    for r in df.itertuples(index=False):
        away = norm_team(getattr(r, "away_abbrev"))
        home = norm_team(getattr(r, "home_abbrev"))
        if away and home and away != "NAN" and home != "NAN":
            pairs.append((away, home))

    slate_date_from_file = None
    if "game_date" in df.columns and df["game_date"].notna().any():
        slate_date_from_file = pd.to_datetime(df["game_date"].iloc[0]).strftime("%Y-%m-%d")

    return pairs, slate_date_from_file


def choose_feature_date(
    *,
    history_df: pd.DataFrame,
    run_date: str,
    feature_date: str | None,
) -> str:
    """
    This controls the date used for feature construction / player recency checks.
    It should never exceed the latest historical date + 1 day.
    """
    max_hist_date = pd.to_datetime(history_df["game_date"], errors="coerce").max()
    if pd.isna(max_hist_date):
        raise ValueError("history_df has no valid game_date values")

    fallback_feature_date = (
        run_date
        if pd.to_datetime(run_date) <= (max_hist_date + pd.Timedelta(days=1))
        else max_hist_date.strftime("%Y-%m-%d")
    )
    return feature_date or fallback_feature_date


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
      slate_df, run_date, resolved_feature_date, results_dir

    run_date:
      today's date; used for output folder

    resolved_feature_date:
      date used inside slate_df/game rows for building features safely
    """
    run_date = get_run_date_str()
    resolved_feature_date = choose_feature_date(
        history_df=history_df,
        run_date=run_date,
        feature_date=feature_date,
    )

    # 1) live schedule first
    df_games = safe_get_games(schedule_dt)
    matchups = matchups_from_games_df(df_games)
    schedule_source = "live"

    # 2) local fallback
    if not matchups:
        matchups, local_slate_date = load_local_slate_csv()
        schedule_source = "local_csv"

        # hard fail on stale local slate
        if matchups and local_slate_date is not None and local_slate_date != run_date:
            raise ValueError(
                f"Local fallback slate is stale: file game_date={local_slate_date}, run_date={run_date}. "
                f"Refresh data/cache/todays_games.csv before running predictions."
            )

    # 3) manual fallback
    if not matchups:
        if not away_team or not home_team:
            raise ValueError(
                f"No live schedule found for run_date={run_date}, "
                "no valid local fallback slate, and no manual matchup provided."
            )
        matchups = [(norm_team(away_team), norm_team(home_team))]
        schedule_source = "manual"

    slate_df = slate_from_team_pairs(
        game_date=resolved_feature_date,
        matchups=matchups,
    )

    results_dir = make_results_dir(run_date)

    # attach lightweight metadata columns for debugging if you want
    slate_df = slate_df.copy()
    slate_df["_run_date"] = run_date
    slate_df["_feature_date"] = resolved_feature_date
    slate_df["_schedule_source"] = schedule_source

    return slate_df, run_date, resolved_feature_date, results_dir


def print_slate_debug(*, prefix: str, slate_df: pd.DataFrame, run_date: str, feature_date: str) -> None:
    schedule_source = slate_df["_schedule_source"].iloc[0] if "_schedule_source" in slate_df.columns else "unknown"

    print(f"[{prefix}] Using run_date = {run_date}")
    print(f"[{prefix}] Using feature_date = {feature_date}")
    print(f"[{prefix}] Schedule source = {schedule_source}")
    print("[INFO] Matchups used:")

    cols = ["team", "opp", "game_date"]
    for _, row in slate_df[cols].iterrows():
        print(f"  {row['team']} vs {row['opp']} | {row['game_date']}")