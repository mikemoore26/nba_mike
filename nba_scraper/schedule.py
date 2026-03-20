# nba_scraper/schedule.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2


@dataclass(frozen=True)
class Game:
    game_id: str
    game_date: str  # YYYY-MM-DD
    away_team_id: int
    home_team_id: int
    away_abbrev: str
    home_abbrev: str
    status_text: str  # e.g. "7:30 pm ET", "Final"


def _retry(fn, *, retries: int = 4, base_delay: float = 1.5):
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == retries:
                raise
            sleep_s = (base_delay * attempt) + (0.25 * (attempt % 3))
            time.sleep(sleep_s)


def fetch_scoreboard_games(game_date: Date | None = None) -> pd.DataFrame:
    """
    Returns today's NBA schedule (and game_ids) using NBA Stats scoreboardv2.
    """
    d = game_date or Date.today()
    game_date_str = d.strftime("%Y-%m-%d")

    def _call():
        ep = scoreboardv2.ScoreboardV2(
            game_date=game_date_str,
            day_offset=0,
            league_id="00",
        )
        # 2 key frames: GameHeader (game ids + teams) and LineScore (abbrevs + status)
        gh = ep.game_header.get_data_frame()
        ls = ep.line_score.get_data_frame()
        return gh, ls

    gh, ls = _retry(_call, retries=4, base_delay=2.0)

    if gh.empty:
        # no games today (ASB, offseason, etc.)
        return pd.DataFrame(columns=[
            "game_id","game_date","away_team_id","home_team_id",
            "away_abbrev","home_abbrev","status_text"
        ])

    # GameHeader has GAME_ID, GAME_DATE_EST, HOME_TEAM_ID, VISITOR_TEAM_ID
    # LineScore has GAME_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_CITY_NAME, etc.
    # We pivot abbrevs by TEAM_ID for each GAME_ID.
    ls_small = ls[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]].copy()

    away = (
        gh[["GAME_ID", "VISITOR_TEAM_ID"]]
        .merge(ls_small, left_on=["GAME_ID", "VISITOR_TEAM_ID"], right_on=["GAME_ID", "TEAM_ID"], how="left")
        .rename(columns={"TEAM_ABBREVIATION": "away_abbrev", "VISITOR_TEAM_ID": "away_team_id"})
        .drop(columns=["TEAM_ID"])
    )

    home = (
        gh[["GAME_ID", "HOME_TEAM_ID"]]
        .merge(ls_small, left_on=["GAME_ID", "HOME_TEAM_ID"], right_on=["GAME_ID", "TEAM_ID"], how="left")
        .rename(columns={"TEAM_ABBREVIATION": "home_abbrev", "HOME_TEAM_ID": "home_team_id"})
        .drop(columns=["TEAM_ID"])
    )

    out = (
        gh[["GAME_ID", "GAME_DATE_EST", "GAME_STATUS_TEXT"]]
        .merge(away, on="GAME_ID", how="left")
        .merge(home, on="GAME_ID", how="left")
        .rename(columns={
            "GAME_ID": "game_id",
            "GAME_DATE_EST": "game_date",
            "GAME_STATUS_TEXT": "status_text",
        })
        .sort_values(["game_date", "game_id"], kind="stable")
        .reset_index(drop=True)
    )

    # normalize date string
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return out[[
        "game_id", "game_date",
        "away_team_id", "home_team_id",
        "away_abbrev", "home_abbrev",
        "status_text"
    ]]


def get_todays_games_cached(
    *,
    cache_dir: Path,
    game_date: Date | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Cache by date to avoid repeated requests.
    """
    d = game_date or Date.today()
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"scoreboard_{d.isoformat()}.json"

    if p.exists() and not refresh:
        with p.open("r") as f:
            return pd.DataFrame(json.load(f))

    df = fetch_scoreboard_games(d)
    with p.open("w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    return df
