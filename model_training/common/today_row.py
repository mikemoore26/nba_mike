from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_SLATE_COLS = ["game_date", "team", "opp", "is_home"]
REQUIRED_HIST_COLS = ["game_date", "player", "team", "opp"]


@dataclass(frozen=True)
class TodayRowConfig:
    """
    Generic candidate filter config for building pregame rows.
    These thresholds are intentionally conservative for betting use.
    """
    min_games_required: int = 3
    active_within_days: int | None = 21
    min_minutes_threshold: float = 8.0
    max_players_per_team: int = 15
    error_on_empty: bool = True


def _validate_slate_df(slate_df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_SLATE_COLS if c not in slate_df.columns]
    if missing:
        raise ValueError(f"slate_df missing required columns: {missing}")


def _validate_hist_df(df_hist: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_HIST_COLS if c not in df_hist.columns]
    if missing:
        raise ValueError(f"df_hist missing required columns: {missing}")


def _canon_slate_df(slate_df: pd.DataFrame) -> pd.DataFrame:
    out = slate_df.copy()
    _validate_slate_df(out)

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()

    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["opp"] = out["opp"].astype(str).str.strip().str.upper()
    out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(int)

    out = (
        out.sort_values(["game_date", "team", "opp"], kind="mergesort")
        .drop_duplicates(subset=["game_date", "team"], keep="last")
        .reset_index(drop=True)
    )

    return out


def _canon_hist_df(df_hist: pd.DataFrame) -> pd.DataFrame:
    out = df_hist.copy()
    _validate_hist_df(out)

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()

    out["player"] = out["player"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["opp"] = out["opp"].astype(str).str.strip().str.upper()

    if "mp_minutes" in out.columns:
        out["mp_minutes"] = pd.to_numeric(out["mp_minutes"], errors="coerce")

    return out


def _add_player_history_summary(df_hist: pd.DataFrame) -> pd.DataFrame:
    out = df_hist.copy()
    out = out.sort_values(["player", "game_date"], kind="mergesort").copy()

    g = out.groupby("player", sort=False)

    out["games_played_career_to_date"] = g.cumcount() + 1
    out["last_game_date"] = g["game_date"].transform("max")

    if "mp_minutes" in out.columns:
        out["min_rolling_3_proxy"] = (
            g["mp_minutes"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)
        )
        out["min_rolling_5_proxy"] = (
            g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
        )
    else:
        out["min_rolling_3_proxy"] = np.nan
        out["min_rolling_5_proxy"] = np.nan

    return out


def _latest_player_rows(df_hist: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_hist.sort_values(["player", "game_date"], kind="mergesort")
        .groupby("player", as_index=False)
        .tail(1)
        .copy()
    )
    return out.reset_index(drop=True)


def _team_candidate_pool(
    latest_player_df: pd.DataFrame,
    slate_team: str,
    slate_game_date: pd.Timestamp,
    cfg: TodayRowConfig,
) -> tuple[pd.DataFrame, dict]:
    """
    Filter latest player rows into a likely pregame candidate pool for one team.
    Returns:
        (filtered_df, diagnostics)
    """
    base = latest_player_df.copy()
    diagnostics: dict[str, int | float | str] = {"team": slate_team}

    # Must currently belong to that team in latest historical row
    base = base[base["team"] == slate_team].copy()
    diagnostics["n_latest_on_team"] = int(len(base))

    if base.empty:
        diagnostics["reason"] = "no_latest_rows_for_team"
        return base, diagnostics

    # recency filter
    days_since_last = (slate_game_date - pd.to_datetime(base["game_date"])).dt.days
    base["days_since_last_game"] = days_since_last

    after_games = base[base["games_played_career_to_date"] >= cfg.min_games_required].copy()
    diagnostics["n_after_min_games"] = int(len(after_games))

    if cfg.active_within_days is None:
        after_recency = after_games.copy()
    else:
        after_recency = after_games[
            after_games["days_since_last_game"] <= cfg.active_within_days
        ].copy()
    diagnostics["n_after_recency"] = int(len(after_recency))

    if "min_rolling_5" in after_recency.columns:
        after_recency["minutes_filter_value"] = pd.to_numeric(after_recency["min_rolling_5"], errors="coerce")
    elif "min_rolling_5_proxy" in after_recency.columns:
        after_recency["minutes_filter_value"] = pd.to_numeric(after_recency["min_rolling_5_proxy"], errors="coerce")
    elif "mp_minutes" in after_recency.columns:
        after_recency["minutes_filter_value"] = pd.to_numeric(after_recency["mp_minutes"], errors="coerce")
    else:
        after_recency["minutes_filter_value"] = np.nan

    after_minutes = after_recency[
        after_recency["minutes_filter_value"].fillna(0) >= cfg.min_minutes_threshold
    ].copy()
    diagnostics["n_after_minutes"] = int(len(after_minutes))

    if after_minutes.empty:
        if diagnostics["n_after_recency"] == 0:
            diagnostics["reason"] = "failed_recency_filter"
        elif diagnostics["n_after_minutes"] == 0:
            diagnostics["reason"] = "failed_minutes_filter"
        else:
            diagnostics["reason"] = "empty_after_filters"
        return after_minutes, diagnostics

    sort_cols: list[str] = []
    ascending: list[bool] = []

    if "minutes_filter_value" in after_minutes.columns:
        sort_cols.append("minutes_filter_value")
        ascending.append(False)

    if "games_played_career_to_date" in after_minutes.columns:
        sort_cols.append("games_played_career_to_date")
        ascending.append(False)

    if sort_cols:
        after_minutes = after_minutes.sort_values(sort_cols, ascending=ascending, kind="mergesort").copy()

    out = after_minutes.head(cfg.max_players_per_team).copy()
    diagnostics["n_final"] = int(len(out))
    diagnostics["reason"] = "ok"

    return out.reset_index(drop=True), diagnostics


def _overwrite_today_context(
    team_pool_df: pd.DataFrame,
    *,
    game_date: pd.Timestamp,
    team: str,
    opp: str,
    is_home: int,
) -> pd.DataFrame:
    out = team_pool_df.copy()

    out["game_date"] = game_date
    out["date"] = game_date
    out["team"] = team
    out["opp"] = opp
    out["is_home"] = int(is_home)

    if "home_game" in out.columns:
        out["home_game"] = int(is_home)

    return out


def build_today_rows(
    df_hist: pd.DataFrame,
    slate_df: pd.DataFrame,
    *,
    cfg: TodayRowConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = TodayRowConfig()

    hist = _canon_hist_df(df_hist)
    hist = _add_player_history_summary(hist)

    slate = _canon_slate_df(slate_df)
    latest = _latest_player_rows(hist)

    hist_max_date = pd.to_datetime(hist["game_date"]).max()
    slate_min_date = pd.to_datetime(slate["game_date"]).min()
    slate_max_date = pd.to_datetime(slate["game_date"]).max()

    if cfg.active_within_days is not None and hist_max_date < (slate_min_date - pd.Timedelta(days=cfg.active_within_days)):
        msg = (
            "Historical data is too stale for the requested slate under current recency settings.\n"
            f"hist_max_date={hist_max_date.date()} | "
            f"slate_min_date={slate_min_date.date()} | "
            f"active_within_days={cfg.active_within_days}\n"
            "Either update history, use a slate date closer to your data, "
            "or relax active_within_days for structural testing."
        )
        raise ValueError(msg)

    all_rows: list[pd.DataFrame] = []
    diagnostics_rows: list[dict] = []

    for srow in slate.itertuples(index=False):
        team_pool, diag = _team_candidate_pool(
            latest_player_df=latest,
            slate_team=srow.team,
            slate_game_date=srow.game_date,
            cfg=cfg,
        )
        diag["game_date"] = str(pd.Timestamp(srow.game_date).date())
        diag["opp"] = srow.opp
        diagnostics_rows.append(diag)

        if team_pool.empty:
            continue

        today_rows = _overwrite_today_context(
            team_pool,
            game_date=srow.game_date,
            team=srow.team,
            opp=srow.opp,
            is_home=int(srow.is_home),
        )
        all_rows.append(today_rows)

    if not all_rows:
        diagnostics_df = pd.DataFrame(diagnostics_rows)
        msg = (
            "No rotation players found for this slate under current filters.\n"
            f"hist_max_date={hist_max_date.date()} | "
            f"slate_range={slate_min_date.date()} to {slate_max_date.date()} | "
            f"min_games_required={cfg.min_games_required} | "
            f"active_within_days={cfg.active_within_days} | "
            f"min_minutes_threshold={cfg.min_minutes_threshold}\n\n"
            "Team diagnostics:\n"
            f"{diagnostics_df.to_string(index=False)}"
        )
        if cfg.error_on_empty:
            raise ValueError(msg)
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)

    out = (
        out.sort_values(["game_date", "team", "player"], kind="mergesort")
        .drop_duplicates(subset=["game_date", "team", "player"], keep="last")
        .reset_index(drop=True)
    )

    return out


def build_today_rows_v2(
    df_hist: pd.DataFrame,
    slate_df: pd.DataFrame,
    *,
    min_games_required: int = 3,
    active_within_days: int | None = 21,
    min_minutes_threshold: float = 8.0,
    max_players_per_team: int = 15,
    error_on_empty: bool = True,
) -> pd.DataFrame:
    cfg = TodayRowConfig(
        min_games_required=min_games_required,
        active_within_days=active_within_days,
        min_minutes_threshold=min_minutes_threshold,
        max_players_per_team=max_players_per_team,
        error_on_empty=error_on_empty,
    )
    return build_today_rows(df_hist=df_hist, slate_df=slate_df, cfg=cfg)


def slate_from_team_pairs(
    *,
    game_date: str | pd.Timestamp,
    matchups: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    game_date = pd.Timestamp(game_date)

    rows = []
    for away_team, home_team in matchups:
        away_team = str(away_team).strip().upper()
        home_team = str(home_team).strip().upper()

        rows.append(
            {
                "game_date": game_date,
                "team": away_team,
                "opp": home_team,
                "is_home": 0,
            }
        )
        rows.append(
            {
                "game_date": game_date,
                "team": home_team,
                "opp": away_team,
                "is_home": 1,
            }
        )

    return pd.DataFrame(rows)