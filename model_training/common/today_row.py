from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_SLATE_COLS = ["game_date", "team", "opp", "is_home"]
REQUIRED_HIST_COLS = ["game_date", "player", "team", "opp"]


@dataclass
class TodayRowConfig:
    min_games_required: int = 3
    active_within_days: Optional[int] = 21
    min_minutes_threshold: float = 8.0
    max_players_per_team: int = 12
    error_on_empty: bool = True


def _parse_single_mp_value(val) -> float:
    """
    Parse one Basketball-Reference-style mp value into decimal minutes.

    Handles:
    - '34:12' -> 34.2
    - '12'    -> 12.0
    - 15      -> 15.0
    - blanks / DNP-ish strings -> np.nan
    """
    if pd.isna(val):
        return np.nan

    s = str(val).strip()
    if s == "":
        return np.nan

    lowered = s.lower()
    bad_tokens = {
        "did not play",
        "dnp",
        "inactive",
        "not with team",
        "did not dress",
        "player suspended",
    }
    if lowered in bad_tokens:
        return np.nan

    if ":" in s:
        parts = s.split(":", 1)
        try:
            mins = float(parts[0])
            secs = float(parts[1])
            return mins + secs / 60.0
        except Exception:
            return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def _parse_mp_to_minutes(mp_series: pd.Series) -> pd.Series:
    return mp_series.apply(_parse_single_mp_value).astype(float)


def _canon_slate_df(slate_df: pd.DataFrame) -> pd.DataFrame:
    out = slate_df.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()

    out["team"] = out["team"].astype(str).str.upper().str.strip()
    out["opp"] = out["opp"].astype(str).str.upper().str.strip()
    out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(int)

    return out


def _canon_hist_df(df_hist: pd.DataFrame) -> pd.DataFrame:
    out = df_hist.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()

    out["player"] = out["player"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.upper().str.strip()
    out["opp"] = out["opp"].astype(str).str.upper().str.strip()

    parsed_from_mp = None
    if "mp" in out.columns:
        parsed_from_mp = _parse_mp_to_minutes(out["mp"])

    if "mp_minutes" in out.columns:
        out["mp_minutes"] = pd.to_numeric(out["mp_minutes"], errors="coerce")

        non_null_rate = out["mp_minutes"].notna().mean()
        positive_rate = (out["mp_minutes"].fillna(0) > 0).mean()

        needs_repair = (non_null_rate < 0.5) or (positive_rate < 0.05)

        if needs_repair and parsed_from_mp is not None:
            out["mp_minutes"] = parsed_from_mp
    else:
        if parsed_from_mp is not None:
            out["mp_minutes"] = parsed_from_mp

    if "minutes" in out.columns:
        out["minutes"] = pd.to_numeric(out["minutes"], errors="coerce")

    return out


def _latest_pre_date_player_rows(df_hist: pd.DataFrame, run_date: pd.Timestamp) -> pd.DataFrame:
    pre = df_hist.loc[df_hist["game_date"] < run_date].copy()
    if pre.empty:
        return pre

    # Use latest REAL game played, not latest row blindly.
    if "mp_minutes" in pre.columns:
        pre = pre.loc[pd.to_numeric(pre["mp_minutes"], errors="coerce") > 0].copy()
    elif "minutes" in pre.columns:
        pre = pre.loc[pd.to_numeric(pre["minutes"], errors="coerce") > 0].copy()

    if pre.empty:
        return pre

    pre = pre.sort_values(["player", "game_date"])
    return pre.groupby("player").tail(1).reset_index(drop=True)


def _team_candidate_pool(
    *,
    latest_player_df: pd.DataFrame,
    slate_team: str,
    slate_game_date: pd.Timestamp,
    cfg: TodayRowConfig,
):
    base = latest_player_df[latest_player_df["team"] == slate_team].copy()

    if base.empty:
        return base, {}

    if "games_played_to_date" in base.columns:
        games = pd.to_numeric(base["games_played_to_date"], errors="coerce")
    else:
        games = pd.Series(999, index=base.index, dtype=float)

    base = base[games >= cfg.min_games_required]

    if cfg.active_within_days is not None:
        days = (slate_game_date - base["game_date"]).dt.days
        base = base[days <= cfg.active_within_days]

    if base.empty:
        return base, {}

    minutes_val = None

    if "min_rolling_5" in base.columns:
        minutes_val = pd.to_numeric(base["min_rolling_5"], errors="coerce")

    if minutes_val is None or minutes_val.isna().all():
        if "min_rolling_5_proxy" in base.columns:
            minutes_val = pd.to_numeric(base["min_rolling_5_proxy"], errors="coerce")

    if minutes_val is None or minutes_val.isna().all():
        if "min_rolling_3_proxy" in base.columns:
            minutes_val = pd.to_numeric(base["min_rolling_3_proxy"], errors="coerce")

    if minutes_val is None or minutes_val.isna().all():
        if "mp_minutes" in base.columns:
            minutes_val = pd.to_numeric(base["mp_minutes"], errors="coerce")

    if minutes_val is None or minutes_val.isna().all():
        if "minutes" in base.columns:
            minutes_val = pd.to_numeric(base["minutes"], errors="coerce")

    base["minutes_filter_value"] = minutes_val

    mask = (
        base["minutes_filter_value"].notna()
        & (base["minutes_filter_value"] >= cfg.min_minutes_threshold)
    )

    if len(base) > 0 and mask.sum() == 0:
        print(
            f"[TODAY_ROWS][WARN] Team={slate_team} all players failed minutes filter "
            f"(threshold={cfg.min_minutes_threshold}). Falling back."
        )
        filtered = base
    else:
        filtered = base.loc[mask]

    if filtered.empty:
        return filtered, {}

    filtered = filtered.sort_values("minutes_filter_value", ascending=False)

    return filtered.head(cfg.max_players_per_team).reset_index(drop=True), {}


def _overwrite_today_context(df, game_date, team, opp, is_home):
    out = df.copy()

    out["game_date"] = game_date
    out["date"] = game_date
    out["team"] = team
    out["opp"] = opp
    out["is_home"] = int(is_home)

    return out


def build_today_rows(df_hist, slate_df, *, cfg: TodayRowConfig | None = None):
    if cfg is None:
        cfg = TodayRowConfig()

    hist = _canon_hist_df(df_hist)
    slate = _canon_slate_df(slate_df)

    all_rows = []

    for row in slate.itertuples(index=False):
        latest = _latest_pre_date_player_rows(hist, row.game_date)

        pool, _ = _team_candidate_pool(
            latest_player_df=latest,
            slate_team=row.team,
            slate_game_date=row.game_date,
            cfg=cfg,
        )

        if pool.empty:
            continue

        today = _overwrite_today_context(
            pool,
            game_date=row.game_date,
            team=row.team,
            opp=row.opp,
            is_home=row.is_home,
        )

        all_rows.append(today)

    if not all_rows:
        if cfg.error_on_empty:
            raise ValueError("No players found for slate")
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    return out


def build_today_rows_v2(
    df_hist,
    slate_df,
    *,
    min_games_required=3,
    active_within_days=21,
    min_minutes_threshold=8.0,
    max_players_per_team=12,
    error_on_empty=True,
):
    cfg = TodayRowConfig(
        min_games_required=min_games_required,
        active_within_days=active_within_days,
        min_minutes_threshold=min_minutes_threshold,
        max_players_per_team=max_players_per_team,
        error_on_empty=error_on_empty,
    )
    return build_today_rows(df_hist, slate_df, cfg=cfg)


def slate_from_team_pairs(
    *,
    game_date,
    matchups: Iterable[tuple[str, str]],
):
    game_date = pd.Timestamp(game_date)

    rows = []
    for away, home in matchups:
        rows.append({"game_date": game_date, "team": away, "opp": home, "is_home": 0})
        rows.append({"game_date": game_date, "team": home, "opp": away, "is_home": 1})

    return pd.DataFrame(rows)