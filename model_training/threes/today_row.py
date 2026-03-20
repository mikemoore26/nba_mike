# model_training/threes/today_row.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------
# Ceiling-aware expected FG3A (heuristic)
# ----------------------------
def expected_fg3a_ceiling(ph: pd.DataFrame, recent_n: int = 5) -> float:
    tail = ph["fg3a"].tail(max(10, recent_n * 2)).dropna()
    if tail.empty:
        return 0.0

    base = float(tail.tail(recent_n).mean())
    ceiling = float(tail.quantile(0.80))
    exp = 0.65 * base + 0.35 * ceiling
    cap = float(tail.quantile(0.95))
    return float(np.clip(exp, 0, cap))


# ----------------------------
# Internals
# ----------------------------
def _canonicalize_history(history: pd.DataFrame) -> pd.DataFrame:
    """
    Make today_row builders accept either (date) or (game_date) history.
    Emits both columns for backward compat.

    Canonical column: game_date
    Legacy column:    date
    """
    h = history.copy()

    if "game_date" in h.columns:
        h["game_date"] = pd.to_datetime(h["game_date"], errors="coerce")

    if "date" in h.columns:
        h["date"] = pd.to_datetime(h["date"], errors="coerce")

    if "date" not in h.columns and "game_date" in h.columns:
        h["date"] = h["game_date"]

    if "game_date" not in h.columns and "date" in h.columns:
        h["game_date"] = h["date"]

    # drop invalid dates
    h = h.dropna(subset=["game_date"]).copy()

    # stable sort for rolling windows + “last game” logic
    h = h.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    # types
    for c in ["player", "team", "opp"]:
        if c in h.columns:
            h[c] = h[c].astype("string")

    return h


def _emit_today_row(
    *,
    player: str,
    season: int,
    game_date: pd.Timestamp,
    team: str,
    opp: str,
    is_home: int,
    mp_minutes: float,
    fga: float,
    fg3a: float,
    pts: float,
    usage: float,
) -> dict:
    """
    Emits BOTH `game_date` (canonical) and `date` (legacy) so downstream code
    can migrate gradually.
    """
    return {
        "player": player,
        "season": season,
        "game_date": game_date,
        "date": game_date,  # legacy alias
        "team": team,
        "opp": opp,
        "is_home": int(is_home),
        # placeholders (not used for today's rolling due to shift(1) in feature builder)
        "mp_minutes": float(mp_minutes) if mp_minutes is not None else np.nan,
        "fga": float(fga) if fga is not None else np.nan,
        "fg3a": float(fg3a) if fg3a is not None else np.nan,
        "pts": float(pts) if pts is not None else np.nan,
        "usage": float(usage) if usage is not None else np.nan,
        "fg3": np.nan,
    }


# ----------------------------
# Today rows (v1)
# ----------------------------
def build_today_rows(
    history: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    min_games_required: int = 10,
    recent_n: int = 5,
) -> pd.DataFrame:
    h = _canonicalize_history(history)
    game_date_ts = pd.Timestamp(game_date)

    latest_team = (
        h.sort_values("game_date")
        .groupby("player")
        .tail(1)[["player", "team", "season"]]
    )
    players = latest_team[latest_team["team"].isin([away_team, home_team])]["player"].tolist()

    game_counts = h.groupby("player").size()
    players = [p for p in players if game_counts.get(p, 0) >= min_games_required]

    rows: list[dict] = []
    for p in players:
        ph = h[h["player"] == p].sort_values("game_date")
        if ph.empty:
            continue
        prev = ph.iloc[-1]

        team = str(prev["team"])
        is_home = 1 if team == home_team else 0
        opp = away_team if is_home else home_team

        rows.append(
            _emit_today_row(
                player=str(p),
                season=int(pd.to_numeric(prev["season"], errors="coerce")) if "season" in prev else int(pd.to_numeric(h["season"], errors="coerce").dropna().max()),
                game_date=game_date_ts,
                team=team,
                opp=str(opp),
                is_home=is_home,
                mp_minutes=float(ph["mp_minutes"].tail(recent_n).mean()),
                fga=float(ph["fga"].tail(recent_n).mean()),
                fg3a=expected_fg3a_ceiling(ph, recent_n=recent_n),
                pts=float(ph["pts"].tail(recent_n).mean()),
                usage=float(ph["usage"].tail(recent_n).mean()),
            )
        )

    if not rows:
        raise ValueError("No eligible players found for this matchup.")
    return pd.DataFrame(rows)


# ----------------------------
# Today rows (v2) rotation-safe
# ----------------------------
def build_today_rows_v2(
    history: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    min_games_required: int = 10,
    recent_n: int = 5,
    active_within_days: int = 14,
    min_minutes_threshold: float = 18.0,
) -> pd.DataFrame:
    h = _canonicalize_history(history)
    game_date_ts = pd.Timestamp(game_date)

    latest = (
        h.sort_values("game_date")
        .groupby("player")
        .tail(1)[["player", "team", "season", "game_date"]]
        .rename(columns={"game_date": "last_game_date"})
    )

    candidates = latest[latest["team"].isin([away_team, home_team])].copy()

    game_counts = h.groupby("player").size()
    candidates["games_played"] = candidates["player"].map(game_counts).fillna(0).astype(int)
    candidates = candidates[candidates["games_played"] >= min_games_required]

    candidates["last_game_date"] = pd.to_datetime(candidates["last_game_date"], errors="coerce")
    candidates = candidates.dropna(subset=["last_game_date"])

    candidates["days_since_last"] = (game_date_ts - candidates["last_game_date"]).dt.days
    candidates = candidates[candidates["days_since_last"] <= active_within_days]

    rows: list[dict] = []
    for _, r in candidates.iterrows():
        p = str(r["player"])
        ph = h[h["player"] == p].sort_values("game_date")
        if ph.empty:
            continue

        prev = ph.iloc[-1]
        team = str(r["team"])
        is_home = 1 if team == home_team else 0
        opp = away_team if is_home else home_team

        mp_med = float(ph["mp_minutes"].tail(recent_n).median())
        if np.isnan(mp_med) or mp_med < float(min_minutes_threshold):
            continue

        season_val = pd.to_numeric(prev.get("season", np.nan), errors="coerce")
        if np.isnan(season_val):
            season_val = pd.to_numeric(h["season"], errors="coerce").dropna().max()

        rows.append(
            _emit_today_row(
                player=p,
                season=int(season_val),
                game_date=game_date_ts,
                team=team,
                opp=str(opp),
                is_home=is_home,
                mp_minutes=mp_med,
                fga=float(ph["fga"].tail(recent_n).mean()),
                fg3a=expected_fg3a_ceiling(ph, recent_n=recent_n),
                pts=float(ph["pts"].tail(recent_n).mean()),
                usage=float(ph["usage"].tail(recent_n).mean()),
            )
        )

    if not rows:
        raise ValueError(
            "No rotation players found for this matchup under current filters. "
            f"(min_games_required={min_games_required}, active_within_days={active_within_days}, "
            f"min_minutes_threshold={min_minutes_threshold})"
        )

    today_df = pd.DataFrame(rows)

    # Ensure canonical date is valid
    today_df["game_date"] = pd.to_datetime(today_df["game_date"], errors="coerce")
    if today_df["game_date"].isna().any():
        raise ValueError("today_df contains invalid game_date after coercion.")

    # Legacy alias stays in sync
    today_df["date"] = today_df["game_date"]

    return today_df