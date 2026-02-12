# model_training/threes/today_row.py
import pandas as pd
import numpy as np


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
    history = history.sort_values(["player", "date"]).copy()
    game_date = pd.Timestamp(game_date)

    latest_team = (
        history.sort_values("date")
        .groupby("player")
        .tail(1)[["player", "team", "season"]]
    )
    players = latest_team[latest_team["team"].isin([away_team, home_team])]["player"].tolist()

    game_counts = history.groupby("player").size()
    players = [p for p in players if game_counts.get(p, 0) >= min_games_required]

    rows = []
    for p in players:
        ph = history[history["player"] == p].sort_values("date")
        prev = ph.iloc[-1]

        team = prev["team"]
        is_home = 1 if team == home_team else 0
        opp = away_team if is_home else home_team

        rows.append({
            "player": p,
            "season": prev["season"],
            "date": game_date,
            "team": team,
            "opp": opp,

            # placeholders (not used for today's features due to shift(1))
            "mp_minutes": float(ph["mp_minutes"].tail(recent_n).mean()),
            "fga": float(ph["fga"].tail(recent_n).mean()),
            "fg3a": expected_fg3a_ceiling(ph, recent_n=recent_n),
            "pts": float(ph["pts"].tail(recent_n).mean()),
            "usage": float(ph["usage"].tail(recent_n).mean()),

            "is_home": int(is_home),
            "fg3": np.nan,
        })

    today_df = pd.DataFrame(rows)
    if today_df.empty:
        raise ValueError("No eligible players found for this matchup.")
    return today_df


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
    history = history.sort_values(["player", "date"]).copy()
    game_date = pd.Timestamp(game_date)

    latest = (
        history.sort_values("date")
        .groupby("player")
        .tail(1)[["player", "team", "season", "date"]]
        .rename(columns={"date": "last_game_date"})
    )

    candidates = latest[latest["team"].isin([away_team, home_team])].copy()

    game_counts = history.groupby("player").size()
    candidates["games_played"] = candidates["player"].map(game_counts).fillna(0).astype(int)
    candidates = candidates[candidates["games_played"] >= min_games_required]

    # last_game_date should already be datetime in history; enforce just in case
    candidates["last_game_date"] = pd.to_datetime(candidates["last_game_date"], errors="coerce")
    candidates = candidates.dropna(subset=["last_game_date"])

    candidates["days_since_last"] = (game_date - candidates["last_game_date"]).dt.days
    candidates = candidates[candidates["days_since_last"] <= active_within_days]

    rows: list[dict] = []
    for _, r in candidates.iterrows():
        p = r["player"]
        ph = history[history["player"] == p].sort_values("date")
        if ph.empty:
            continue

        prev = ph.iloc[-1]
        team = r["team"]
        is_home = 1 if team == home_team else 0
        opp = away_team if is_home else home_team

        mp_med = float(ph["mp_minutes"].tail(recent_n).median())
        if np.isnan(mp_med) or mp_med < min_minutes_threshold:
            continue

        rows.append(
            {
                "player": p,
                "season": prev["season"],
                "date": game_date,
                "team": team,
                "opp": opp,
                # placeholders (not used for today's features due to shift(1))
                "mp_minutes": mp_med,
                "fga": float(ph["fga"].tail(recent_n).mean()),
                "fg3a": expected_fg3a_ceiling(ph, recent_n=recent_n),
                "pts": float(ph["pts"].tail(recent_n).mean()),
                "usage": float(ph["usage"].tail(recent_n).mean()),
                "is_home": int(is_home),
                "fg3": np.nan,
            }
        )

    # ✅ EMPTY CHECK BEFORE COLUMN ACCESS
    if not rows:
        raise ValueError(
            "No rotation players found for this matchup under current filters. "
            f"(min_games_required={min_games_required}, active_within_days={active_within_days}, "
            f"min_minutes_threshold={min_minutes_threshold})"
        )

    today_df = pd.DataFrame(rows)

    # Ensure season exists + filled
    if "season" not in today_df.columns or today_df["season"].isna().any():
        current_season = int(pd.to_numeric(history["season"], errors="coerce").dropna().max())
        today_df["season"] = today_df["season"].fillna(current_season)

    # Ensure date is datetime
    today_df["date"] = pd.to_datetime(today_df["date"], errors="coerce")
    if today_df["date"].isna().any():
        raise ValueError("today_df contains invalid dates after coercion.")

    return today_df
