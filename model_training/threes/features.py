# model_training/threes/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Feature lists (must match training artifacts)
# ------------------------------------------------------------
FG3A_FEATURES = [
    # role/volume signals
    "min_rolling_5",
    "fga_rolling_5",
    "fg3a_rolling_5",

    # player baselines (season to date)
    "player_fg3a_season_avg",
    "player_min_season_avg",
    "player_usage_season",

    # context
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_flag",

    # stint/team change
    "games_played_to_date",
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",

    # opponent 3P defense allowed-to-date
    "opp_fg3a_allowed_pg_to_date",
    "opp_fg3m_allowed_pg_to_date",
    "opp_3p_pct_allowed_to_date",
    "opp_def_3p_rank_to_date",
]

RATE_FEATURES = [
    "fg3_pct_rolling_10",
    "fg3_att_rolling_10",
    "player_fg3_pct_season",
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_prob_10",
]


# ------------------------------------------------------------
# Canonical date helpers (game_date-only inside features)
# ------------------------------------------------------------
def _canon_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical internal time column: game_date
    Accepts legacy `date` and creates/keeps both for compatibility.

    IMPORTANT: all feature logic uses `game_date` to avoid drift/leakage bugs.
    """
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "game_date" not in out.columns and "date" in out.columns:
        out["game_date"] = out["date"]

    if "date" not in out.columns and "game_date" in out.columns:
        out["date"] = out["game_date"]

    # drop invalid dates once (prevents weird rolling alignment)
    out = out.dropna(subset=["game_date"]).copy()

    return out


# ------------------------------------------------------------
# Core: no-leak rolling features
# ------------------------------------------------------------
def build_features_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing/rolling features using shift(1) so each row uses ONLY past games.

    Requires columns:
      player, game_date, season, team, opp,
      mp_minutes, fga, fg3a, fg3, usage, is_home
    """
    df = _canon_dates(df)
    df = df.sort_values(["player", "game_date"], kind="mergesort").copy()

    g = df.groupby("player", sort=False)

    # trailing rolling means (past-only)
    df["min_rolling_5"] = g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fga_rolling_5"] = g["fga"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg3a_rolling_5"] = g["fg3a"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg3_rolling_5"] = g["fg3"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)

    made10 = g["fg3"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    att10 = g["fg3a"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)

    df["fg3_att_rolling_10"] = att10
    df["fg3_pct_rolling_10"] = (made10 / att10).where(att10 > 0)

    # rest/context (past-only by definition: diff over historical dates)
    df["days_rest"] = g["game_date"].diff().dt.days.reset_index(level=0, drop=True)
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)

    # home flag (pregame-known)
    df["home_game"] = df["is_home"].astype(int)

    # starter_prob_10: proxy for minutes share / role stability (past-only)
    df["starter_prob_10"] = (
        g["mp_minutes"].shift(1).rolling(10).mean().reset_index(level=0, drop=True) / 30.0
    ).clip(0, 1)

    # starter_flag: if missing, default 0 (pregame unknown)
    if "starter_flag" not in df.columns:
        df["starter_flag"] = 0

    return df


# ------------------------------------------------------------
# Player baselines (season-scoped, past-only)
# ------------------------------------------------------------
def add_player_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-scoped expanding stats via shift(1) (past-only).

    Requires: player, season, game_date, fg3a, fg3, mp_minutes, usage
    """
    df = _canon_dates(df)
    df = df.sort_values(["player", "season", "game_date"], kind="mergesort").copy()
    g = df.groupby(["player", "season"], sort=False)

    df["player_fg3a_season_avg"] = (
        g["fg3a"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    made = g["fg3"].apply(lambda s: s.shift(1).expanding().sum())
    att = g["fg3a"].apply(lambda s: s.shift(1).expanding().sum())
    df["player_fg3_pct_season"] = (made / att).reset_index(level=[0, 1], drop=True)

    df["player_min_season_avg"] = (
        g["mp_minutes"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    df["player_usage_season"] = (
        g["usage"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    # Fill rolling pct when no attempts in last 10 with season pct
    df["fg3_pct_rolling_10"] = df["fg3_pct_rolling_10"].fillna(df["player_fg3_pct_season"])

    return df


# ------------------------------------------------------------
# Team stint / team-change features (pregame-safe)
# ------------------------------------------------------------
def add_team_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
      - games_played_to_date
      - team_games_in_stint_to_date
      - new_team_game
      - recent_team_change_5

    Requires: player, season, game_date, team
    """
    df = _canon_dates(df)
    df = df.sort_values(["player", "season", "game_date"], kind="mergesort").copy()

    g = df.groupby(["player", "season"], sort=False)

    # game index within season (0-based)
    df["games_played_to_date"] = g.cumcount()

    prev_team = g["team"].shift(1)
    changed = ((df["team"] != prev_team) & prev_team.notna()).astype(int)
    df["_changed_team"] = changed

    # first game after team change
    df["new_team_game"] = df["_changed_team"]

    # stint id increments when team changes
    df["_stint_id"] = g["_changed_team"].cumsum()

    # games into current stint (0-based)
    df["team_games_in_stint_to_date"] = df.groupby(["player", "season", "_stint_id"], sort=False).cumcount()

    # recent team change in last 5 PRIOR games (shifted)
    df["recent_team_change_5"] = (
        g["_changed_team"]
        .apply(lambda s: s.shift(1).rolling(5).max())
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype(int)
    )

    return df.drop(columns=["_changed_team", "_stint_id"])


# ------------------------------------------------------------
# Opponent 3P defense allowed-to-date (pregame-safe)
# ------------------------------------------------------------
def add_opp_3p_defense_to_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds opponent defense 'allowed' features to date (pregame-safe), derived from
    team-game totals aggregated from player rows.

    Adds:
      - opp_fg3a_allowed_pg_to_date
      - opp_fg3m_allowed_pg_to_date
      - opp_3p_pct_allowed_to_date
      - opp_def_3p_rank_to_date

    Requires: season, game_date, team, opp, fg3a, fg3
    """
    df = _canon_dates(df)
    df = df.sort_values(["season", "game_date", "team", "opp"], kind="mergesort").copy()

    # 1) Team-game offense totals (aggregated from player rows)
    team_game = (
        df.groupby(["season", "game_date", "team", "opp"], as_index=False)[["fg3a", "fg3"]]
        .sum()
        .rename(columns={"fg3a": "team_fg3a_game", "fg3": "team_fg3m_game"})
    )

    # 2) Convert to defense "allowed" rows (opp is defending team)
    allowed = team_game.rename(columns={
        "opp": "def_team",
        "team_fg3a_game": "allowed_fg3a_game",
        "team_fg3m_game": "allowed_fg3m_game",
    })[["season", "game_date", "def_team", "allowed_fg3a_game", "allowed_fg3m_game"]]

    allowed = allowed.sort_values(["def_team", "season", "game_date"], kind="mergesort").copy()
    g = allowed.groupby(["def_team", "season"], sort=False)

    # 3) To-date means (shifted)
    allowed["opp_fg3a_allowed_pg_to_date"] = (
        g["allowed_fg3a_game"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    allowed["opp_fg3m_allowed_pg_to_date"] = (
        g["allowed_fg3m_game"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # 4) To-date pct allowed (shifted expanding sums)
    allowed_att_sum = (
        g["allowed_fg3a_game"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=[0, 1], drop=True)
    )
    allowed_made_sum = (
        g["allowed_fg3m_game"]
        .apply(lambda s: s.shift(1).expanding().sum())
        .reset_index(level=[0, 1], drop=True)
    )

    allowed["opp_3p_pct_allowed_to_date"] = allowed_made_sum / allowed_att_sum

    # 5) Rank defenses on each date within season (lower pct allowed = better defense)
    allowed["opp_def_3p_rank_to_date"] = (
        allowed.groupby(["season", "game_date"])["opp_3p_pct_allowed_to_date"]
        .rank(method="average", ascending=True)
    )

    # 6) Merge onto player rows by opponent faced
    allowed = allowed.rename(columns={"def_team": "opp"})
    df = df.merge(
        allowed[[
            "season", "game_date", "opp",
            "opp_fg3a_allowed_pg_to_date",
            "opp_fg3m_allowed_pg_to_date",
            "opp_3p_pct_allowed_to_date",
            "opp_def_3p_rank_to_date",
        ]],
        on=["season", "game_date", "opp"],
        how="left",
    )

    # 7) Safe fills for early season / missing
    league_pct = float(df["fg3"].sum() / max(df["fg3a"].sum(), 1))

    df["opp_3p_pct_allowed_to_date"] = df["opp_3p_pct_allowed_to_date"].fillna(league_pct)
    df["opp_fg3a_allowed_pg_to_date"] = df["opp_fg3a_allowed_pg_to_date"].fillna(df["fg3a"].mean())
    df["opp_fg3m_allowed_pg_to_date"] = df["opp_fg3m_allowed_pg_to_date"].fillna(df["fg3"].mean())

    if df["opp_def_3p_rank_to_date"].notna().any():
        df["opp_def_3p_rank_to_date"] = df["opp_def_3p_rank_to_date"].fillna(
            df["opp_def_3p_rank_to_date"].median()
        )
    else:
        df["opp_def_3p_rank_to_date"] = 15.0

    return df


# ------------------------------------------------------------
# One-stop builder
# ------------------------------------------------------------
def build_all_threes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-stop feature builder for threes models.
    Order matters.

    Internal canonical: game_date
    Output retains legacy `date` alias for compatibility.
    """
    df = _canon_dates(df)
    df = build_features_no_leak(df)
    df = add_player_baselines(df)
    df = add_team_stint_features(df)
    df = add_opp_3p_defense_to_date(df)
    # keep legacy alias synchronized
    df["date"] = df["game_date"]
    return df


# ------------------------------------------------------------
# Backward-compatible aliases (old names)
# ------------------------------------------------------------
def add_opp_3p_defense_features_roll(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias (old name) -> new implementation."""
    return add_opp_3p_defense_to_date(df)


def add_opp_3p_defense_features_to_date(df: pd.DataFrame) -> pd.DataFrame:
    """Optional alias if any old code uses this name."""
    return add_opp_3p_defense_to_date(df)