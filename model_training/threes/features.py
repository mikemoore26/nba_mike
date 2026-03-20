from __future__ import annotations

import numpy as np
import pandas as pd


FG3_FEATURES_ATT = [
    # role / volume
    "min_rolling_5",
    "fga_rolling_5",
    "fg3a_rolling_5",
    "fg3m_rolling_5",

    # player baselines
    "player_fg3a_season_avg",
    "player_min_season_avg",
    "player_usage_season",

    # context
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_flag",

    # stint / team change
    "games_played_to_date",
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",

    # opponent 3P defense
    "opp_fg3a_allowed_pg_to_date",
    "opp_fg3m_allowed_pg_to_date",
    "opp_3p_pct_allowed_to_date",
    "opp_def_3p_rank_to_date",
]

FG3_FEATURES_RATE = [
    "fg3_pct_rolling_10",
    "fg3_att_rolling_10",
    "player_fg3_pct_season",
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_prob_10",
    "opp_3p_pct_allowed_to_date",
]


def _canon_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "game_date" not in out.columns and "date" in out.columns:
        out["game_date"] = out["date"]

    if "date" not in out.columns and "game_date" in out.columns:
        out["date"] = out["game_date"]

    out = out.dropna(subset=["game_date"]).copy()
    return out


def build_features_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing/rolling features using shift(1), safe for combined history + today rows.
    """
    df = _canon_dates(df)
    df = df.sort_values(["player", "game_date"], kind="mergesort").copy()

    g = df.groupby("player", sort=False)

    df["min_rolling_5"] = g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fga_rolling_5"] = g["fga"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg3a_rolling_5"] = g["fg3a"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg3m_rolling_5"] = g["fg3m"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)

    made10 = g["fg3m"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    att10 = g["fg3a"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)

    df["fg3_att_rolling_10"] = att10
    df["fg3_pct_rolling_10"] = (made10 / att10).where(att10 > 0)

    df["days_rest"] = g["game_date"].diff().dt.days.reset_index(level=0, drop=True)
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["home_game"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int)

    df["starter_prob_10"] = (
        g["mp_minutes"].shift(1).rolling(10).mean().reset_index(level=0, drop=True) / 30.0
    ).clip(0, 1)

    if "starter_flag" not in df.columns:
        df["starter_flag"] = 0

    return df


def add_player_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["player", "season", "game_date"], kind="mergesort").copy()
    g = df.groupby(["player", "season"], sort=False)

    df["player_fg3a_season_avg"] = (
        g["fg3a"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    made = g["fg3m"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    att = g["fg3a"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    df["player_fg3_pct_season"] = (made / att).where(att > 0)

    df["player_min_season_avg"] = (
        g["mp_minutes"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    if "usage" in df.columns:
        df["player_usage_season"] = (
            g["usage"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_usage_season"] = np.nan

    df["fg3_pct_rolling_10"] = df["fg3_pct_rolling_10"].fillna(df["player_fg3_pct_season"])

    return df


def add_team_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["player", "season", "game_date"], kind="mergesort").copy()

    g = df.groupby(["player", "season"], sort=False)

    df["games_played_to_date"] = g.cumcount()

    prev_team = g["team"].shift(1)
    changed = ((df["team"] != prev_team) & prev_team.notna()).astype(int)
    df["_changed_team"] = changed

    df["new_team_game"] = df["_changed_team"]
    df["_stint_id"] = g["_changed_team"].cumsum()

    df["team_games_in_stint_to_date"] = (
        df.groupby(["player", "season", "_stint_id"], sort=False).cumcount()
    )

    df["recent_team_change_5"] = (
        g["_changed_team"]
        .apply(lambda s: s.shift(1).rolling(5).max())
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype(int)
    )

    return df.drop(columns=["_changed_team", "_stint_id"])


def add_opp_3p_defense_to_date(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["season", "game_date", "team", "opp"], kind="mergesort").copy()

    team_game = (
        df.groupby(["season", "game_date", "team", "opp"], as_index=False)[["fg3a", "fg3m"]]
        .sum()
        .rename(columns={"fg3a": "team_fg3a_game", "fg3m": "team_fg3m_game"})
    )

    allowed = team_game.rename(
        columns={
            "opp": "def_team",
            "team_fg3a_game": "allowed_fg3a_game",
            "team_fg3m_game": "allowed_fg3m_game",
        }
    )[["season", "game_date", "def_team", "allowed_fg3a_game", "allowed_fg3m_game"]]

    allowed = allowed.sort_values(["def_team", "season", "game_date"], kind="mergesort").copy()
    g = allowed.groupby(["def_team", "season"], sort=False)

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
    allowed["opp_def_3p_rank_to_date"] = (
        allowed.groupby(["season", "game_date"])["opp_3p_pct_allowed_to_date"]
        .rank(method="average", ascending=True)
    )

    allowed = allowed.rename(columns={"def_team": "opp"})
    df = df.merge(
        allowed[
            [
                "season",
                "game_date",
                "opp",
                "opp_fg3a_allowed_pg_to_date",
                "opp_fg3m_allowed_pg_to_date",
                "opp_3p_pct_allowed_to_date",
                "opp_def_3p_rank_to_date",
            ]
        ],
        on=["season", "game_date", "opp"],
        how="left",
    )

    league_pct = float(df["fg3m"].sum() / max(df["fg3a"].sum(), 1))
    df["opp_3p_pct_allowed_to_date"] = df["opp_3p_pct_allowed_to_date"].fillna(league_pct)
    df["opp_fg3a_allowed_pg_to_date"] = df["opp_fg3a_allowed_pg_to_date"].fillna(df["fg3a"].mean())
    df["opp_fg3m_allowed_pg_to_date"] = df["opp_fg3m_allowed_pg_to_date"].fillna(df["fg3m"].mean())

    if df["opp_def_3p_rank_to_date"].notna().any():
        df["opp_def_3p_rank_to_date"] = df["opp_def_3p_rank_to_date"].fillna(
            df["opp_def_3p_rank_to_date"].median()
        )
    else:
        df["opp_def_3p_rank_to_date"] = 15.0

    return df


def build_all_threes_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = build_features_no_leak(df)
    df = add_player_baselines(df)
    df = add_team_stint_features(df)
    df = add_opp_3p_defense_to_date(df)
    df["date"] = df["game_date"]
    return df