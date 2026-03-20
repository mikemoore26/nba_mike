from __future__ import annotations

import numpy as np
import pandas as pd


PTS_FEATURES_FG2A = [
    "min_rolling_5",
    "fga_rolling_5",
    "fg2a_rolling_5",
    "pts_rolling_5",
    "player_fg2a_season_avg",
    "player_min_season_avg",
    "player_usage_season",
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_flag",
    "games_played_to_date",
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",
    "team_pts_pg_to_date",
    "opp_pts_allowed_pg_to_date",
    "opp_fg_pct_allowed_to_date",
]

PTS_FEATURES_FG2RATE = [
    "fg2_pct_rolling_10",
    "fg2a_rolling_10",
    "player_fg2_pct_season",
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_prob_10",
    "opp_fg_pct_allowed_to_date",
]

PTS_FEATURES_FTA = [
    "min_rolling_5",
    "fta_rolling_5",
    "player_fta_season_avg",
    "player_usage_season",
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_flag",
    "games_played_to_date",
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",
    "opp_fta_allowed_pg_to_date",
    "opp_pts_allowed_pg_to_date",
]

PTS_FEATURES_FTRATE = [
    "ft_pct_rolling_10",
    "fta_rolling_10",
    "player_ft_pct_season",
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_prob_10",
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


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer / denom.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _ensure_scoring_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make point-composition columns robust to schema variants:
      - fgm or fg
      - ftm or ft
      - fg3m or fg3
    """
    out = df.copy()

    # Canonical aliases if only raw names exist
    if "fgm" not in out.columns and "fg" in out.columns:
        out["fgm"] = out["fg"]

    if "ftm" not in out.columns and "ft" in out.columns:
        out["ftm"] = out["ft"]

    if "fg3m" not in out.columns and "fg3" in out.columns:
        out["fg3m"] = out["fg3"]

    # numeric coercion for source cols
    for col in ["fga", "fgm", "fg3a", "fg3m", "fta", "ftm"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # derive 2PA / 2PM
    if "fg2a" not in out.columns:
        if {"fga", "fg3a"}.issubset(out.columns):
            out["fg2a"] = out["fga"] - out["fg3a"]
        else:
            out["fg2a"] = np.nan

    if "fg2m" not in out.columns:
        if {"fgm", "fg3m"}.issubset(out.columns):
            out["fg2m"] = out["fgm"] - out["fg3m"]
        else:
            out["fg2m"] = np.nan

    out["fg2a"] = pd.to_numeric(out["fg2a"], errors="coerce")
    out["fg2m"] = pd.to_numeric(out["fg2m"], errors="coerce")

    out.loc[out["fg2a"] < 0, "fg2a"] = np.nan
    out.loc[out["fg2m"] < 0, "fg2m"] = np.nan

    return out


def build_features_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = _ensure_scoring_columns(df)
    df = df.sort_values(["player", "game_date"], kind="mergesort").copy()

    g = df.groupby("player", sort=False)

    df["min_rolling_5"] = g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fga_rolling_5"] = g["fga"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["pts_rolling_5"] = g["pts"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)

    df["fg2a_rolling_5"] = g["fg2a"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg2a_rolling_10"] = g["fg2a"].shift(1).rolling(10).mean().reset_index(level=0, drop=True)

    df["fta_rolling_5"] = g["fta"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fta_rolling_10"] = g["fta"].shift(1).rolling(10).mean().reset_index(level=0, drop=True)

    fg2m_10 = g["fg2m"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    fg2a_10 = g["fg2a"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    df["fg2_pct_rolling_10"] = (fg2m_10 / fg2a_10).where(fg2a_10 > 0)

    ftm_10 = g["ftm"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    fta_10 = g["fta"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    df["ft_pct_rolling_10"] = (ftm_10 / fta_10).where(fta_10 > 0)

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
    df = _ensure_scoring_columns(df)
    df = df.sort_values(["player", "season", "game_date"], kind="mergesort").copy()
    g = df.groupby(["player", "season"], sort=False)

    df["player_fg2a_season_avg"] = (
        g["fg2a"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    fg2m_sum = g["fg2m"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    fg2a_sum = g["fg2a"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    df["player_fg2_pct_season"] = _safe_div(fg2m_sum, fg2a_sum)

    df["player_fta_season_avg"] = (
        g["fta"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    ftm_sum = g["ftm"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    fta_sum = g["fta"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    df["player_ft_pct_season"] = _safe_div(ftm_sum, fta_sum)

    df["player_min_season_avg"] = (
        g["mp_minutes"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    if "usage" in df.columns:
        df["player_usage_season"] = (
            g["usage"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_usage_season"] = np.nan

    df["fg2_pct_rolling_10"] = df["fg2_pct_rolling_10"].fillna(df["player_fg2_pct_season"])
    df["ft_pct_rolling_10"] = df["ft_pct_rolling_10"].fillna(df["player_ft_pct_season"])

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


def add_team_context_to_date(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["season", "game_date", "team"], kind="mergesort").copy()

    team_game = (
        df.groupby(["season", "game_date", "team"], as_index=False)[["pts"]]
        .sum()
        .sort_values(["team", "season", "game_date"], kind="mergesort")
        .copy()
    )

    g = team_game.groupby(["team", "season"], sort=False)
    team_game["team_pts_pg_to_date"] = (
        g["pts"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    df = df.merge(
        team_game[["season", "game_date", "team", "team_pts_pg_to_date"]],
        on=["season", "game_date", "team"],
        how="left",
    )
    return df


def add_opp_context_to_date(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = _ensure_scoring_columns(df)
    df = df.sort_values(["season", "game_date", "team", "opp"], kind="mergesort").copy()

    team_game = (
        df.groupby(["season", "game_date", "team", "opp"], as_index=False)[["pts", "fta", "fgm", "fga"]]
        .sum()
        .sort_values(["season", "game_date", "team", "opp"], kind="mergesort")
        .copy()
    )

    allowed = team_game.rename(
        columns={
            "opp": "def_team",
            "pts": "allowed_pts_game",
            "fta": "allowed_fta_game",
            "fgm": "allowed_fgm_game",
            "fga": "allowed_fga_game",
        }
    )[["season", "game_date", "def_team", "allowed_pts_game", "allowed_fta_game", "allowed_fgm_game", "allowed_fga_game"]]

    allowed = allowed.sort_values(["def_team", "season", "game_date"], kind="mergesort").copy()
    g = allowed.groupby(["def_team", "season"], sort=False)

    allowed["opp_pts_allowed_pg_to_date"] = (
        g["allowed_pts_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )
    allowed["opp_fta_allowed_pg_to_date"] = (
        g["allowed_fta_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    fgm_sum = g["allowed_fgm_game"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    fga_sum = g["allowed_fga_game"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    allowed["opp_fg_pct_allowed_to_date"] = _safe_div(fgm_sum, fga_sum)

    allowed = allowed.rename(columns={"def_team": "opp"})
    df = df.merge(
        allowed[
            [
                "season",
                "game_date",
                "opp",
                "opp_pts_allowed_pg_to_date",
                "opp_fta_allowed_pg_to_date",
                "opp_fg_pct_allowed_to_date",
            ]
        ],
        on=["season", "game_date", "opp"],
        how="left",
    )

    df["opp_pts_allowed_pg_to_date"] = df["opp_pts_allowed_pg_to_date"].fillna(df["pts"].mean())
    df["opp_fta_allowed_pg_to_date"] = df["opp_fta_allowed_pg_to_date"].fillna(df["fta"].mean())

    if {"fgm", "fga"}.issubset(df.columns):
        league_fg_pct = float(df["fgm"].sum() / max(df["fga"].sum(), 1))
    else:
        league_fg_pct = 0.47
    df["opp_fg_pct_allowed_to_date"] = df["opp_fg_pct_allowed_to_date"].fillna(league_fg_pct)

    return df


def build_all_points_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = build_features_no_leak(df)
    df = add_player_baselines(df)
    df = add_team_stint_features(df)
    df = add_team_context_to_date(df)
    df = add_opp_context_to_date(df)
    df["date"] = df["game_date"]
    return df