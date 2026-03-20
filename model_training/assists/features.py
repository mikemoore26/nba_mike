from __future__ import annotations

import numpy as np
import pandas as pd


AST_FEATURES = [
    # role / opportunity
    "min_rolling_5",
    "ast_rolling_5",
    "ast_lag_1",
    "ast_per_min_5",
    "ast_per_min_10",

    # player baselines (season to date)
    "player_ast_season_avg",
    "player_min_season_avg",
    "player_ast_per_min_season",
    "player_tov_season_avg",
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

    # team context
    "team_pts_pg_to_date",
    "team_ast_pg_to_date",
    "team_fg_pct_to_date",
    "team_3p_pct_to_date",

    # opponent defense allowed-to-date
    "opp_pts_allowed_pg_to_date",
    "opp_ast_allowed_pg_to_date",
    "opp_fg_pct_allowed_to_date",
    "opp_3p_pct_allowed_to_date",
    "opp_def_ast_rank_to_date",
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


def build_features_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing/rolling features using shift(1), safe for combined history + today rows.
    """
    df = _canon_dates(df)
    df = df.sort_values(["player", "game_date"], kind="mergesort").copy()

    g = df.groupby("player", sort=False)

    # trailing rolling means
    df["min_rolling_5"] = g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["min_rolling_10"] = g["mp_minutes"].shift(1).rolling(10).mean().reset_index(level=0, drop=True)

    df["ast_lag_1"] = g["ast"].shift(1)
    df["ast_rolling_5"] = g["ast"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["ast_rolling_10"] = g["ast"].shift(1).rolling(10).mean().reset_index(level=0, drop=True)

    if "tov" in df.columns:
        df["tov_rolling_5"] = g["tov"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    else:
        df["tov_rolling_5"] = np.nan

    if "pts" in df.columns:
        df["pts_rolling_5"] = g["pts"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    else:
        df["pts_rolling_5"] = np.nan

    # rate features
    ast_sum_5 = g["ast"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    ast_sum_10 = g["ast"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    min_sum_5 = g["mp_minutes"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    min_sum_10 = g["mp_minutes"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)

    df["ast_per_min_5"] = _safe_div(ast_sum_5, min_sum_5)
    df["ast_per_min_10"] = _safe_div(ast_sum_10, min_sum_10)

    # context
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

    df["player_ast_season_avg"] = (
        g["ast"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    df["player_min_season_avg"] = (
        g["mp_minutes"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    ast_sum = g["ast"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    min_sum = g["mp_minutes"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    df["player_ast_per_min_season"] = _safe_div(ast_sum, min_sum)

    if "tov" in df.columns:
        df["player_tov_season_avg"] = (
            g["tov"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_tov_season_avg"] = np.nan

    if "usage" in df.columns:
        df["player_usage_season"] = (
            g["usage"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_usage_season"] = np.nan

    df["ast_per_min_5"] = df["ast_per_min_5"].fillna(df["player_ast_per_min_season"])
    df["ast_per_min_10"] = df["ast_per_min_10"].fillna(df["player_ast_per_min_season"])
    df["ast_rolling_5"] = df["ast_rolling_5"].fillna(df["player_ast_season_avg"])
    df["ast_rolling_10"] = df["ast_rolling_10"].fillna(df["player_ast_season_avg"])

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

    agg_map = {}
    for col in ["pts", "ast", "fgm", "fga", "fg3m", "fg3", "fg3a"]:
        if col in df.columns:
            agg_map[col] = "sum"

    if not agg_map:
        return df

    team_game = (
        df.groupby(["season", "game_date", "team"], as_index=False)
        .agg(agg_map)
        .sort_values(["team", "season", "game_date"], kind="mergesort")
        .copy()
    )

    g = team_game.groupby(["team", "season"], sort=False)

    if "pts" in team_game.columns:
        team_game["team_pts_pg_to_date"] = (
            g["pts"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_pts_pg_to_date"] = np.nan

    if "ast" in team_game.columns:
        team_game["team_ast_pg_to_date"] = (
            g["ast"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_ast_pg_to_date"] = np.nan

    if {"fgm", "fga"}.issubset(team_game.columns):
        fgm_sum = g["fgm"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        fga_sum = g["fga"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        team_game["team_fg_pct_to_date"] = _safe_div(fgm_sum, fga_sum)
    else:
        team_game["team_fg_pct_to_date"] = np.nan

    fg3m_col = "fg3m" if "fg3m" in team_game.columns else ("fg3" if "fg3" in team_game.columns else None)
    if fg3m_col is not None and "fg3a" in team_game.columns:
        fg3m_sum = g[fg3m_col].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        fg3a_sum = g["fg3a"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        team_game["team_3p_pct_to_date"] = _safe_div(fg3m_sum, fg3a_sum)
    else:
        team_game["team_3p_pct_to_date"] = np.nan

    df = df.merge(
        team_game[
            [
                "season",
                "game_date",
                "team",
                "team_pts_pg_to_date",
                "team_ast_pg_to_date",
                "team_fg_pct_to_date",
                "team_3p_pct_to_date",
            ]
        ],
        on=["season", "game_date", "team"],
        how="left",
    )

    return df


def add_opp_ast_defense_to_date(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["season", "game_date", "team", "opp"], kind="mergesort").copy()

    agg_map = {}
    for col in ["pts", "ast", "fgm", "fga", "fg3m", "fg3", "fg3a"]:
        if col in df.columns:
            agg_map[col] = "sum"

    if not agg_map:
        return df

    team_game = (
        df.groupby(["season", "game_date", "team", "opp"], as_index=False)
        .agg(agg_map)
        .sort_values(["season", "game_date", "team", "opp"], kind="mergesort")
        .copy()
    )

    rename_map = {
        "opp": "def_team",
        "pts": "allowed_pts_game",
        "ast": "allowed_ast_game",
        "fgm": "allowed_fgm_game",
        "fga": "allowed_fga_game",
        "fg3a": "allowed_fg3a_game",
    }
    if "fg3m" in team_game.columns:
        rename_map["fg3m"] = "allowed_fg3m_game"
    elif "fg3" in team_game.columns:
        rename_map["fg3"] = "allowed_fg3m_game"

    allowed = team_game.rename(columns=rename_map)

    keep_cols = [
        "season",
        "game_date",
        "def_team",
        "allowed_pts_game",
        "allowed_ast_game",
        "allowed_fgm_game",
        "allowed_fga_game",
        "allowed_fg3m_game",
        "allowed_fg3a_game",
    ]
    keep_cols = [c for c in keep_cols if c in allowed.columns]
    allowed = allowed[keep_cols].copy()

    allowed = allowed.sort_values(["def_team", "season", "game_date"], kind="mergesort").copy()
    g = allowed.groupby(["def_team", "season"], sort=False)

    if "allowed_pts_game" in allowed.columns:
        allowed["opp_pts_allowed_pg_to_date"] = (
            g["allowed_pts_game"]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_pts_allowed_pg_to_date"] = np.nan

    if "allowed_ast_game" in allowed.columns:
        allowed["opp_ast_allowed_pg_to_date"] = (
            g["allowed_ast_game"]
            .apply(lambda s: s.shift(1).expanding().mean())
            .reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_ast_allowed_pg_to_date"] = np.nan

    if {"allowed_fgm_game", "allowed_fga_game"}.issubset(allowed.columns):
        fgm_sum = (
            g["allowed_fgm_game"]
            .apply(lambda s: s.shift(1).expanding().sum())
            .reset_index(level=[0, 1], drop=True)
        )
        fga_sum = (
            g["allowed_fga_game"]
            .apply(lambda s: s.shift(1).expanding().sum())
            .reset_index(level=[0, 1], drop=True)
        )
        allowed["opp_fg_pct_allowed_to_date"] = _safe_div(fgm_sum, fga_sum)
    else:
        allowed["opp_fg_pct_allowed_to_date"] = np.nan

    if {"allowed_fg3m_game", "allowed_fg3a_game"}.issubset(allowed.columns):
        fg3m_sum = (
            g["allowed_fg3m_game"]
            .apply(lambda s: s.shift(1).expanding().sum())
            .reset_index(level=[0, 1], drop=True)
        )
        fg3a_sum = (
            g["allowed_fg3a_game"]
            .apply(lambda s: s.shift(1).expanding().sum())
            .reset_index(level=[0, 1], drop=True)
        )
        allowed["opp_3p_pct_allowed_to_date"] = _safe_div(fg3m_sum, fg3a_sum)
    else:
        allowed["opp_3p_pct_allowed_to_date"] = np.nan

    if "opp_ast_allowed_pg_to_date" in allowed.columns:
        allowed["opp_def_ast_rank_to_date"] = (
            allowed.groupby(["season", "game_date"])["opp_ast_allowed_pg_to_date"]
            .rank(method="average", ascending=True)
        )
    else:
        allowed["opp_def_ast_rank_to_date"] = np.nan

    allowed = allowed.rename(columns={"def_team": "opp"})

    merge_cols = [
        "season",
        "game_date",
        "opp",
        "opp_pts_allowed_pg_to_date",
        "opp_ast_allowed_pg_to_date",
        "opp_fg_pct_allowed_to_date",
        "opp_3p_pct_allowed_to_date",
        "opp_def_ast_rank_to_date",
    ]
    merge_cols = [c for c in merge_cols if c in allowed.columns]

    df = df.merge(
        allowed[merge_cols],
        on=["season", "game_date", "opp"],
        how="left",
    )

    if "pts" in df.columns:
        df["opp_pts_allowed_pg_to_date"] = df["opp_pts_allowed_pg_to_date"].fillna(df["pts"].mean())
    else:
        df["opp_pts_allowed_pg_to_date"] = df["opp_pts_allowed_pg_to_date"].fillna(110.0)

    df["opp_ast_allowed_pg_to_date"] = df["opp_ast_allowed_pg_to_date"].fillna(df["ast"].mean())

    if {"fgm", "fga"}.issubset(df.columns):
        league_fg_pct = float(df["fgm"].sum() / max(df["fga"].sum(), 1))
    else:
        league_fg_pct = 0.47
    df["opp_fg_pct_allowed_to_date"] = df["opp_fg_pct_allowed_to_date"].fillna(league_fg_pct)

    fg3m_col = "fg3m" if "fg3m" in df.columns else ("fg3" if "fg3" in df.columns else None)
    if fg3m_col is not None and "fg3a" in df.columns:
        league_3p_pct = float(df[fg3m_col].sum() / max(df["fg3a"].sum(), 1))
    else:
        league_3p_pct = 0.36
    df["opp_3p_pct_allowed_to_date"] = df["opp_3p_pct_allowed_to_date"].fillna(league_3p_pct)

    if df["opp_def_ast_rank_to_date"].notna().any():
        df["opp_def_ast_rank_to_date"] = df["opp_def_ast_rank_to_date"].fillna(
            df["opp_def_ast_rank_to_date"].median()
        )
    else:
        df["opp_def_ast_rank_to_date"] = 15.0

    return df


def build_all_assists_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = build_features_no_leak(df)
    df = add_player_baselines(df)
    df = add_team_stint_features(df)
    df = add_team_context_to_date(df)
    df = add_opp_ast_defense_to_date(df)
    df["date"] = df["game_date"]
    return df