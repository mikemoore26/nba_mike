from __future__ import annotations

import numpy as np
import pandas as pd


REBOUND_FEATURES = [
    # role / opportunity
    "min_rolling_5",
    "reb_rolling_5",
    "reb_lag_1",
    "reb_per_min_5",
    "reb_per_min_10",

    # nonlinear minutes / stability / trend
    "min_rolling_10",
    "min_std_5",
    "min_std_10",
    "reb_rolling_10",
    "reb_std_5",
    "reb_std_10",
    "reb_per_min_3",
    "min_trend_3v10",
    "reb_trend_3v10",
    "reb_pm_trend_3v10",
    "starter_rate_5",
    "starter_rate_10",
    "minutes_bucket_code",
    "high_min_flag",
    "reb_pm_x_min",

    # NEW: ceiling features
    "reb_max_5",
    "reb_max_10",
    "reb_p80_10",

    # player baselines (season to date)
    "player_reb_season_avg",
    "player_min_season_avg",
    "player_reb_per_min_season",
    "player_orb_season_avg",
    "player_drb_season_avg",
    "player_usage_season",

    # context
    "home_game",
    "days_rest",
    "back_to_back",
    "starter_flag",
    "days_rest_capped",
    "days_rest_3plus",

    # stint / team change
    "games_played_to_date",
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",

    # team context
    "team_reb_pg_to_date",
    "team_orb_pg_to_date",
    "team_drb_pg_to_date",
    "team_fg_pct_to_date",
    "team_3p_pct_to_date",
    "team_missed_fg_pg_to_date",
    "team_missed_3fg_pg_to_date",

    # opponent context
    "opp_reb_allowed_pg_to_date",
    "opp_orb_allowed_pg_to_date",
    "opp_drb_allowed_pg_to_date",
    "opp_fg_pct_allowed_to_date",
    "opp_3p_pct_allowed_to_date",
    "opp_def_reb_rank_to_date",
    "opp_missed_fg_allowed_pg_to_date",
    "opp_missed_3fg_allowed_pg_to_date",

    # combined environment
    "game_missed_fg_env_to_date",
    "game_missed_3fg_env_to_date",
    "reb_env_interaction",

    # teammate rebound competition
    "teammate_top1_rebpm_10",
    "teammate_top2_rebpm_sum_10",
    "teammate_top3_rebpm_sum_10",
    "reb_share_proxy",

    # NEW: stronger interactions
    "high_min_rebpm_interaction",
    "high_min_env_interaction",
    "reb_share_x_min",
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
    Trailing/rolling features using shift(1).
    Works on combined history + appended today rows as long as feature inputs
    are ordered by player/game_date.
    """
    df = _canon_dates(df)
    df = df.sort_values(["player", "game_date"], kind="mergesort").copy()

    g = df.groupby("player", sort=False)

    # core rolling
    df["min_rolling_5"] = g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["min_rolling_10"] = g["mp_minutes"].shift(1).rolling(10).mean().reset_index(level=0, drop=True)
    df["min_rolling_3"] = g["mp_minutes"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)

    df["min_std_5"] = g["mp_minutes"].shift(1).rolling(5).std().reset_index(level=0, drop=True)
    df["min_std_10"] = g["mp_minutes"].shift(1).rolling(10).std().reset_index(level=0, drop=True)

    df["reb_lag_1"] = g["reb"].shift(1)
    df["reb_rolling_5"] = g["reb"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["reb_rolling_10"] = g["reb"].shift(1).rolling(10).mean().reset_index(level=0, drop=True)
    df["reb_rolling_3"] = g["reb"].shift(1).rolling(3).mean().reset_index(level=0, drop=True)

    df["reb_std_5"] = g["reb"].shift(1).rolling(5).std().reset_index(level=0, drop=True)
    df["reb_std_10"] = g["reb"].shift(1).rolling(10).std().reset_index(level=0, drop=True)

    # NEW: ceiling memory
    df["reb_max_5"] = g["reb"].shift(1).rolling(5).max().reset_index(level=0, drop=True)
    df["reb_max_10"] = g["reb"].shift(1).rolling(10).max().reset_index(level=0, drop=True)
    df["reb_p80_10"] = (
        g["reb"]
        .shift(1)
        .rolling(10)
        .quantile(0.8)
        .reset_index(level=0, drop=True)
    )

    if "orb" in df.columns:
        df["orb_rolling_5"] = g["orb"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    else:
        df["orb_rolling_5"] = np.nan

    if "drb" in df.columns:
        df["drb_rolling_5"] = g["drb"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    else:
        df["drb_rolling_5"] = np.nan

    # rate features
    reb_sum_3 = g["reb"].shift(1).rolling(3).sum().reset_index(level=0, drop=True)
    reb_sum_5 = g["reb"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    reb_sum_10 = g["reb"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)

    min_sum_3 = g["mp_minutes"].shift(1).rolling(3).sum().reset_index(level=0, drop=True)
    min_sum_5 = g["mp_minutes"].shift(1).rolling(5).sum().reset_index(level=0, drop=True)
    min_sum_10 = g["mp_minutes"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)

    df["reb_per_min_3"] = _safe_div(reb_sum_3, min_sum_3)
    df["reb_per_min_5"] = _safe_div(reb_sum_5, min_sum_5)
    df["reb_per_min_10"] = _safe_div(reb_sum_10, min_sum_10)

    # context
    df["days_rest"] = g["game_date"].diff().dt.days.reset_index(level=0, drop=True)
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)
    df["home_game"] = pd.to_numeric(df["is_home"], errors="coerce").fillna(0).astype(int)

    df["days_rest_capped"] = df["days_rest"].clip(lower=0, upper=4)
    df["days_rest_3plus"] = (df["days_rest"] >= 3).astype(int)

    if "starter_flag" not in df.columns:
        df["starter_flag"] = 0

    starter_shift = g["starter_flag"].shift(1)
    df["starter_rate_5"] = starter_shift.rolling(5).mean().reset_index(level=0, drop=True)
    df["starter_rate_10"] = starter_shift.rolling(10).mean().reset_index(level=0, drop=True)

    # role / form trend
    df["min_trend_3v10"] = df["min_rolling_3"] - df["min_rolling_10"]
    df["reb_trend_3v10"] = df["reb_rolling_3"] - df["reb_rolling_10"]
    df["reb_pm_trend_3v10"] = df["reb_per_min_3"] - df["reb_per_min_10"]

    # nonlinear minute structure
    df["minutes_bucket_code"] = pd.cut(
        df["min_rolling_5"],
        bins=[0, 12, 20, 28, 36, np.inf],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(float)

    df["high_min_flag"] = (df["min_rolling_5"] >= 30).astype(int)
    df["reb_pm_x_min"] = df["reb_per_min_5"] * df["min_rolling_5"]

    return df


def add_player_baselines(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["player", "season", "game_date"], kind="mergesort").copy()
    g = df.groupby(["player", "season"], sort=False)

    df["player_reb_season_avg"] = (
        g["reb"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    df["player_min_season_avg"] = (
        g["mp_minutes"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
    )

    reb_sum = g["reb"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    min_sum = g["mp_minutes"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
    df["player_reb_per_min_season"] = _safe_div(reb_sum, min_sum)

    if "orb" in df.columns:
        df["player_orb_season_avg"] = (
            g["orb"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_orb_season_avg"] = np.nan

    if "drb" in df.columns:
        df["player_drb_season_avg"] = (
            g["drb"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_drb_season_avg"] = np.nan

    if "usage" in df.columns:
        df["player_usage_season"] = (
            g["usage"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        df["player_usage_season"] = np.nan

    df["reb_per_min_5"] = df["reb_per_min_5"].fillna(df["player_reb_per_min_season"])
    df["reb_per_min_10"] = df["reb_per_min_10"].fillna(df["player_reb_per_min_season"])
    df["reb_rolling_5"] = df["reb_rolling_5"].fillna(df["player_reb_season_avg"])
    df["reb_rolling_10"] = df["reb_rolling_10"].fillna(df["player_reb_season_avg"])
    df["reb_per_min_3"] = df["reb_per_min_3"].fillna(df["player_reb_per_min_season"])
    df["reb_max_5"] = df["reb_max_5"].fillna(df["player_reb_season_avg"])
    df["reb_max_10"] = df["reb_max_10"].fillna(df["player_reb_season_avg"])
    df["reb_p80_10"] = df["reb_p80_10"].fillna(df["player_reb_season_avg"])

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
    for col in ["reb", "orb", "drb", "fgm", "fga", "fg3m", "fg3", "fg3a"]:
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

    if {"fga", "fgm"}.issubset(team_game.columns):
        team_game["missed_fg_game"] = team_game["fga"] - team_game["fgm"]
    else:
        team_game["missed_fg_game"] = np.nan

    fg3m_col = "fg3m" if "fg3m" in team_game.columns else ("fg3" if "fg3" in team_game.columns else None)
    if fg3m_col is not None and "fg3a" in team_game.columns:
        team_game["missed_3fg_game"] = team_game["fg3a"] - team_game[fg3m_col]
    else:
        team_game["missed_3fg_game"] = np.nan

    g = team_game.groupby(["team", "season"], sort=False)

    if "reb" in team_game.columns:
        team_game["team_reb_pg_to_date"] = (
            g["reb"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_reb_pg_to_date"] = np.nan

    if "orb" in team_game.columns:
        team_game["team_orb_pg_to_date"] = (
            g["orb"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_orb_pg_to_date"] = np.nan

    if "drb" in team_game.columns:
        team_game["team_drb_pg_to_date"] = (
            g["drb"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_drb_pg_to_date"] = np.nan

    if {"fgm", "fga"}.issubset(team_game.columns):
        fgm_sum = g["fgm"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        fga_sum = g["fga"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        team_game["team_fg_pct_to_date"] = _safe_div(fgm_sum, fga_sum)
    else:
        team_game["team_fg_pct_to_date"] = np.nan

    if fg3m_col is not None and "fg3a" in team_game.columns:
        fg3m_sum = g[fg3m_col].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        fg3a_sum = g["fg3a"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        team_game["team_3p_pct_to_date"] = _safe_div(fg3m_sum, fg3a_sum)
    else:
        team_game["team_3p_pct_to_date"] = np.nan

    if "missed_fg_game" in team_game.columns:
        team_game["team_missed_fg_pg_to_date"] = (
            g["missed_fg_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_missed_fg_pg_to_date"] = np.nan

    if "missed_3fg_game" in team_game.columns:
        team_game["team_missed_3fg_pg_to_date"] = (
            g["missed_3fg_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        team_game["team_missed_3fg_pg_to_date"] = np.nan

    df = df.merge(
        team_game[
            [
                "season",
                "game_date",
                "team",
                "team_reb_pg_to_date",
                "team_orb_pg_to_date",
                "team_drb_pg_to_date",
                "team_fg_pct_to_date",
                "team_3p_pct_to_date",
                "team_missed_fg_pg_to_date",
                "team_missed_3fg_pg_to_date",
            ]
        ],
        on=["season", "game_date", "team"],
        how="left",
    )

    return df


def add_opp_reb_context_to_date(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["season", "game_date", "team", "opp"], kind="mergesort").copy()

    agg_map = {}
    for col in ["reb", "orb", "drb", "fgm", "fga", "fg3m", "fg3", "fg3a"]:
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
        "reb": "allowed_reb_game",
        "orb": "allowed_orb_game",
        "drb": "allowed_drb_game",
        "fgm": "allowed_fgm_game",
        "fga": "allowed_fga_game",
        "fg3a": "allowed_fg3a_game",
    }
    if "fg3m" in team_game.columns:
        rename_map["fg3m"] = "allowed_fg3m_game"
    elif "fg3" in team_game.columns:
        rename_map["fg3"] = "allowed_fg3m_game"

    allowed = team_game.rename(columns=rename_map)

    if {"allowed_fga_game", "allowed_fgm_game"}.issubset(allowed.columns):
        allowed["allowed_missed_fg_game"] = allowed["allowed_fga_game"] - allowed["allowed_fgm_game"]
    else:
        allowed["allowed_missed_fg_game"] = np.nan

    if {"allowed_fg3a_game", "allowed_fg3m_game"}.issubset(allowed.columns):
        allowed["allowed_missed_3fg_game"] = allowed["allowed_fg3a_game"] - allowed["allowed_fg3m_game"]
    else:
        allowed["allowed_missed_3fg_game"] = np.nan

    keep_cols = [
        "season",
        "game_date",
        "def_team",
        "allowed_reb_game",
        "allowed_orb_game",
        "allowed_drb_game",
        "allowed_fgm_game",
        "allowed_fga_game",
        "allowed_fg3m_game",
        "allowed_fg3a_game",
        "allowed_missed_fg_game",
        "allowed_missed_3fg_game",
    ]
    keep_cols = [c for c in keep_cols if c in allowed.columns]
    allowed = allowed[keep_cols].copy()

    allowed = allowed.sort_values(["def_team", "season", "game_date"], kind="mergesort").copy()
    g = allowed.groupby(["def_team", "season"], sort=False)

    if "allowed_reb_game" in allowed.columns:
        allowed["opp_reb_allowed_pg_to_date"] = (
            g["allowed_reb_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_reb_allowed_pg_to_date"] = np.nan

    if "allowed_orb_game" in allowed.columns:
        allowed["opp_orb_allowed_pg_to_date"] = (
            g["allowed_orb_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_orb_allowed_pg_to_date"] = np.nan

    if "allowed_drb_game" in allowed.columns:
        allowed["opp_drb_allowed_pg_to_date"] = (
            g["allowed_drb_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_drb_allowed_pg_to_date"] = np.nan

    if {"allowed_fgm_game", "allowed_fga_game"}.issubset(allowed.columns):
        fgm_sum = g["allowed_fgm_game"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        fga_sum = g["allowed_fga_game"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        allowed["opp_fg_pct_allowed_to_date"] = _safe_div(fgm_sum, fga_sum)
    else:
        allowed["opp_fg_pct_allowed_to_date"] = np.nan

    if {"allowed_fg3m_game", "allowed_fg3a_game"}.issubset(allowed.columns):
        fg3m_sum = g["allowed_fg3m_game"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        fg3a_sum = g["allowed_fg3a_game"].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=[0, 1], drop=True)
        allowed["opp_3p_pct_allowed_to_date"] = _safe_div(fg3m_sum, fg3a_sum)
    else:
        allowed["opp_3p_pct_allowed_to_date"] = np.nan

    if "allowed_missed_fg_game" in allowed.columns:
        allowed["opp_missed_fg_allowed_pg_to_date"] = (
            g["allowed_missed_fg_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_missed_fg_allowed_pg_to_date"] = np.nan

    if "allowed_missed_3fg_game" in allowed.columns:
        allowed["opp_missed_3fg_allowed_pg_to_date"] = (
            g["allowed_missed_3fg_game"].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=[0, 1], drop=True)
        )
    else:
        allowed["opp_missed_3fg_allowed_pg_to_date"] = np.nan

    if "opp_reb_allowed_pg_to_date" in allowed.columns:
        allowed["opp_def_reb_rank_to_date"] = (
            allowed.groupby(["season", "game_date"])["opp_reb_allowed_pg_to_date"]
            .rank(method="average", ascending=True)
        )
    else:
        allowed["opp_def_reb_rank_to_date"] = np.nan

    allowed = allowed.rename(columns={"def_team": "opp"})

    merge_cols = [
        "season",
        "game_date",
        "opp",
        "opp_reb_allowed_pg_to_date",
        "opp_orb_allowed_pg_to_date",
        "opp_drb_allowed_pg_to_date",
        "opp_fg_pct_allowed_to_date",
        "opp_3p_pct_allowed_to_date",
        "opp_def_reb_rank_to_date",
        "opp_missed_fg_allowed_pg_to_date",
        "opp_missed_3fg_allowed_pg_to_date",
    ]
    merge_cols = [c for c in merge_cols if c in allowed.columns]

    df = df.merge(allowed[merge_cols], on=["season", "game_date", "opp"], how="left")

    df["opp_reb_allowed_pg_to_date"] = df["opp_reb_allowed_pg_to_date"].fillna(df["reb"].mean())

    if "orb" in df.columns:
        df["opp_orb_allowed_pg_to_date"] = df["opp_orb_allowed_pg_to_date"].fillna(df["orb"].mean())
    else:
        df["opp_orb_allowed_pg_to_date"] = df["opp_orb_allowed_pg_to_date"].fillna(3.0)

    if "drb" in df.columns:
        df["opp_drb_allowed_pg_to_date"] = df["opp_drb_allowed_pg_to_date"].fillna(df["drb"].mean())
    else:
        df["opp_drb_allowed_pg_to_date"] = df["opp_drb_allowed_pg_to_date"].fillna(7.0)

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

    if "opp_missed_fg_allowed_pg_to_date" in df.columns:
        df["opp_missed_fg_allowed_pg_to_date"] = df["opp_missed_fg_allowed_pg_to_date"].fillna(
            df["opp_reb_allowed_pg_to_date"]
        )

    if "opp_missed_3fg_allowed_pg_to_date" in df.columns:
        df["opp_missed_3fg_allowed_pg_to_date"] = df["opp_missed_3fg_allowed_pg_to_date"].fillna(
            df["opp_missed_3fg_allowed_pg_to_date"].median()
            if df["opp_missed_3fg_allowed_pg_to_date"].notna().any()
            else 12.0
        )

    if df["opp_def_reb_rank_to_date"].notna().any():
        df["opp_def_reb_rank_to_date"] = df["opp_def_reb_rank_to_date"].fillna(
            df["opp_def_reb_rank_to_date"].median()
        )
    else:
        df["opp_def_reb_rank_to_date"] = 15.0

    if "team_missed_fg_pg_to_date" in df.columns and "opp_missed_fg_allowed_pg_to_date" in df.columns:
        df["game_missed_fg_env_to_date"] = (
            df["team_missed_fg_pg_to_date"] + df["opp_missed_fg_allowed_pg_to_date"]
        )
    else:
        df["game_missed_fg_env_to_date"] = np.nan

    if "team_missed_3fg_pg_to_date" in df.columns and "opp_missed_3fg_allowed_pg_to_date" in df.columns:
        df["game_missed_3fg_env_to_date"] = (
            df["team_missed_3fg_pg_to_date"] + df["opp_missed_3fg_allowed_pg_to_date"]
        )
    else:
        df["game_missed_3fg_env_to_date"] = np.nan

    return df


def add_teammate_rebound_competition(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = df.sort_values(["season", "game_date", "team", "player"], kind="mergesort").copy()

    if "reb_per_min_10" not in df.columns:
        df["teammate_top1_rebpm_10"] = np.nan
        df["teammate_top2_rebpm_sum_10"] = np.nan
        df["teammate_top3_rebpm_sum_10"] = np.nan
        df["reb_share_proxy"] = np.nan
        return df

    out_frames = []

    for _, sub in df.groupby(["season", "game_date", "team"], sort=False):
        sub = sub.copy()
        vals = sub["reb_per_min_10"].fillna(-9999.0).to_numpy()

        top1_list = []
        top2_sum_list = []
        top3_sum_list = []

        for i in range(len(sub)):
            others = np.delete(vals, i)
            others = others[others > -9999.0]

            if len(others) == 0:
                top1_list.append(np.nan)
                top2_sum_list.append(np.nan)
                top3_sum_list.append(np.nan)
                continue

            others_sorted = np.sort(others)[::-1]

            top1_list.append(float(others_sorted[:1].sum()) if len(others_sorted) >= 1 else np.nan)
            top2_sum_list.append(float(others_sorted[:2].sum()) if len(others_sorted) >= 2 else float(others_sorted.sum()))
            top3_sum_list.append(float(others_sorted[:3].sum()) if len(others_sorted) >= 3 else float(others_sorted.sum()))

        sub["teammate_top1_rebpm_10"] = top1_list
        sub["teammate_top2_rebpm_sum_10"] = top2_sum_list
        sub["teammate_top3_rebpm_sum_10"] = top3_sum_list

        out_frames.append(sub)

    df = pd.concat(out_frames, axis=0).sort_index()

    if df["teammate_top1_rebpm_10"].notna().any():
        df["teammate_top1_rebpm_10"] = df["teammate_top1_rebpm_10"].fillna(df["teammate_top1_rebpm_10"].median())
    if df["teammate_top2_rebpm_sum_10"].notna().any():
        df["teammate_top2_rebpm_sum_10"] = df["teammate_top2_rebpm_sum_10"].fillna(df["teammate_top2_rebpm_sum_10"].median())
    if df["teammate_top3_rebpm_10"].notna().any() if "teammate_top3_rebpm_10" in df.columns else False:
        pass
    if df["teammate_top3_rebpm_sum_10"].notna().any():
        df["teammate_top3_rebpm_sum_10"] = df["teammate_top3_rebpm_sum_10"].fillna(df["teammate_top3_rebpm_sum_10"].median())

    df["reb_share_proxy"] = _safe_div(
        df["reb_per_min_5"],
        df["teammate_top2_rebpm_sum_10"] + 1e-6,
    )

    return df


def add_rebound_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"reb_per_min_5", "game_missed_fg_env_to_date"}.issubset(df.columns):
        df["reb_env_interaction"] = df["reb_per_min_5"] * df["game_missed_fg_env_to_date"]
    else:
        df["reb_env_interaction"] = np.nan

    if {"high_min_flag", "reb_per_min_10"}.issubset(df.columns):
        df["high_min_rebpm_interaction"] = df["high_min_flag"] * df["reb_per_min_10"]
    else:
        df["high_min_rebpm_interaction"] = np.nan

    if {"high_min_flag", "game_missed_fg_env_to_date"}.issubset(df.columns):
        df["high_min_env_interaction"] = df["high_min_flag"] * df["game_missed_fg_env_to_date"]
    else:
        df["high_min_env_interaction"] = np.nan

    if {"reb_share_proxy", "min_rolling_5"}.issubset(df.columns):
        df["reb_share_x_min"] = df["reb_share_proxy"] * df["min_rolling_5"]
    else:
        df["reb_share_x_min"] = np.nan

    return df


def build_all_rebounds_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _canon_dates(df)
    df = build_features_no_leak(df)
    df = add_player_baselines(df)
    df = add_team_stint_features(df)
    df = add_team_context_to_date(df)
    df = add_opp_reb_context_to_date(df)
    df = add_teammate_rebound_competition(df)
    df = add_rebound_interactions(df)
    df["date"] = df["game_date"]
    return df