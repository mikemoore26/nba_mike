# model_training/points/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def played_mask(df: pd.DataFrame, *, mp_col: str = "mp_minutes") -> pd.Series:
    mm = pd.to_numeric(df[mp_col], errors="coerce")
    return mm.notna() & (mm > 0)


# =============================================================================
# Feature lists (single source of truth)
# =============================================================================

FG2A_FEATURES = [
    # recent form (volume + role)
    "min_rolling_5",
    "fga_rolling_5",
    "fg2a_rolling_5",
    "expected_min_10",
    "min_share_10",
    # baselines (identity)
    "player_fg2a_season_avg",
    "player_min_season_avg",
    "player_usage_season",
    # change signals (role shift)
    "fg2a_delta_5",
    "min_delta_5",
    # context (pregame-known)
    "home_game",
    "days_rest",
    "back_to_back",
    # team context (leakage-safe)
    "team_fga_pg_to_date",
    # interaction (KEEP BOTH; models may expect either)
    "usage_x_min",
    "usage_x_min4",
    # backward compat
    "starter_prob_10",
]

FG2_RATE_FEATURES = [
    # recent form (accuracy) + confidence
    "fg2_pct_rolling_10",
    "fg2_att_rolling_10",
    # baseline (stabilized)
    "player_fg2_pct_season",
    # role proxy
    "expected_min_10",
    "min_share_10",
    # context
    "home_game",
    "days_rest",
    "back_to_back",
    # interaction
    "usage_x_min",
    "usage_x_min4",
    # backward compat
    "starter_prob_10",
]

FTA_FEATURES = [
    "min_rolling_5",
    "fta_rolling_5",
    "expected_min_10",
    "min_share_10",
    "player_fta_season_avg",
    "player_min_season_avg",
    "player_usage_season",
    "fta_delta_5",
    "min_delta_5",
    "home_game",
    "days_rest",
    "back_to_back",
    # team context (leakage-safe)
    "team_fta_pg_to_date",
    # interaction
    "usage_x_min",
    "usage_x_min4",
    "starter_prob_10",
]

FT_RATE_FEATURES = [
    "ft_pct_rolling_15",
    "ft_att_rolling_15",
    "player_ft_pct_season",
    "expected_min_10",
    "min_share_10",
    "home_game",
    "days_rest",
    "back_to_back",
    "usage_x_min",
    "usage_x_min4",
    "starter_prob_10",
]

# --- Opponent "to date" defense features (2P + FT) ---
OPP_2P_DEF_FEATURES = [
    "games_played_to_date_2p",
    "opp_fg2a_allowed_pg_to_date",
    "opp_fg2m_allowed_pg_to_date",
    "opp_2p_pct_allowed_to_date",
    "opp_def_2p_rank_to_date",
]

OPP_FT_DEF_FEATURES = [
    "games_played_to_date_ft",
    "opp_fta_allowed_pg_to_date",
    "opp_ftm_allowed_pg_to_date",
    "opp_ft_pct_allowed_to_date",
    "opp_def_ft_rank_to_date",
]

TEAM_STINT_FEATURES = [
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",
]

# Extend feature sets
FG2A_FEATURES = FG2A_FEATURES + OPP_2P_DEF_FEATURES + TEAM_STINT_FEATURES
FG2_RATE_FEATURES = FG2_RATE_FEATURES + OPP_2P_DEF_FEATURES + TEAM_STINT_FEATURES
FTA_FEATURES = FTA_FEATURES + OPP_FT_DEF_FEATURES + TEAM_STINT_FEATURES
FT_RATE_FEATURES = FT_RATE_FEATURES + OPP_FT_DEF_FEATURES + TEAM_STINT_FEATURES


# =============================================================================
# Helper: derive 2PT columns
# =============================================================================
def add_derived_2pt_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive 2PT attempts/makes from boxscore:
      fg2a = fga - fg3a
      fg2m = fg  - fg3
    """
    out = df.copy()
    for c in ["fg", "fga", "fg3", "fg3a", "ft", "fta"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["fg2a"] = (out["fga"] - out["fg3a"]).clip(lower=0)
    out["fg2m"] = (out["fg"] - out["fg3"]).clip(lower=0)
    return out


# =============================================================================
# Team "to date" pace/context (LEAKAGE SAFE)
# =============================================================================
def add_team_to_date_features(
    df: pd.DataFrame,
    *,
    min_games_for_rate: int = 1,
) -> pd.DataFrame:
    """
    Leakage-safe TEAM offensive context to-date (excluding current game).

    Produces:
      - team_games_played_to_date
      - team_fga_pg_to_date
      - team_fta_pg_to_date

    Requires: season, team, date, fga, fta
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    for c in ["season", "team", "fga", "fta"]:
        if c not in out.columns:
            raise ValueError(f"add_team_to_date_features missing required column: {c}")

    out = out.dropna(subset=["date"]).copy()
    out["fga"] = pd.to_numeric(out["fga"], errors="coerce")
    out["fta"] = pd.to_numeric(out["fta"], errors="coerce")

    team_game = (
        out.groupby(["season", "team", "date"], as_index=False)
           .agg(team_fga=("fga", "sum"), team_fta=("fta", "sum"))
           .sort_values(["season", "team", "date"])
    )

    g = team_game.groupby(["season", "team"], sort=False)

    team_game["team_games_played_to_date"] = g.cumcount()
    team_game["team_fga_to_date"] = g["team_fga"].cumsum() - team_game["team_fga"]
    team_game["team_fta_to_date"] = g["team_fta"].cumsum() - team_game["team_fta"]

    denom = team_game["team_games_played_to_date"].replace(0, np.nan)
    team_game["team_fga_pg_to_date"] = team_game["team_fga_to_date"] / denom
    team_game["team_fta_pg_to_date"] = team_game["team_fta_to_date"] / denom

    if min_games_for_rate is not None:
        early = team_game["team_games_played_to_date"] < int(min_games_for_rate)
        team_game.loc[early, ["team_fga_pg_to_date", "team_fta_pg_to_date"]] = np.nan

    feat_cols = [
        "season", "team", "date",
        "team_games_played_to_date",
        "team_fga_pg_to_date",
        "team_fta_pg_to_date",
    ]
    return out.merge(team_game[feat_cols], on=["season", "team", "date"], how="left")


# =============================================================================
# Opponent defense features (leakage-safe, to-date excluding current game)
# =============================================================================
def add_opp_2p_defense_features_roll(
    df: pd.DataFrame,
    *,
    min_games_for_rank: int = 5,
    fill_early_rank: float | None = None,
) -> pd.DataFrame:
    data = add_derived_2pt_cols(df).copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).copy()

    opp_game = (
        data.groupby(["season", "opp", "date"], as_index=False)
            .agg(fg2a_allowed=("fg2a", "sum"), fg2m_allowed=("fg2m", "sum"))
            .sort_values(["season", "opp", "date"])
    )

    g = opp_game.groupby(["season", "opp"], sort=False)
    opp_game["games_played_to_date_2p"] = g.cumcount()
    opp_game["fg2a_allowed_to_date"] = g["fg2a_allowed"].cumsum() - opp_game["fg2a_allowed"]
    opp_game["fg2m_allowed_to_date"] = g["fg2m_allowed"].cumsum() - opp_game["fg2m_allowed"]

    denom_games = opp_game["games_played_to_date_2p"].replace(0, np.nan)
    opp_game["opp_fg2a_allowed_pg_to_date"] = opp_game["fg2a_allowed_to_date"] / denom_games
    opp_game["opp_fg2m_allowed_pg_to_date"] = opp_game["fg2m_allowed_to_date"] / denom_games
    opp_game["opp_2p_pct_allowed_to_date"] = (
        opp_game["fg2m_allowed_to_date"] /
        opp_game["fg2a_allowed_to_date"].replace(0, np.nan)
    )

    eligible = opp_game["games_played_to_date_2p"] >= min_games_for_rank
    opp_game["opp_def_2p_rank_to_date"] = np.nan
    opp_game.loc[eligible, "opp_def_2p_rank_to_date"] = (
        opp_game.loc[eligible]
            .groupby(["season", "date"])["opp_fg2m_allowed_pg_to_date"]
            .rank(method="min", ascending=True)
    )

    if fill_early_rank is not None:
        opp_game["opp_def_2p_rank_to_date"] = opp_game["opp_def_2p_rank_to_date"].fillna(fill_early_rank)

    feat_cols = [
        "season", "opp", "date",
        "games_played_to_date_2p",
        "opp_fg2a_allowed_pg_to_date",
        "opp_fg2m_allowed_pg_to_date",
        "opp_2p_pct_allowed_to_date",
        "opp_def_2p_rank_to_date",
    ]
    return data.merge(opp_game[feat_cols], on=["season", "opp", "date"], how="left")


def add_opp_ft_defense_features_roll(
    df: pd.DataFrame,
    *,
    min_games_for_rank: int = 5,
    fill_early_rank: float | None = None,
) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).copy()

    data["fta"] = pd.to_numeric(data["fta"], errors="coerce")
    data["ft"] = pd.to_numeric(data["ft"], errors="coerce")

    opp_game = (
        data.groupby(["season", "opp", "date"], as_index=False)
            .agg(fta_allowed=("fta", "sum"), ftm_allowed=("ft", "sum"))
            .sort_values(["season", "opp", "date"])
    )

    g = opp_game.groupby(["season", "opp"], sort=False)
    opp_game["games_played_to_date_ft"] = g.cumcount()
    opp_game["fta_allowed_to_date"] = g["fta_allowed"].cumsum() - opp_game["fta_allowed"]
    opp_game["ftm_allowed_to_date"] = g["ftm_allowed"].cumsum() - opp_game["ftm_allowed"]

    denom_games = opp_game["games_played_to_date_ft"].replace(0, np.nan)
    opp_game["opp_fta_allowed_pg_to_date"] = opp_game["fta_allowed_to_date"] / denom_games
    opp_game["opp_ftm_allowed_pg_to_date"] = opp_game["ftm_allowed_to_date"] / denom_games
    opp_game["opp_ft_pct_allowed_to_date"] = (
        opp_game["ftm_allowed_to_date"] /
        opp_game["fta_allowed_to_date"].replace(0, np.nan)
    )

    eligible = opp_game["games_played_to_date_ft"] >= min_games_for_rank
    opp_game["opp_def_ft_rank_to_date"] = np.nan
    opp_game.loc[eligible, "opp_def_ft_rank_to_date"] = (
        opp_game.loc[eligible]
            .groupby(["season", "date"])["opp_ftm_allowed_pg_to_date"]
            .rank(method="min", ascending=True)
    )

    if fill_early_rank is not None:
        opp_game["opp_def_ft_rank_to_date"] = opp_game["opp_def_ft_rank_to_date"].fillna(fill_early_rank)

    feat_cols = [
        "season", "opp", "date",
        "games_played_to_date_ft",
        "opp_fta_allowed_pg_to_date",
        "opp_ftm_allowed_pg_to_date",
        "opp_ft_pct_allowed_to_date",
        "opp_def_ft_rank_to_date",
    ]
    return data.merge(opp_game[feat_cols], on=["season", "opp", "date"], how="left")


# =============================================================================
# No-leak feature builder (rolling / context)
# =============================================================================
def build_points_features_no_leak(
    df: pd.DataFrame,
    *,
    roll5: int = 5,
    roll10: int = 10,
    roll15: int = 15,
    rest_fill: int = 4,
    rest_cap: int = 7,
    fg2_pct_prior: float = 0.52,
    fg2_pct_prior_att: float = 120.0,
    ft_pct_prior: float = 0.78,
    ft_pct_prior_att: float = 80.0,
) -> pd.DataFrame:
    """
    Past-only rolling features for points components.
    """
    out = add_derived_2pt_cols(df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    out = out.sort_values(["player", "date"]).copy()
    g = out.groupby("player", sort=False)

    # rolling role/volume (PAST ONLY)
    out["min_rolling_5"] = g["mp_minutes"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["fga_rolling_5"] = g["fga"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["fg2a_rolling_5"] = g["fg2a"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)
    out["fta_rolling_5"] = g["fta"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)

    # FG2% rolling (sum-based + prior)
    fg2m10 = g["fg2m"].shift(1).rolling(roll10, min_periods=1).sum().reset_index(level=0, drop=True)
    fg2a10 = g["fg2a"].shift(1).rolling(roll10, min_periods=1).sum().reset_index(level=0, drop=True)
    out["fg2_att_rolling_10"] = fg2a10
    prior_fg2m = fg2_pct_prior * fg2_pct_prior_att
    out["fg2_pct_rolling_10"] = (fg2m10 + prior_fg2m) / (fg2a10 + fg2_pct_prior_att)

    # FT% rolling (sum-based + prior)
    ftm15 = g["ft"].shift(1).rolling(roll15, min_periods=1).sum().reset_index(level=0, drop=True)
    fta15 = g["fta"].shift(1).rolling(roll15, min_periods=1).sum().reset_index(level=0, drop=True)
    out["ft_att_rolling_15"] = fta15
    prior_ftm = ft_pct_prior * ft_pct_prior_att
    out["ft_pct_rolling_15"] = (ftm15 + prior_ftm) / (fta15 + ft_pct_prior_att)

    # rest
    days_rest = g["date"].diff().dt.days.reset_index(level=0, drop=True)
    out["days_rest"] = days_rest.clip(lower=0, upper=rest_cap).fillna(rest_fill)
    out["back_to_back"] = (out["days_rest"] == 1).astype(int)

    # home
    out["home_game"] = out["is_home"].astype(int)

    # expected minutes (PAST ONLY)
    out["expected_min_10"] = g["mp_minutes"].shift(1).rolling(roll10, min_periods=1).mean().reset_index(level=0, drop=True)
    out["min_share_10"] = (out["expected_min_10"] / 36.0).clip(0, 1)
    out["starter_prob_10"] = out["min_share_10"]  # backward compat

    # interaction (KEEP BOTH NAMES)
    usage_num = pd.to_numeric(out.get("usage", 0.0), errors="coerce").fillna(0.0)
    exp_min = pd.to_numeric(out["expected_min_10"], errors="coerce").fillna(0.0)
    out["usage_x_min"] = (usage_num * exp_min).astype(float)
    out["usage_x_min4"] = out["usage_x_min"]

    # TEAM to-date context (LEAKAGE SAFE; exclude current game)
    out = add_team_to_date_features(out)

    return out


# =============================================================================
# Player baselines (season-to-date, past-only)
# =============================================================================
def add_player_baselines_points(
    df: pd.DataFrame,
    *,
    fg2_pct_prior: float = 0.52,
    fg2_pct_prior_att: float = 300.0,
    ft_pct_prior: float = 0.78,
    ft_pct_prior_att: float = 200.0,
) -> pd.DataFrame:
    out = add_derived_2pt_cols(df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    out = out.sort_values(["player", "season", "date"]).copy()
    g = out.groupby(["player", "season"], sort=False)

    # FG2A baseline
    out["player_fg2a_season_avg"] = g["fg2a"].transform(lambda s: s.shift(1).expanding().mean())

    # FG2% baseline (sum-based + prior)
    fg2m_sum = g["fg2m"].transform(lambda s: s.shift(1).expanding().sum())
    fg2a_sum = g["fg2a"].transform(lambda s: s.shift(1).expanding().sum())
    prior_fg2m = fg2_pct_prior * fg2_pct_prior_att
    out["player_fg2_pct_season"] = (fg2m_sum + prior_fg2m) / (fg2a_sum + fg2_pct_prior_att)

    # FTA baseline
    out["player_fta_season_avg"] = g["fta"].transform(lambda s: s.shift(1).expanding().mean())

    # FT% baseline (sum-based + prior)
    ftm_sum = g["ft"].transform(lambda s: s.shift(1).expanding().sum())
    fta_sum = g["fta"].transform(lambda s: s.shift(1).expanding().sum())
    prior_ftm = ft_pct_prior * ft_pct_prior_att
    out["player_ft_pct_season"] = (ftm_sum + prior_ftm) / (fta_sum + ft_pct_prior_att)

    # role baselines
    out["player_min_season_avg"] = g["mp_minutes"].transform(lambda s: s.shift(1).expanding().mean())
    out["player_usage_season"] = g["usage"].transform(lambda s: s.shift(1).expanding().mean())

    # deltas (need rolling already computed upstream; safe if NaN)
    if "fg2a_rolling_5" in out.columns:
        out["fg2a_delta_5"] = out["fg2a_rolling_5"] - out["player_fg2a_season_avg"]
    if "fta_rolling_5" in out.columns:
        out["fta_delta_5"] = out["fta_rolling_5"] - out["player_fta_season_avg"]
    if "min_rolling_5" in out.columns:
        out["min_delta_5"] = out["min_rolling_5"] - out["player_min_season_avg"]

    return out

