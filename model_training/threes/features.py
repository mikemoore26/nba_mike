# model_training/threes/features.py
from __future__ import annotations

import pandas as pd

# ----------------------------
# Feature lists (single source of truth)
# ----------------------------
# NOTE:
# - We keep `starter_prob_10` for backward compatibility, but the meaning is really
#   "minutes share proxy" (expected minutes / 36). Prefer `min_share_10` going forward.

FG3A_FEATURES = [
    # recent form (volume + role)
    "min_rolling_5",
    "fga_rolling_5",
    "fg3a_rolling_5",
    "expected_min_10",
    "min_share_10",
    # baselines (identity)
    "player_fg3a_season_avg",
    "player_min_season_avg",
    "player_usage_season",
    # change signals (role shift)
    "fg3a_delta_5",
    "min_delta_5",
    # context (pregame-known)
    "home_game",
    "days_rest",
    "back_to_back",
    # backward compat
    "starter_prob_10",
]



RATE_FEATURES = [
    # recent form (accuracy) + confidence
    "fg3_pct_rolling_10",
    "fg3_att_rolling_10",
    # baseline (stabilized)
    "player_fg3_pct_season",
    # role proxy
    "expected_min_10",
    "min_share_10",
    # context (pregame-known)
    "home_game",
    "days_rest",
    "back_to_back",
    # backward compat
    "starter_prob_10",
]

OPP_3P_DEF_FEATURES = [
    "games_played_to_date",
    "opp_fg3a_allowed_pg_to_date",
    "opp_fg3m_allowed_pg_to_date",
    "opp_3p_pct_allowed_to_date",
    "opp_def_3p_rank_to_date",
]

TEAM_STINT_FEATURES = [
    "team_games_in_stint_to_date",
    "new_team_game",
    "recent_team_change_5",
]

# Then extend your model feature sets
FG3A_FEATURES = FG3A_FEATURES + OPP_3P_DEF_FEATURES + TEAM_STINT_FEATURES
RATE_FEATURES = RATE_FEATURES + OPP_3P_DEF_FEATURES + TEAM_STINT_FEATURES





# model_training/threes/features.py
import numpy as np
import pandas as pd

def add_team_stint_features(
    df: pd.DataFrame,
    *,
    player_col: str = "player",
    team_col: str = "team",
    date_col: str = "date",
    recent_k: int = 5,
) -> pd.DataFrame:
    """
    Leakage-safe "consecutive games with current team" features.

    Definitions (per player row at date t):
      - team_games_in_stint_to_date:
          number of prior games in the current team stint (0 for first game on team)
      - new_team_game:
          1 if first game of a new team stint, else 0
      - recent_team_change_5:
          1 if player changed teams within the last `recent_k` games (pregame-known), else 0

    Requires columns: player, team, date
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values([player_col, date_col])

    g = out.groupby(player_col, sort=False)

    # Detect team change (this game vs previous game)
    prev_team = g[team_col].shift(1)
    changed_team = (out[team_col] != prev_team) & prev_team.notna()

    # Stint id increments whenever team changes (first game -> stint 0)
    # Use cumsum over the boolean change flag within player
    out["_team_stint_id"] = g.apply(lambda x: (x[team_col].ne(x[team_col].shift(1)) & x[team_col].shift(1).notna()).cumsum())\
                             .reset_index(level=0, drop=True)

    # Game index within stint (0,1,2...) including this game
    out["_game_in_stint"] = out.groupby([player_col, "_team_stint_id"], sort=False).cumcount()

    # Leakage-safe: number of prior games in this stint (shifted by 1)
    out["team_games_in_stint_to_date"] = out["_game_in_stint"].astype(float)

    # First game with new team stint (pregame-known from prior logs)
    out["new_team_game"] = (out["team_games_in_stint_to_date"] == 0).astype(int)

    # "Changed teams within last K games" — pregame-known:
    # look back at the last K *prior* games and see if any had a team change event.
    # (changed_team marks changes at that prior game relative to its previous game)
    out["_changed_team_flag"] = changed_team.astype(int)
    out["recent_team_change_5"] = (
        g["_changed_team_flag"].shift(1).rolling(recent_k, min_periods=1).max()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .astype(int)
    )

    out.drop(columns=["_team_stint_id", "_game_in_stint", "_changed_team_flag"], inplace=True)
    return out


def add_opp_3p_defense_features_roll(
    df: pd.DataFrame,
    min_games_for_rank: int = 5,
    fill_early_rank: float | None = None,
) -> pd.DataFrame:
    """
    Leakage-safe opponent 3P defense features from player game logs.

    Defensive team is df['opp'].
    For each (season, opp, date), we compute opponent defense *to date* (excluding the current date).
    Then merge back to each player row on (season, opp, date).
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Aggregate all players vs that opponent on that date
    opp_game = (
        data.groupby(["season", "opp", "date"], as_index=False)
            .agg(
                fg3a_allowed=("fg3a", "sum"),
                fg3m_allowed=("fg3", "sum"),
            )
            .sort_values(["season", "opp", "date"])
    )

    g = opp_game.groupby(["season", "opp"], sort=False)

    # number of prior games (so current date is excluded)
    opp_game["games_played_to_date"] = g.cumcount()
    opp_game["fg3a_allowed_to_date"] = g["fg3a_allowed"].cumsum() - opp_game["fg3a_allowed"]
    opp_game["fg3m_allowed_to_date"] = g["fg3m_allowed"].cumsum() - opp_game["fg3m_allowed"]

    denom_games = opp_game["games_played_to_date"].replace(0, np.nan)

    opp_game["opp_fg3a_allowed_pg_to_date"] = opp_game["fg3a_allowed_to_date"] / denom_games
    opp_game["opp_fg3m_allowed_pg_to_date"] = opp_game["fg3m_allowed_to_date"] / denom_games

    opp_game["opp_3p_pct_allowed_to_date"] = (
        opp_game["fg3m_allowed_to_date"] /
        opp_game["fg3a_allowed_to_date"].replace(0, np.nan)
    )

    # date-by-date rank inside season (lower allowed = better defense)
    eligible = opp_game["games_played_to_date"] >= min_games_for_rank
    opp_game["opp_def_3p_rank_to_date"] = np.nan
    opp_game.loc[eligible, "opp_def_3p_rank_to_date"] = (
        opp_game.loc[eligible]
            .groupby(["season", "date"])["opp_fg3m_allowed_pg_to_date"]
            .rank(method="min", ascending=True)
    )

    if fill_early_rank is not None:
        opp_game["opp_def_3p_rank_to_date"] = opp_game["opp_def_3p_rank_to_date"].fillna(fill_early_rank)

    feat_cols = [
        "season", "opp", "date",
        "games_played_to_date",
        "opp_fg3a_allowed_pg_to_date",
        "opp_fg3m_allowed_pg_to_date",
        "opp_3p_pct_allowed_to_date",
        "opp_def_3p_rank_to_date",
    ]

    out = data.merge(opp_game[feat_cols], on=["season", "opp", "date"], how="left")
    return out



# ----------------------------
# No-leak feature builder
# ----------------------------
def build_features_no_leak(
    df: pd.DataFrame,
    *,
    roll5: int = 5,
    roll10: int = 10,
    rest_fill: int = 4,
    rest_cap: int = 7,
    pct_prior: float = 0.36,
    pct_prior_att: float = 50.0,
) -> pd.DataFrame:
    """
    Creates trailing/rolling features using shift(1) so each row uses ONLY past games.

    Requires columns:
      player, date, season, team, opp, mp_minutes, fga, fg3a, fg3, usage, is_home

    Notes:
      - Uses min_periods=1 for rolling means so early-season rows aren't all NaN.
      - Uses a Beta-style prior for rolling FG3% when attempts are small or zero.
    """
    df = df.sort_values(["player", "date"]).copy()
    g = df.groupby("player", sort=False)

    # --- trailing rolling means (past-only) ---
    df["min_rolling_5"] = (
        g["mp_minutes"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["fga_rolling_5"] = (
        g["fga"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["fg3a_rolling_5"] = (
        g["fg3a"].shift(1).rolling(roll5, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # --- recent 10-game pct using sums (past-only), plus attempts count for confidence ---
    made10 = g["fg3"].shift(1).rolling(roll10, min_periods=1).sum().reset_index(level=0, drop=True)
    att10 = g["fg3a"].shift(1).rolling(roll10, min_periods=1).sum().reset_index(level=0, drop=True)

    df["fg3_att_rolling_10"] = att10

    # Stabilized rolling pct with prior (handles att10 == 0 and low-sample volatility)
    prior_made = pct_prior * pct_prior_att
    df["fg3_pct_rolling_10"] = (made10 + prior_made) / (att10 + pct_prior_att)

    # --- rest/context (pregame-known) ---
    days_rest = g["date"].diff().dt.days.reset_index(level=0, drop=True)
    df["days_rest"] = days_rest.clip(lower=0, upper=rest_cap).fillna(rest_fill)
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)

    # --- home flag (pregame-known) ---
    df["home_game"] = df["is_home"].astype(int)

    # --- pregame-safe role proxy: expected minutes based on last 10 games (past-only) ---
    df["expected_min_10"] = (
        g["mp_minutes"].shift(1).rolling(roll10, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["min_share_10"] = (df["expected_min_10"] / 36.0).clip(0, 1)

    # backward compatibility (old name)
    df["starter_prob_10"] = df["min_share_10"]

    return df


# ----------------------------
# Player baselines (season-to-date, past-only)
# ----------------------------
import numpy as np
import pandas as pd

def add_player_baselines(
    df: pd.DataFrame,
    *,
    pct_prior: float = 0.36,
    pct_prior_att: float = 200.0,
) -> pd.DataFrame:
    """
    Adds season-to-date baselines using shift(1) expanding stats within (player, season).

    Also exposes baseline strength diagnostics:
      - player_fg3a_season_sum (past-only)
      - player_fg3_made_season_sum (past-only)
      - pct_prior_weight in [0,1] = how much the prior still dominates the pct estimate
    """
    df = df.sort_values(["player", "season", "date"]).copy()
    g = df.groupby(["player", "season"], sort=False)

    # expanding mean of past fg3a (season-to-date, past-only)
    df["player_fg3a_season_avg"] = g["fg3a"].transform(lambda s: s.shift(1).expanding().mean())

    # expanding sums for pct (season-to-date, past-only)
    made_sum = g["fg3"].transform(lambda s: s.shift(1).expanding().sum())
    att_sum  = g["fg3a"].transform(lambda s: s.shift(1).expanding().sum())

    # keep sums (needed for credibility gating)
    df["player_fg3_made_season_sum"] = made_sum
    df["player_fg3a_season_sum"] = att_sum

    # Stabilize season pct with a prior (Beta-style shrinkage)
    prior_made = pct_prior * pct_prior_att
    df["player_fg3_pct_season"] = (made_sum + prior_made) / (att_sum + pct_prior_att)

    # how prior-dominated is the pct estimate?
    # if att_sum is small, prior_weight ~ 1 (bad for baseline identity)
    df["pct_prior_weight"] = pct_prior_att / (att_sum + pct_prior_att)

    # role baselines
    df["player_min_season_avg"] = g["mp_minutes"].transform(lambda s: s.shift(1).expanding().mean())
    df["player_usage_season"] = g["usage"].transform(lambda s: s.shift(1).expanding().mean())

    # change signals
    if "fg3a_rolling_5" in df.columns:
        df["fg3a_delta_5"] = df["fg3a_rolling_5"] - df["player_fg3a_season_avg"]

    if "min_rolling_5" in df.columns:
        df["min_delta_5"] = df["min_rolling_5"] - df["player_min_season_avg"]

    return df
