import pandas as pd

def build_features_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates trailing/rolling features using shift(1) so each row uses ONLY past games.
    Requires columns:
    player, date, season, team, opp, mp_minutes, fga, fg3a, fg3, usage, is_home
    """
    df = df.sort_values(["player", "date"]).copy()
    g = df.groupby("player")

    # trailing rolling means (past-only)
    df["min_rolling_5"]  = g["mp_minutes"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fga_rolling_5"]  = g["fga"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg3a_rolling_5"] = g["fg3a"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)
    df["fg3_rolling_5"]  = g["fg3"].shift(1).rolling(5).mean().reset_index(level=0, drop=True)

    made10 = g["fg3"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    att10  = g["fg3a"].shift(1).rolling(10).sum().reset_index(level=0, drop=True)
    df["fg3_pct_rolling_10"] = made10 / att10

    # rest/context
    df["days_rest"] = g["date"].diff().dt.days
    df["back_to_back"] = (df["days_rest"] == 1).astype(int)

    # home flag
    df["home_game"] = df["is_home"].astype(int)

    # starter flag: if you don't have it, set constant 1
    if "starter_flag" not in df.columns:
        df["starter_flag"] = 1

    return df


def add_player_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Player identity WITHOUT using player name as a categorical feature.
    Past-only expanding stats via shift(1).
    """
    df = df.sort_values(["player", "date"]).copy()
    g = df.groupby("player")

    # "who is this player" baseline volume/skill/role
    df["player_fg3a_season_avg"] = g["fg3a"].shift(1).expanding().mean().reset_index(level=0, drop=True)

    made = g["fg3"].shift(1).expanding().sum()
    att  = g["fg3a"].shift(1).expanding().sum()
    df["player_fg3_pct_season"] = (made / att).reset_index(level=0, drop=True)

    df["player_min_season_avg"] = g["mp_minutes"].shift(1).expanding().mean().reset_index(level=0, drop=True)
    df["player_usage_season"]   = g["usage"].shift(1).expanding().mean().reset_index(level=0, drop=True)

    return df

