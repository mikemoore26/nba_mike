from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.where(np.asarray(y_true) == 0, np.nan, np.asarray(y_true))
    out = np.abs((np.asarray(y_true) - y_pred) / denom)
    return float(np.nanmean(out))


def fit_dispersion_alpha_mom(y_true: pd.Series, mu_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(mu_pred, dtype=float), 1e-6, None)

    numer = np.mean((y - mu) ** 2 - mu)
    denom = np.mean(mu ** 2)

    if denom <= 0:
        return 0.0

    alpha = numer / denom
    return float(max(alpha, 0.0))


def make_fg3a_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Baseline attempt model:
      1) fg3a_rolling_5
      2) player_fg3a_season_avg
      3) fga_rolling_5 * 0.35
      4) league-ish fallback
    """
    baseline = df["fg3a_rolling_5"]

    fallback_1 = df["player_fg3a_season_avg"]
    fallback_2 = df["fga_rolling_5"] * 0.35
    fallback_3 = pd.Series(3.5, index=df.index, dtype=float)

    baseline = baseline.fillna(fallback_1)
    baseline = baseline.fillna(fallback_2)
    baseline = baseline.fillna(fallback_3)

    return np.clip(baseline.to_numpy(dtype=float), 0.0, None)


def make_fg3_rate_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Baseline make-rate model:
      1) fg3_pct_rolling_10
      2) player_fg3_pct_season
      3) rolling made / rolling attempts
      4) league fallback
    """
    baseline = df["fg3_pct_rolling_10"].copy()

    if "player_fg3_pct_season" in df.columns:
        baseline = baseline.fillna(df["player_fg3_pct_season"])

    if {"fg3m_rolling_5", "fg3a_rolling_5"}.issubset(df.columns):
        rolling_rate_5 = (
            df["fg3m_rolling_5"] / df["fg3a_rolling_5"].replace(0, np.nan)
        )
        baseline = baseline.fillna(rolling_rate_5)

    # league fallback from available history frame
    if {"fg3m", "fg3a"}.issubset(df.columns):
        league_rate = float(df["fg3m"].sum() / max(df["fg3a"].sum(), 1))
        if not np.isfinite(league_rate) or league_rate <= 0:
            league_rate = 0.36
    else:
        league_rate = 0.36

    baseline = baseline.fillna(league_rate)
    baseline = baseline.replace([np.inf, -np.inf], np.nan).fillna(0.36)

    return np.clip(baseline.to_numpy(dtype=float), 0.0, 1.0)


def make_fg3a_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=42,
    )


def make_fg3_rate_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.5,
        random_state=42,
    )