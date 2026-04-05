from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.where(np.asarray(y_true) == 0, np.nan, np.asarray(y_true))
    out = np.abs((np.asarray(y_true) - y_pred) / denom)
    return float(np.nanmean(out))


def fit_dispersion_alpha_mom(y_true: pd.Series, mu_pred: np.ndarray) -> float:
    """
    NB2 method-of-moments style estimate:
        Var = mu + alpha * mu^2
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(mu_pred, dtype=float), 1e-6, None)

    numer = np.mean((y - mu) ** 2 - mu)
    denom = np.mean(mu ** 2)

    if denom <= 0:
        return 0.0

    alpha = numer / denom
    return float(max(alpha, 0.0))


def make_rebounds_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Basketball-consistent baseline:
        recent minutes * recent reb_per_min
    """
    baseline = df["min_rolling_5"] * df["reb_per_min_5"]
    fallback_1 = df["player_min_season_avg"] * df["player_reb_per_min_season"]
    fallback_2 = df["player_reb_season_avg"]

    baseline = baseline.fillna(fallback_1)
    baseline = baseline.fillna(fallback_2)

    return np.clip(baseline.to_numpy(dtype=float), 0.0, None)


def make_rebounds_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=42,
    )


def apply_low_minutes_dampener(
    pred_mean: np.ndarray,
    minutes_proj: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Safe post-model shrink for chaotic low-minute players.
    Based on your diagnostics, 0-12 minute players are materially overpredicted.
    This does not change schema or training artifacts.
    """
    mu = np.asarray(pred_mean, dtype=float).copy()
    mins = np.asarray(minutes_proj, dtype=float)

    low_mask = mins < 12
    midlow_mask = (mins >= 12) & (mins < 16)

    mu[low_mask] *= 0.75
    mu[midlow_mask] *= 0.90

    return np.clip(mu, 0.0, None)


def conditional_dispersion_alpha(
    *,
    base_alpha: float,
    minutes_proj: pd.Series | np.ndarray,
    pred_mean: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """
    Safe conditional alpha adjustment:
    - low-minute roles are noisier -> larger alpha
    - high-minute roles are more stable -> smaller alpha
    - optional small mean-based tweak for strong rebounders

    Returns row-level alpha values while preserving downstream schema:
    still one 'dispersion' value per row.
    """
    mins = np.asarray(minutes_proj, dtype=float)
    alpha = np.full(shape=len(mins), fill_value=float(base_alpha), dtype=float)

    alpha[mins < 12] *= 1.60
    alpha[(mins >= 12) & (mins < 20)] *= 1.25
    alpha[(mins >= 20) & (mins < 30)] *= 1.00
    alpha[(mins >= 30) & (mins < 36)] *= 0.85
    alpha[mins >= 36] *= 0.75

    if pred_mean is not None:
        mu = np.asarray(pred_mean, dtype=float)
        alpha[mu >= 10] *= 0.90
        alpha[mu >= 12] *= 0.90

    return np.clip(alpha, 1e-8, None)