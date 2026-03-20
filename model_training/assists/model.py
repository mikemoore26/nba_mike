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


def make_assists_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Basketball-consistent baseline:
        recent minutes * recent ast_per_min
    """
    baseline = df["min_rolling_5"] * df["ast_per_min_5"]

    fallback_1 = df["player_min_season_avg"] * df["player_ast_per_min_season"]
    fallback_2 = df["player_ast_season_avg"]
    fallback_3 = pd.Series(2.5, index=df.index, dtype=float)

    baseline = baseline.fillna(fallback_1)
    baseline = baseline.fillna(fallback_2)
    baseline = baseline.fillna(fallback_3)

    return np.clip(baseline.to_numpy(dtype=float), 0.0, None)


def make_assists_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=42,
    )