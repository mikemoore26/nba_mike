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


def make_fg2a_baseline(df: pd.DataFrame) -> np.ndarray:
    baseline = df["fg2a_rolling_5"]
    fallback_1 = df["player_fg2a_season_avg"]
    fallback_2 = df["fga_rolling_5"] * 0.65
    fallback_3 = pd.Series(8.0, index=df.index, dtype=float)

    baseline = baseline.fillna(fallback_1)
    baseline = baseline.fillna(fallback_2)
    baseline = baseline.fillna(fallback_3)

    return np.clip(baseline.to_numpy(dtype=float), 0.0, None)


def make_fg2_rate_baseline(df: pd.DataFrame) -> np.ndarray:
    baseline = df["fg2_pct_rolling_10"].copy()
    if "player_fg2_pct_season" in df.columns:
        baseline = baseline.fillna(df["player_fg2_pct_season"])

    if {"fg2a_rolling_10"}.issubset(df.columns):
        baseline = baseline.fillna(0.52)

    baseline = baseline.replace([np.inf, -np.inf], np.nan).fillna(0.52)
    return np.clip(baseline.to_numpy(dtype=float), 0.0, 1.0)


def make_fta_baseline(df: pd.DataFrame) -> np.ndarray:
    baseline = df["fta_rolling_5"]
    fallback_1 = df["player_fta_season_avg"]
    fallback_2 = pd.Series(2.5, index=df.index, dtype=float)

    baseline = baseline.fillna(fallback_1)
    baseline = baseline.fillna(fallback_2)

    return np.clip(baseline.to_numpy(dtype=float), 0.0, None)


def make_ft_rate_baseline(df: pd.DataFrame) -> np.ndarray:
    baseline = df["ft_pct_rolling_10"].copy()
    if "player_ft_pct_season" in df.columns:
        baseline = baseline.fillna(df["player_ft_pct_season"])

    baseline = baseline.replace([np.inf, -np.inf], np.nan).fillna(0.78)
    return np.clip(baseline.to_numpy(dtype=float), 0.0, 1.0)


def make_fg2a_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=42,
    )


def make_fg2_rate_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.5,
        random_state=42,
    )


def make_fta_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=42,
    )


def make_ft_rate_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.5,
        random_state=42,
    )