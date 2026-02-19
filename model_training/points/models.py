# model_training/points/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor


# -----------------------------------------------------------------------------
# Math helpers (same as threes/models.py)
# -----------------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def smoothed_rate(
    makes: pd.Series,
    att: pd.Series,
    *,
    alpha: float = 0.5,
    beta: float = 1.0
) -> np.ndarray:
    """
    p = (makes + alpha) / (att + alpha + beta)
    Avoids 0%/100% labels from 0/1 attempt games.
    """
    return ((makes.astype(float) + alpha) / (att.astype(float) + alpha + beta)).to_numpy()


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------
def make_poisson_hgbr(
    max_depth=4,
    learning_rate=0.05,
    max_iter=600,
    random_state=42
) -> Pipeline:
    """
    Use Poisson loss for non-negative count-like targets (FG2A, FTA).
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])


def make_logit_hgbr(
    max_depth=6,
    learning_rate=0.05,
    max_iter=900,
    random_state=42
) -> Pipeline:
    """
    Train on logit(p) with squared error.
    Output is unbounded logits; wrap with sigmoid.
    """

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])


@dataclass
class LogitRateWrapper:
    """
    Wraps a regressor trained on logit(p).

    Contract:
      - predict_p(X) returns probability in [0, 1]
      - predict_logit(X) returns logits
    """
    pipe: Pipeline
    feature_names: list[str]
    alpha: float = 0.5
    beta: float = 1.0

    def predict_logit(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.pipe.predict(X[self.feature_names])
        return np.asarray(Z, dtype=float)

    def predict_p(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.predict_logit(X))


# -----------------------------------------------------------------------------
# Fit helpers for points components
# -----------------------------------------------------------------------------
def fit_attempt_model(
    train_df: pd.DataFrame,
    *,
    feature_names: list[str],
    target_col: str,
    max_depth=4,
    learning_rate=0.05,
    max_iter=600,
    random_state=42,
) -> Pipeline:
    """
    Fits Poisson HGBR for attempt volume targets (FG2A, FTA).
    """
    pipe = make_poisson_hgbr(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )
    pipe.fit(train_df[feature_names], train_df[target_col].astype(float).clip(lower=0))
    return pipe


def fit_rate_model_logit(
    train_df: pd.DataFrame,
    *,
    feature_names: list[str],
    makes_col: str,
    att_col: str,
    alpha: float = 0.5,
    beta: float = 1.0,
    weight_clip_min: float = 0.0,
    max_depth=4,
    learning_rate=0.05,
    max_iter=600,
    random_state=42,
) -> LogitRateWrapper:
    """
    Fits logit(rate) regressor for FG2% or FT% using stabilized labels and attempt weights.

    - label: smoothed_rate(makes, att)
    - target: logit(label)
    - sample_weight: attempts (optionally clipped)
    """
    pipe = make_logit_hgbr(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )

    makes = train_df[makes_col].astype(float)
    att = train_df[att_col].astype(float).clip(lower=0)

    p = smoothed_rate(makes, att, alpha=alpha, beta=beta)
    y = logit(p)

    w = att.to_numpy()
    if weight_clip_min > 0:
        w = np.clip(w, weight_clip_min, None)

    pipe.fit(train_df[feature_names], y, model__sample_weight=w)

    return LogitRateWrapper(pipe=pipe, feature_names=feature_names, alpha=alpha, beta=beta)

