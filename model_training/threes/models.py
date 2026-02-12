# model_training/threes/models.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def smoothed_rate(makes: pd.Series, att: pd.Series, alpha: float = 0.5, beta: float = 1.0) -> np.ndarray:
    """
    p = (makes + alpha) / (att + alpha + beta)
    Avoids 0%/100% labels from 0/1 attempt games.
    """
    return ((makes.astype(float) + alpha) / (att.astype(float) + alpha + beta)).to_numpy()


def make_poisson_hgbr(max_depth=4, learning_rate=0.05, max_iter=600, random_state=42) -> Pipeline:
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


def make_logit_hgbr(max_depth=4, learning_rate=0.05, max_iter=600, random_state=42) -> Pipeline:
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
