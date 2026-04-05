from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score


TAIL_THRESHOLDS = [8, 10, 12]


def make_rebounds_tail_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.5,
        random_state=42,
    )


def make_tail_target(y: pd.Series | np.ndarray, threshold: int) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return (arr >= threshold).astype(int)


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def tail_sample_weights(
    train_df: pd.DataFrame,
    threshold: int,
) -> np.ndarray:
    """
    Tail classifier weighting:
    - low-minute chaos downweighted
    - positive class moderately upweighted
    """
    minutes = train_df["mp_minutes"].fillna(0).to_numpy(dtype=float)
    rebounds = train_df["reb"].fillna(0).to_numpy(dtype=float)

    w = np.ones(len(train_df), dtype=float)

    w[minutes < 12] *= 0.25
    w[(minutes >= 12) & (minutes < 16)] *= 0.60
    w[(minutes >= 16) & (minutes < 20)] *= 0.85

    w[(minutes >= 28) & (minutes < 36)] *= 1.10
    w[minutes >= 36] *= 1.20

    # positive-class emphasis by threshold
    if threshold == 8:
        w[rebounds >= 8] *= 1.35
    elif threshold == 10:
        w[rebounds >= 10] *= 1.60
    elif threshold == 12:
        w[rebounds >= 12] *= 1.90

    return np.clip(w, 0.10, 5.0)


def evaluate_tail_model(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "positive_rate": float(np.mean(y_true)),
        "mean_pred_prob": float(np.mean(y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
    }