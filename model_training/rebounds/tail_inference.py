from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


TAIL_THRESHOLDS = [8, 10, 12]

# 🔥 calibration from your validation results
TAIL_CALIBRATION = {
    8: 1.11,
    10: 1.20,
    12: 1.27,
}


# -------------------------
# LOAD MODELS
# -------------------------
def load_tail_models(model_dir: Path) -> dict[int, object]:
    models = {}
    for t in TAIL_THRESHOLDS:
        path = model_dir / f"reb_ge_{t}_model.joblib"
        if path.exists():
            models[t] = joblib.load(path)

    print(f"[TAIL MODELS LOADED]: {list(models.keys())}")
    return models


# -------------------------
# FEATURE PREP
# -------------------------
def prepare_tail_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    X = df[feature_cols].copy()

    for col in feature_cols:
        X[col] = X[col].fillna(medians.get(col, 0.0))

    return X


# -------------------------
# RAW TAIL PROBS
# -------------------------
def get_tail_probs(
    df: pd.DataFrame,
    tail_models: dict[int, object],
    feature_cols: list[str],
    medians: dict[str, float],
) -> dict[int, np.ndarray]:

    X = prepare_tail_features(df, feature_cols, medians)

    probs = {}
    for t, model in tail_models.items():
        probs[t] = model.predict_proba(X)[:, 1]

    return probs


# -------------------------
# CALIBRATION
# -------------------------
def apply_tail_calibration(
    probs: np.ndarray,
    threshold: int,
) -> np.ndarray:
    scale = TAIL_CALIBRATION.get(threshold, 1.15)
    return np.clip(probs * scale, 0.0, 1.0)


# -------------------------
# 🔥 DYNAMIC BLENDING
# -------------------------
def get_tail_weight(minutes: np.ndarray, threshold: int) -> np.ndarray:
    """
    Dynamic blending:
    More minutes → trust tail model more
    """

    minutes = np.asarray(minutes)
    w = np.zeros_like(minutes, dtype=float)

    # base weight by minutes
    w += np.where(minutes < 18, 0.60, 0.0)
    w += np.where((minutes >= 18) & (minutes < 26), 0.70, 0.0)
    w += np.where((minutes >= 26) & (minutes < 32), 0.80, 0.0)
    w += np.where((minutes >= 32) & (minutes < 38), 0.88, 0.0)
    w += np.where(minutes >= 38, 0.95, 0.0)

    # threshold boost
    if threshold == 10:
        w += 0.03
    elif threshold == 12:
        w += 0.07

    return np.clip(w, 0.5, 0.97)


# -------------------------
# MAIN BLENDING ENGINE
# -------------------------
def compute_blended_tail_probs(
    df: pd.DataFrame,
    *,
    p_nb_dict: dict[int, np.ndarray],
    tail_models: dict[int, object],
    feature_cols: list[str],
    medians: dict[str, float],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:

    tail_probs = get_tail_probs(df, tail_models, feature_cols, medians)

    blended = {}
    tail_raw = {}

    # minutes proxy
    if "min_rolling_5" in df.columns:
        minutes = df["min_rolling_5"].values
    else:
        minutes = np.zeros(len(df))

    for t in p_nb_dict.keys():
        p_nb = p_nb_dict[t]
        p_tail = tail_probs.get(t)

        if p_tail is None:
            blended[t] = p_nb
            tail_raw[t] = np.full_like(p_nb, np.nan)
            continue

        # calibration
        p_tail = apply_tail_calibration(p_tail, t)

        # 🔥 dynamic weights
        w = get_tail_weight(minutes, t)

        # blend
        blended[t] = (w * p_tail) + ((1.0 - w) * p_nb)

        tail_raw[t] = p_tail

    return blended, tail_raw