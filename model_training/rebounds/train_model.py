# model_training/rebounds/train_model.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model_training.rebounds.features import (
    build_features_no_leak,
    REB_FEATURES,
)

from model_training.rebounds.models import (
    make_reb_hgbr,
)


# ----------------------------
# Split + feature checks
# ----------------------------
def time_split(df: pd.DataFrame, split_date: str = "2025-01-01"):
    split_date = pd.Timestamp(split_date)
    train = df[df["date"] < split_date].copy()
    valid = df[df["date"] >= split_date].copy()
    return train, valid


def _check_features(name: str, feats: list[str], df: pd.DataFrame):
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing feature columns: {missing}")

    if len(feats) != len(set(feats)):
        dupes = pd.Series(feats).value_counts()
        dupes = dupes[dupes > 1].index.tolist()
        raise ValueError(f"[{name}] Duplicate features found: {dupes}")


# ----------------------------
# Dispersion fit (NegBin)
# ----------------------------
def _fit_dispersion_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Fit alpha for:
        Var = mu + alpha * mu^2

    Method-of-moments style:
        (y - mu)^2 - mu ≈ alpha * mu^2
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-6, None)

    resid2 = (y - mu) ** 2
    num = np.sum((resid2 - mu) * (mu ** 2))
    den = np.sum((mu ** 2) ** 2)

    if den <= 0:
        return 0.0

    alpha = num / den
    return float(max(alpha, 0.0))


# ----------------------------
# Model trainer
# ----------------------------
def train_reb(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    pipe = make_reb_hgbr()

    X_tr = train_df[REB_FEATURES]
    y_tr = train_df["reb"].astype(float)

    X_va = valid_df[REB_FEATURES]
    y_va = valid_df["reb"].astype(float)

    pipe.fit(X_tr, y_tr)
    pred = np.clip(pipe.predict(X_va), 0, None)

    model_mae = mean_absolute_error(y_va, pred)
    print("REB MAE:", round(model_mae, 4))

    # Compare against naive baseline
    baseline = X_va["player_reb_season_avg"].to_numpy()
    baseline = np.nan_to_num(baseline, nan=0.0)
    baseline_mae = mean_absolute_error(y_va, baseline)
    print("REB baseline MAE:", round(baseline_mae, 4))

    # Fit NegBin dispersion on validation
    alpha = _fit_dispersion_alpha(y_va.to_numpy(), pred)
    print("REB dispersion alpha (val-fit):", round(alpha, 4))

    return pipe, alpha


# ----------------------------
# Public training API
# ----------------------------
def train_models(
    *,
    csv_path: str,
    reb_model_path: str,
    features_path: str,
    split_date: str = "2025-01-01",
):
    """
    Trains:
      - REB model (regression on rebounds)

    Pure function:
      - no scraping
      - no retrain policy
      - no schedule
    """

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.dropna(subset=["date", "player"]).copy()

    # Pregame-safe feature engineering
    df = build_features_no_leak(df)

    _check_features("REB", REB_FEATURES, df)

    # basic hygiene
    df = df.dropna(subset=["reb", "min"]).copy()
    df = df[df["reb"] >= 0]
    df = df[df["min"] >= 0]

    train_df, valid_df = time_split(df, split_date=split_date)

    if train_df.empty or valid_df.empty:
        raise ValueError("Train or validation split is empty. Check split_date.")

    reb_pipe, alpha = train_reb(train_df, valid_df)

    features_artifact = {
        "REB_FEATURES": REB_FEATURES,
        "split_date": split_date,
        "model": "HGBR(median_impute)",
        "target": "reb",
        "distribution": "NegBin (var = mu + alpha*mu^2)",
        "dispersion_alpha": alpha,
    }

    joblib.dump(reb_pipe, reb_model_path)
    joblib.dump(features_artifact, features_path)

    print("\nSaved:")
    print(" -", reb_model_path)
    print(" -", features_path)

    return reb_pipe, features_artifact
