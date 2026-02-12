# model_training/threes/train_model.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model_training.threes.features import (
    build_features_no_leak,
    add_player_baselines,
    add_opp_3p_defense_features_roll,
    add_team_stint_features,
    FG3A_FEATURES,
    RATE_FEATURES,
)

from model_training.threes.models import (
    make_poisson_hgbr,
    make_logit_hgbr,
    smoothed_rate,
    logit,
    sigmoid,
    LogitRateWrapper,
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
# Model trainers
# ----------------------------
def train_fg3a(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    pipe = make_poisson_hgbr()

    X_tr, y_tr = train_df[FG3A_FEATURES], train_df["fg3a"].astype(float)
    X_va, y_va = valid_df[FG3A_FEATURES], valid_df["fg3a"].astype(float)

    pipe.fit(X_tr, y_tr)
    pred = np.clip(pipe.predict(X_va), 0, None)

    print("FG3A MAE:", mean_absolute_error(y_va, pred))
    return pipe


def train_rate_logit(train_df: pd.DataFrame, valid_df: pd.DataFrame, *, alpha=0.5, beta=1.0):
    tr = train_df[train_df["fg3a"] > 0].copy()
    va = valid_df[valid_df["fg3a"] > 0].copy()

    X_tr = tr[RATE_FEATURES]
    X_va = va[RATE_FEATURES]

    p_tr = smoothed_rate(tr["fg3"], tr["fg3a"], alpha=alpha, beta=beta)
    p_va = smoothed_rate(va["fg3"], va["fg3a"], alpha=alpha, beta=beta)

    y_tr = logit(p_tr)  # train in logit space
    w_tr = tr["fg3a"].astype(float).to_numpy()

    pipe = make_logit_hgbr()
    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)

    # Validate in probability space
    pred_p = sigmoid(pipe.predict(X_va))
    w_va = va["fg3a"].astype(float).to_numpy()
    wmae = np.average(np.abs(p_va - pred_p), weights=w_va)
    print("RATE wMAE:", wmae)

    # Wrap so inference is ALWAYS probability-safe
    return LogitRateWrapper(pipe=pipe, feature_names=RATE_FEATURES, alpha=alpha, beta=beta)


# ----------------------------
# Public training API
# ----------------------------
def train_models(
    *,
    csv_path: str,
    fg3a_model_path: str,
    fg3_rate_model_path: str,
    features_path: str,
    split_date: str = "2025-01-01",
):
    """
    Trains:
      - FG3A (Poisson-ish regression)
      - FG3 rate (logit(smoothed fg3/fg3a), attempt-weighted)

    NOTE: This is a pure function: no scraping, no schedule, no retrain policy.
    Those belong in scripts/.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Pregame-safe feature engineering
    df = build_features_no_leak(df)
    df = add_player_baselines(df)
    df = add_team_stint_features(df)
    df = add_opp_3p_defense_features_roll(df)

    _check_features("FG3A", FG3A_FEATURES, df)
    _check_features("RATE", RATE_FEATURES, df)

    df = df.dropna(subset=["fg3a", "fg3"]).copy()
    df = df[df["fg3a"] >= 0].copy()

    train_df, valid_df = time_split(df, split_date=split_date)

    fg3a_pipe = train_fg3a(train_df, valid_df)
    rate_model = train_rate_logit(train_df, valid_df, alpha=0.5, beta=1.0)

    features_union = sorted(set(FG3A_FEATURES) | set(RATE_FEATURES))
    features_artifact = {
        "FG3A_FEATURES": FG3A_FEATURES,
        "RATE_FEATURES": RATE_FEATURES,
        "FEATURES_UNION": features_union,
        "split_date": split_date,
        "rate_model": "LogitRateWrapper(HGBR)",
        "rate_label": "logit(smoothed_fg3/fg3a)",
        "rate_smoothing": {"alpha": rate_model.alpha, "beta": rate_model.beta},
    }

    joblib.dump(fg3a_pipe, fg3a_model_path)
    joblib.dump(rate_model, fg3_rate_model_path)
    joblib.dump(features_artifact, features_path)

    print("\nSaved:")
    print(" -", fg3a_model_path)
    print(" -", fg3_rate_model_path)
    print(" -", features_path)

    return fg3a_pipe, rate_model
