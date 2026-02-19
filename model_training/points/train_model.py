# model_training/points/train_model.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from model_training.points.features import (
    add_derived_2pt_cols,
    build_points_features_no_leak,
    add_player_baselines_points,
    add_opp_2p_defense_features_roll,
    add_opp_ft_defense_features_roll,
    FG2A_FEATURES,
    FG2_RATE_FEATURES,
    FTA_FEATURES,
    FT_RATE_FEATURES,
)

# Reuse the exact stint logic you already trust
from model_training.threes.features import add_team_stint_features

from model_training.points.models import (
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
def train_fg2a(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    pipe = make_poisson_hgbr()

    X_tr, y_tr = train_df[FG2A_FEATURES], train_df["fg2a"].astype(float)
    X_va, y_va = valid_df[FG2A_FEATURES], valid_df["fg2a"].astype(float)

    pipe.fit(X_tr, y_tr)
    pred = np.clip(pipe.predict(X_va), 0, None)

    print("FG2A MAE:", mean_absolute_error(y_va, pred))
    return pipe


def train_fta(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    pipe = make_poisson_hgbr()

    X_tr, y_tr = train_df[FTA_FEATURES], train_df["fta"].astype(float)
    X_va, y_va = valid_df[FTA_FEATURES], valid_df["fta"].astype(float)

    pipe.fit(X_tr, y_tr)
    pred = np.clip(pipe.predict(X_va), 0, None)

    print("FTA MAE:", mean_absolute_error(y_va, pred))
    return pipe


def train_fg2_rate_logit(train_df: pd.DataFrame, valid_df: pd.DataFrame, *, alpha=0.5, beta=1.0):
    # Keep only games where fg2a > 0 for rate learning, but note:
    # weighting by fg2a already downweights low-attempt games
    tr = train_df[train_df["fg2a"] > 0].copy()
    va = valid_df[valid_df["fg2a"] > 0].copy()

    X_tr = tr[FG2_RATE_FEATURES]
    X_va = va[FG2_RATE_FEATURES]

    p_tr = smoothed_rate(tr["fg2m"], tr["fg2a"], alpha=alpha, beta=beta)
    p_va = smoothed_rate(va["fg2m"], va["fg2a"], alpha=alpha, beta=beta)

    y_tr = logit(p_tr)
    w_tr = tr["fg2a"].astype(float).to_numpy()

    pipe = make_logit_hgbr()
    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)

    pred_p = sigmoid(pipe.predict(X_va))
    w_va = va["fg2a"].astype(float).to_numpy()
    wmae = np.average(np.abs(p_va - pred_p), weights=w_va)
    print("FG2 RATE wMAE:", wmae)

    return LogitRateWrapper(pipe=pipe, feature_names=FG2_RATE_FEATURES, alpha=alpha, beta=beta)


def train_ft_rate_logit(train_df: pd.DataFrame, valid_df: pd.DataFrame, *, alpha=0.5, beta=1.0):
    tr = train_df[train_df["fta"] > 0].copy()
    va = valid_df[valid_df["fta"] > 0].copy()

    X_tr = tr[FT_RATE_FEATURES]
    X_va = va[FT_RATE_FEATURES]

    p_tr = smoothed_rate(tr["ft"], tr["fta"], alpha=alpha, beta=beta)
    p_va = smoothed_rate(va["ft"], va["fta"], alpha=alpha, beta=beta)

    y_tr = logit(p_tr)
    w_tr = tr["fta"].astype(float).to_numpy()

    pipe = make_logit_hgbr()
    pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)

    pred_p = sigmoid(pipe.predict(X_va))
    w_va = va["fta"].astype(float).to_numpy()
    wmae = np.average(np.abs(p_va - pred_p), weights=w_va)
    print("FT RATE wMAE:", wmae)

    return LogitRateWrapper(pipe=pipe, feature_names=FT_RATE_FEATURES, alpha=alpha, beta=beta)


# ----------------------------
# Public training API
# ----------------------------
def train_models(
    *,
    csv_path: str,
    fg2a_model_path: str,
    fg2_rate_model_path: str,
    fta_model_path: str,
    ft_rate_model_path: str,
    features_path: str,
    split_date: str = "2025-01-01",
):
    """
    Trains:
      - FG2A (Poisson-ish regression)
      - FG2 rate (logit(smoothed fg2m/fg2a), attempt-weighted)
      - FTA  (Poisson-ish regression)
      - FT rate (logit(smoothed ft/fta), attempt-weighted)

    NOTE:
      This is a pure function: no scraping, no schedule, no retrain policy.
      Those belong in scripts/.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Derive fg2a/fg2m before feature work
    df = add_derived_2pt_cols(df)

    # Pregame-safe feature engineering
    df = build_points_features_no_leak(df)
    df = add_player_baselines_points(df)
    df = add_team_stint_features(df)
    df = add_opp_2p_defense_features_roll(df)
    df = add_opp_ft_defense_features_roll(df)

    _check_features("FG2A", FG2A_FEATURES, df)
    _check_features("FG2_RATE", FG2_RATE_FEATURES, df)
    _check_features("FTA", FTA_FEATURES, df)
    _check_features("FT_RATE", FT_RATE_FEATURES, df)

    # Basic target sanity
    df = df.dropna(subset=["fg2a", "fg2m", "fta", "ft"]).copy()
    df = df[(df["fg2a"] >= 0) & (df["fg2m"] >= 0) & (df["fta"] >= 0) & (df["ft"] >= 0)].copy()
    # print("\n=== TRAIN TARGET CHECK ===")
    # print(df[["fg2a","fta","pts"]].describe())


    train_df, valid_df = time_split(df, split_date=split_date)

    fg2a_pipe = train_fg2a(train_df, valid_df)
    fg2_rate_model = train_fg2_rate_logit(train_df, valid_df, alpha=0.5, beta=1.0)

    fta_pipe = train_fta(train_df, valid_df)
    ft_rate_model = train_ft_rate_logit(train_df, valid_df, alpha=0.5, beta=1.0)

    features_union = sorted(
        set(FG2A_FEATURES)
        | set(FG2_RATE_FEATURES)
        | set(FTA_FEATURES)
        | set(FT_RATE_FEATURES)
    )

    features_artifact = {
        "FG2A_FEATURES": FG2A_FEATURES,
        "FG2_RATE_FEATURES": FG2_RATE_FEATURES,
        "FTA_FEATURES": FTA_FEATURES,
        "FT_RATE_FEATURES": FT_RATE_FEATURES,
        "FEATURES_UNION": features_union,
        "split_date": split_date,
        "rate_model": "LogitRateWrapper(HGBR)",
        "rate_label": "logit(smoothed_rate)",
        "fg2_rate_smoothing": {"alpha": fg2_rate_model.alpha, "beta": fg2_rate_model.beta},
        "ft_rate_smoothing": {"alpha": ft_rate_model.alpha, "beta": ft_rate_model.beta},
    }

    joblib.dump(fg2a_pipe, fg2a_model_path)
    joblib.dump(fg2_rate_model, fg2_rate_model_path)
    joblib.dump(fta_pipe, fta_model_path)
    joblib.dump(ft_rate_model, ft_rate_model_path)
    joblib.dump(features_artifact, features_path)

    # print("\nSaved:")
    # print(" -", fg2a_model_path)
    # print(" -", fg2_rate_model_path)
    # print(" -", fta_model_path)
    # print(" -", ft_rate_model_path)
    # print(" -", features_path)

    return fg2a_pipe, fg2_rate_model, fta_pipe, ft_rate_model
