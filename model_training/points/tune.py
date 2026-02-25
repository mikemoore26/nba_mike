# model_training/points/tune.py
from __future__ import annotations

import json
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import ParameterGrid

from model_training.points.train_model import time_split
from model_training.points.features import (
    build_points_features_no_leak,
    add_player_baselines_points,
    add_opp_2p_defense_features_roll,
    add_opp_ft_defense_features_roll,
    FG2A_FEATURES,
    FG2_RATE_FEATURES,
    FTA_FEATURES,
    FT_RATE_FEATURES,
)

from model_training.threes.features import (
    build_features_no_leak as build_3p_features,
    add_player_baselines as add_3p_baselines,
    add_opp_3p_defense_features_roll,
    add_team_stint_features,
)

from model_training.points.models import (
    make_poisson_hgbr,
    make_logit_hgbr,
    smoothed_rate,
    logit,
    sigmoid,
    LogitRateWrapper,
)


# -----------------------------------------------------------
# Small but effective grid
# -----------------------------------------------------------
POISSON_GRID = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.04],
    "max_iter": [600, 900],
}

RATE_GRID = {
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.04],
    "max_iter": [600, 900],
}


# -----------------------------------------------------------
# Tuning function
# -----------------------------------------------------------
def tune_and_train_points_models(
    *,
    csv_path: str,
    model_dir: str,
    split_date: str = "2025-01-01",
):

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])

    # ----------- Build Features ONCE (no leakage) ------------
    df = build_points_features_no_leak(df)
    df = add_player_baselines_points(df)
    df = add_team_stint_features(df)
    df = add_opp_2p_defense_features_roll(df)
    df = add_opp_ft_defense_features_roll(df)

    df = build_3p_features(df)
    df = add_3p_baselines(df)
    df = add_opp_3p_defense_features_roll(df)

    df = df.dropna(subset=["fg2a", "fta", "fg3", "pts"]).copy()

    train_df, valid_df = time_split(df, split_date)

    results_summary = {}

    # =========================================================
    # 1️⃣ FG2A (Poisson)
    # =========================================================
    best_score = np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(POISSON_GRID):
        pipe = make_poisson_hgbr(**params)
        pipe.fit(train_df[FG2A_FEATURES], train_df["fg2a"])

        pred = np.clip(pipe.predict(valid_df[FG2A_FEATURES]), 0, None)
        score = mean_poisson_deviance(valid_df["fg2a"], pred)

        if score < best_score:
            best_score = score
            best_model = pipe
            best_params = params

    results_summary["fg2a"] = {
        "poisson_deviance": float(best_score),
        "params": best_params,
    }

    best_model_path = model_dir / "fg2a.joblib"
    import joblib
    joblib.dump(best_model, best_model_path)

    # =========================================================
    # 2️⃣ FTA (Poisson)
    # =========================================================
    best_score = np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(POISSON_GRID):
        pipe = make_poisson_hgbr(**params)
        pipe.fit(train_df[FTA_FEATURES], train_df["fta"])

        pred = np.clip(pipe.predict(valid_df[FTA_FEATURES]), 0, None)
        score = mean_poisson_deviance(valid_df["fta"], pred)

        if score < best_score:
            best_score = score
            best_model = pipe
            best_params = params

    results_summary["fta"] = {
        "poisson_deviance": float(best_score),
        "params": best_params,
    }

    joblib.dump(best_model, model_dir / "fta.joblib")

    # =========================================================
    # 3️⃣ FG2 Rate (Logit space)
    # =========================================================
    tr = train_df[train_df["fg2a"] > 0]
    va = valid_df[valid_df["fg2a"] > 0]

    p_tr = smoothed_rate(tr["fg2m"], tr["fg2a"])
    y_tr = logit(p_tr)
    w_tr = tr["fg2a"]

    best_score = np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(RATE_GRID):
        pipe = make_logit_hgbr(**params)
        pipe.fit(tr[FG2_RATE_FEATURES], y_tr, model__sample_weight=w_tr)

        pred_p = sigmoid(pipe.predict(va[FG2_RATE_FEATURES]))
        score = mean_absolute_error(
            smoothed_rate(va["fg2m"], va["fg2a"]),
            pred_p,
        )

        if score < best_score:
            best_score = score
            best_model = pipe
            best_params = params

    results_summary["fg2_rate"] = {
        "mae": float(best_score),
        "params": best_params,
    }

    joblib.dump(
        LogitRateWrapper(best_model, FG2_RATE_FEATURES),
        model_dir / "fg2_rate.joblib",
    )

    # =========================================================
    # 4️⃣ FT Rate
    # =========================================================
    tr = train_df[train_df["fta"] > 0]
    va = valid_df[valid_df["fta"] > 0]

    p_tr = smoothed_rate(tr["ft"], tr["fta"])
    y_tr = logit(p_tr)
    w_tr = tr["fta"]

    best_score = np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(RATE_GRID):
        pipe = make_logit_hgbr(**params)
        pipe.fit(tr[FT_RATE_FEATURES], y_tr, model__sample_weight=w_tr)

        pred_p = sigmoid(pipe.predict(va[FT_RATE_FEATURES]))
        score = mean_absolute_error(
            smoothed_rate(va["ft"], va["fta"]),
            pred_p,
        )

        if score < best_score:
            best_score = score
            best_model = pipe
            best_params = params

    results_summary["ft_rate"] = {
        "mae": float(best_score),
        "params": best_params,
    }

    joblib.dump(
        LogitRateWrapper(best_model, FT_RATE_FEATURES),
        model_dir / "ft_rate.joblib",
    )

    # ---------------------------------------------------------
    # Save metrics summary
    # ---------------------------------------------------------
    with open(model_dir / "points_tuning_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\nTraining Complete")
    print(json.dumps(results_summary, indent=2))
