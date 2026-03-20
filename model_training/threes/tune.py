# model_training/threes/tune.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

from model_training.points.train_model import time_split  # reuse your existing splitter

from model_training.threes.features import (
    build_features_no_leak,
    add_player_baselines,
    FG3A_FEATURES,
    RATE_FEATURES,
    # If you have these in threes/features.py, include them:
    # add_opp_3p_defense_features_roll,
    # add_team_stint_features,
)

# Reuse your shared model utilities (you already built these for points)
from model_training.points.models import (
    make_poisson_hgbr,
    make_logit_hgbr,
    smoothed_rate,
    logit,
    sigmoid,
    LogitRateWrapper,
)

from model_training.threes.features import build_all_threes_features
# -----------------------------------------------------------
# Small but effective grid (same style as points)
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
def tune_and_train_threes_models(
    *,
    csv_path: str,
    model_dir: str,
    split_date: str = "2025-01-01",
    save_features_artifact: bool = True,
):
    """
    Trains:
      1) FG3A (Poisson)
      2) 3P Rate (logit space) weighted by attempts

    Saves:
      - fg3a.joblib
      - fg3_rate.joblib (wrapped to output probabilities directly)
      - threes_tuning_metrics.json
      - features.joblib (dict contract like your predict expects) [optional]
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
    csv_path,
    parse_dates=["date"],
    low_memory=False,
    dtype={"player": "string", "team": "string", "opp": "string"},
)
    # ------------------ Build features ONCE (no leakage) ------------------
    df = build_all_threes_features(df)  # this is your main features function that calls the others

    # If you have these feature builders, turn them on here:
    # df = add_team_stint_features(df)
    # df = add_opp_3p_defense_features_roll(df)

    # Labels required
    df = df.dropna(subset=["fg3a", "fg3"]).copy()
    df = df[df["fg3a"] >= 0].copy()

    train_df, valid_df = time_split(df, split_date)

    results_summary: dict[str, object] = {}

    # =========================================================
    # 1️⃣ FG3A (Poisson)
    # =========================================================
    best_score = np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(POISSON_GRID):
        pipe = make_poisson_hgbr(**params)
        pipe.fit(train_df[FG3A_FEATURES], train_df["fg3a"])

        pred = np.clip(pipe.predict(valid_df[FG3A_FEATURES]), 0, None)
        score = mean_poisson_deviance(valid_df["fg3a"], pred)

        if score < best_score:
            best_score = score
            best_model = pipe
            best_params = params

    results_summary["fg3a"] = {
        "poisson_deviance": float(best_score),
        "params": best_params,
    }

    joblib.dump(best_model, model_dir / "fg3a.joblib")

    # =========================================================
    # 2️⃣ 3P Rate (Logit space) — weighted by FG3A
    # =========================================================
    tr = train_df[train_df["fg3a"] > 0].copy()
    va = valid_df[valid_df["fg3a"] > 0].copy()

    # smoothed rate labels -> logit space
    p_tr = smoothed_rate(tr["fg3"], tr["fg3a"])
    y_tr = logit(p_tr)
    w_tr = tr["fg3a"]

    best_score = np.inf
    best_model = None
    best_params = None

    for params in ParameterGrid(RATE_GRID):
        pipe = make_logit_hgbr(**params)
        pipe.fit(tr[RATE_FEATURES], y_tr, model__sample_weight=w_tr)

        pred_p = sigmoid(pipe.predict(va[RATE_FEATURES]))
        score = mean_absolute_error(
            smoothed_rate(va["fg3"], va["fg3a"]),
            pred_p,
        )

        if score < best_score:
            best_score = score
            best_model = pipe
            best_params = params

    results_summary["fg3_rate"] = {
        "mae": float(best_score),
        "params": best_params,
    }

    # Wrap so predict can just call .predict(X) -> p in [0,1]
    joblib.dump(
        LogitRateWrapper(best_model, RATE_FEATURES),
        model_dir / "fg3_rate.joblib",
    )

    # ------------------ Save metrics summary ------------------
    with open(model_dir / "threes_tuning_metrics.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # ------------------ Save features artifact contract -------------------
    if save_features_artifact:
        features_union = sorted(set(FG3A_FEATURES) | set(RATE_FEATURES))
        features_artifact = {
            "FG3A_FEATURES": FG3A_FEATURES,
            "RATE_FEATURES": RATE_FEATURES,
            "FEATURES_UNION": features_union,
            "split_date": split_date,
            "rate_label": "logit(smoothed_fg3/fg3a)",
        }
        joblib.dump(features_artifact, model_dir / "features.joblib")

    print("\nTraining Complete")
    print(json.dumps(results_summary, indent=2))
    return results_summary


if __name__ == "__main__":
    # Example:
    # tune_and_train_threes_models(
    #     csv_path="./data/all_gamelogs_combined.csv",
    #     model_dir="./models/threes/",
    #     split_date="2025-01-01",
    # )
    pass