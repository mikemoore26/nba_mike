from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_training.common.history_prep import prepare_history_df
from model_training.common.time_split import time_split
from model_training.config import PATH_GAMLOGS_COMBINED, THREES_MODEL_DIR
from model_training.threes.features import (
    FG3_FEATURES_ATT,
    FG3_FEATURES_RATE,
    build_all_threes_features,
)
from model_training.threes.models import (
    fit_dispersion_alpha_mom,
    make_fg3_rate_baseline,
    make_fg3_rate_model,
    make_fg3a_baseline,
    make_fg3a_model,
    safe_mape,
)
from model_training.utils.team_codes import norm_team


def _select_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["fg3m", "fg3a", "mp_minutes", "game_date"]).copy()
    out = out[(out["fg3m"] >= 0) & (out["fg3a"] >= 0) & (out["mp_minutes"] >= 0)].copy()

    if "min_rolling_5" in out.columns:
        out = out[out["min_rolling_5"].notna()].copy()

    if "fg3a_rolling_5" in out.columns:
        out = out[out["fg3a_rolling_5"].notna() | out["player_fg3a_season_avg"].notna()].copy()

    return out


def _finalize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        raise ValueError("No three-point feature columns found in dataframe.")

    usable = [c for c in existing if not df[c].isna().all()]
    if not usable:
        raise ValueError("All three-point feature columns are entirely NaN.")

    medians: dict[str, float] = {}
    X = df[usable].copy()

    for col in usable:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        medians[col] = float(med)
        X[col] = X[col].fillna(med)

    return X, usable, medians


def train_threes_model(
    *,
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = THREES_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> dict[str, float]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = prepare_history_df(df, norm_team_fn=norm_team)
    df = build_all_threes_features(df)
    df = _select_training_frame(df)

    train_df, valid_df = time_split(df, split_date=split_date, date_col="game_date")

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if valid_df.empty:
        raise ValueError("Validation split is empty.")

    # -------------------------
    # FG3A model
    # -------------------------
    fg3a_baseline_train = make_fg3a_baseline(train_df)
    fg3a_baseline_valid = make_fg3a_baseline(valid_df)

    X_train_att, used_att_features, train_att_medians = _finalize_feature_matrix(train_df, FG3_FEATURES_ATT)
    X_valid_att = valid_df[used_att_features].copy()
    for col in used_att_features:
        X_valid_att[col] = X_valid_att[col].fillna(train_att_medians[col])

    y_train_att = train_df["fg3a"].astype(float).to_numpy()
    y_valid_att = valid_df["fg3a"].astype(float).to_numpy()

    fg3a_model = make_fg3a_model()
    fg3a_model.fit(X_train_att, y_train_att)

    pred_train_att = np.clip(fg3a_model.predict(X_train_att), 0.0, None)
    pred_valid_att = np.clip(fg3a_model.predict(X_valid_att), 0.0, None)

    # -------------------------
    # FG3 rate model
    # only train on rows with attempts > 0 historically
    # -------------------------
    train_rate_df = train_df[train_df["fg3a"] > 0].copy()
    valid_rate_df = valid_df[valid_df["fg3a"] > 0].copy()

    X_train_rate, used_rate_features, train_rate_medians = _finalize_feature_matrix(train_rate_df, FG3_FEATURES_RATE)
    X_valid_rate = valid_rate_df[used_rate_features].copy()
    for col in used_rate_features:
        X_valid_rate[col] = X_valid_rate[col].fillna(train_rate_medians[col])

    y_train_rate = (train_rate_df["fg3m"] / train_rate_df["fg3a"]).clip(0, 1).to_numpy()
    y_valid_rate = (valid_rate_df["fg3m"] / valid_rate_df["fg3a"]).clip(0, 1).to_numpy()

    fg3_rate_model = make_fg3_rate_model()
    fg3_rate_model.fit(X_train_rate, y_train_rate)

    pred_train_rate = np.clip(fg3_rate_model.predict(X_train_rate), 0.0, 1.0)
    pred_valid_rate = np.clip(fg3_rate_model.predict(X_valid_rate), 0.0, 1.0)

    # -------------------------
    # composed FG3M mean validation
    # -------------------------
    X_valid_rate_on_all = valid_df[used_rate_features].copy()
    for col in used_rate_features:
        X_valid_rate_on_all[col] = X_valid_rate_on_all[col].fillna(train_rate_medians[col])

    pred_valid_rate_all = np.clip(fg3_rate_model.predict(X_valid_rate_on_all), 0.0, 1.0)

    fg3m_baseline_valid = np.clip(
        np.nan_to_num(make_fg3a_baseline(valid_df), nan=0.0)
        * np.nan_to_num(make_fg3_rate_baseline(valid_df), nan=0.36),
        0.0,
        None,
    )

    pred_valid_fg3m = np.clip(
        np.nan_to_num(pred_valid_att, nan=0.0)
        * np.nan_to_num(pred_valid_rate_all, nan=0.36),
        0.0,
        None,
    )
    
    dispersion_alpha_valid = fit_dispersion_alpha_mom(valid_df["fg3m"], pred_valid_fg3m)

    metrics = {
        "n_train": float(len(train_df)),
        "n_valid": float(len(valid_df)),
        "fg3a_baseline_valid_mae": float(mean_absolute_error(y_valid_att, fg3a_baseline_valid)),
        "fg3a_model_valid_mae": float(mean_absolute_error(y_valid_att, pred_valid_att)),
        "fg3a_baseline_valid_rmse": float(np.sqrt(mean_squared_error(y_valid_att, fg3a_baseline_valid))),
        "fg3a_model_valid_rmse": float(np.sqrt(mean_squared_error(y_valid_att, pred_valid_att))),
        "fg3_rate_model_valid_mae": float(mean_absolute_error(y_valid_rate, pred_valid_rate)) if len(valid_rate_df) else np.nan,
        "fg3_rate_model_valid_rmse": float(np.sqrt(mean_squared_error(y_valid_rate, pred_valid_rate))) if len(valid_rate_df) else np.nan,
        "fg3m_baseline_valid_mae": float(mean_absolute_error(valid_df["fg3m"], fg3m_baseline_valid)),
        "fg3m_model_valid_mae": float(mean_absolute_error(valid_df["fg3m"], pred_valid_fg3m)),
        "fg3m_baseline_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["fg3m"], fg3m_baseline_valid))),
        "fg3m_model_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["fg3m"], pred_valid_fg3m))),
        "fg3m_model_valid_mape": float(safe_mape(valid_df["fg3m"], pred_valid_fg3m)),
        "valid_dispersion_alpha_mom": float(dispersion_alpha_valid),
    }

    joblib.dump(fg3a_model, model_dir / "fg3a_model.joblib")
    joblib.dump(fg3_rate_model, model_dir / "fg3_rate_model.joblib")

    artifact_meta = {
        "target": "fg3m",
        "model_type": "composed_fg3a_x_fg3rate",
        "split_date": split_date,
        "fg3a_used_features": used_att_features,
        "fg3a_feature_medians": train_att_medians,
        "fg3rate_used_features": used_rate_features,
        "fg3rate_feature_medians": train_rate_medians,
        "dispersion_alpha_mom": float(dispersion_alpha_valid),
        "baseline_name": "fg3a baseline * fg3 rate baseline",
    }

    with open(model_dir / "fg3_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(model_dir / "fg3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    valid_out = valid_df[["game_date", "player", "team", "opp", "fg3m", "fg3a"]].copy()
    valid_out["pred_fg3a"] = pred_valid_att
    valid_out["pred_fg3_rate"] = pred_valid_rate_all
    valid_out["pred_fg3m"] = pred_valid_fg3m
    valid_out.to_csv(model_dir / "fg3_validation_predictions.csv", index=False)

    return metrics


def main(
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = THREES_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> None:
    metrics = train_threes_model(
        csv_path=csv_path,
        model_dir=model_dir,
        split_date=split_date,
    )

    print("[FG3] Training complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()