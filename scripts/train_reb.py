from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_training.common.history_prep import prepare_history_df
from model_training.common.time_split import time_split
from model_training.config import PATH_GAMLOGS_COMBINED, REBOUNDS_MODEL_DIR
from model_training.rebounds.evaluation import (
    build_reb_validation_diagnostics,
    save_reb_validation_diagnostics,
)
from model_training.rebounds.features import REBOUND_FEATURES, build_all_rebounds_features
from model_training.rebounds.model import (
    fit_dispersion_alpha_mom,
    make_rebounds_baseline,
    make_rebounds_model,
    safe_mape,
)
from model_training.utils.team_codes import norm_team


def _select_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["reb", "mp_minutes", "game_date"]).copy()
    out = out[(out["reb"] >= 0) & (out["mp_minutes"] >= 0)].copy()

    if "min_rolling_5" in out.columns:
        out = out[out["min_rolling_5"].notna()].copy()

    signal_ok = (
        out["reb_per_min_5"].notna()
        | out["player_reb_per_min_season"].notna()
        | out["player_reb_season_avg"].notna()
    )
    out = out[signal_ok].copy()

    return out


def _finalize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        raise ValueError("No rebound feature columns found in dataframe.")

    usable = [c for c in existing if not df[c].isna().all()]
    if not usable:
        raise ValueError("All rebound feature columns are entirely NaN.")

    medians: dict[str, float] = {}
    X = df[usable].copy()

    for col in usable:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        medians[col] = float(med)
        X[col] = X[col].fillna(med)

    return X, usable, medians


def _make_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    minutes = train_df["mp_minutes"].fillna(0).to_numpy(dtype=float)
    rebounds = train_df["reb"].fillna(0).to_numpy(dtype=float)

    w = np.ones(len(train_df), dtype=float)

    # harder downweight on chaotic low-minute rows
    w[minutes < 12] *= 0.25
    w[(minutes >= 12) & (minutes < 16)] *= 0.60
    w[(minutes >= 16) & (minutes < 20)] *= 0.85

    # stable full-role rows matter more
    w[(minutes >= 28) & (minutes < 36)] *= 1.10
    w[minutes >= 36] *= 1.20

    # lighter ceiling weighting than Tier 2
    w[rebounds >= 10] *= 1.10
    w[rebounds >= 12] *= 1.05

    return np.clip(w, 0.10, 3.0)


def train_rebounds_model(
    *,
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = REBOUNDS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> dict[str, float]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = prepare_history_df(df, norm_team_fn=norm_team)
    df = build_all_rebounds_features(df)
    df = _select_training_frame(df)

    train_df, valid_df = time_split(df, split_date=split_date, date_col="game_date")

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if valid_df.empty:
        raise ValueError("Validation split is empty.")

    baseline_train = make_rebounds_baseline(train_df)
    baseline_valid = make_rebounds_baseline(valid_df)

    X_train, used_features, train_feature_medians = _finalize_feature_matrix(train_df, REBOUND_FEATURES)
    X_valid = valid_df[used_features].copy()
    for col in used_features:
        X_valid[col] = X_valid[col].fillna(train_feature_medians[col])

    y_train = train_df["reb"].astype(float).to_numpy()
    y_valid = valid_df["reb"].astype(float).to_numpy()

    sample_weight = _make_sample_weights(train_df)

    model = make_rebounds_model()
    model.fit(X_train, y_train, sample_weight=sample_weight)

    pred_train = np.clip(model.predict(X_train), 0.0, None)
    pred_valid = np.clip(model.predict(X_valid), 0.0, None)

    dispersion_alpha_train = fit_dispersion_alpha_mom(train_df["reb"], pred_train)
    dispersion_alpha_valid = fit_dispersion_alpha_mom(valid_df["reb"], pred_valid)

    metrics = {
        "n_train": float(len(train_df)),
        "n_valid": float(len(valid_df)),
        "baseline_train_mae": float(mean_absolute_error(y_train, baseline_train)),
        "baseline_valid_mae": float(mean_absolute_error(y_valid, baseline_valid)),
        "model_train_mae": float(mean_absolute_error(y_train, pred_train)),
        "model_valid_mae": float(mean_absolute_error(y_valid, pred_valid)),
        "baseline_train_rmse": float(np.sqrt(mean_squared_error(y_train, baseline_train))),
        "baseline_valid_rmse": float(np.sqrt(mean_squared_error(y_valid, baseline_valid))),
        "model_train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "model_valid_rmse": float(np.sqrt(mean_squared_error(y_valid, pred_valid))),
        "baseline_valid_mape": float(safe_mape(valid_df["reb"], baseline_valid)),
        "model_valid_mape": float(safe_mape(valid_df["reb"], pred_valid)),
        "train_dispersion_alpha_mom": float(dispersion_alpha_train),
        "valid_dispersion_alpha_mom": float(dispersion_alpha_valid),
    }

    joblib.dump(model, model_dir / "reb_model.joblib")

    artifact_meta = {
        "target": "reb",
        "model_type": "HistGradientBoostingRegressor",
        "split_date": split_date,
        "used_features": used_features,
        "feature_medians": train_feature_medians,
        "dispersion_alpha_mom": float(dispersion_alpha_train),
        "baseline_name": "min_rolling_5 * reb_per_min_5 with season fallbacks",
    }

    with open(model_dir / "reb_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(model_dir / "reb_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    valid_out = valid_df[["game_date", "player", "team", "opp", "reb"]].copy()
    valid_out["baseline_pred"] = baseline_valid
    valid_out["model_pred"] = pred_valid
    valid_out["model_minus_baseline"] = valid_out["model_pred"] - valid_out["baseline_pred"]
    valid_out.to_csv(model_dir / "reb_validation_predictions.csv", index=False)

    diag_df, diag_outputs = build_reb_validation_diagnostics(
        valid_df=valid_df,
        pred_valid=pred_valid,
        baseline_valid=baseline_valid,
    )
    save_reb_validation_diagnostics(
        model_dir=model_dir,
        diag_df=diag_df,
        outputs=diag_outputs,
    )

    return metrics


def main():
    metrics = train_rebounds_model(
        csv_path=PATH_GAMLOGS_COMBINED,
        model_dir=REBOUNDS_MODEL_DIR,
    )

    print("[REB] Training complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()