from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_training.assists.features import AST_FEATURES, build_all_assists_features
from model_training.assists.model import (
    fit_dispersion_alpha_mom,
    make_assists_baseline,
    make_assists_model,
    safe_mape,
)
from model_training.common.history_prep import prepare_history_df
from model_training.common.time_split import time_split
from model_training.config import ASSISTS_MODEL_DIR, PATH_GAMLOGS_COMBINED
from model_training.utils.team_codes import norm_team


def _select_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["ast", "mp_minutes", "game_date"]).copy()
    out = out[(out["ast"] >= 0) & (out["mp_minutes"] >= 0)].copy()

    if "min_rolling_5" in out.columns:
        out = out[out["min_rolling_5"].notna()].copy()

    signal_ok = (
        out["ast_per_min_5"].notna()
        | out["player_ast_per_min_season"].notna()
        | out["player_ast_season_avg"].notna()
    )
    out = out[signal_ok].copy()

    return out


def _finalize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        raise ValueError("No AST feature columns found in dataframe.")

    usable = [c for c in existing if not df[c].isna().all()]
    if not usable:
        raise ValueError("All AST feature columns are entirely NaN.")

    medians: dict[str, float] = {}
    X = df[usable].copy()

    for col in usable:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        medians[col] = float(med)
        X[col] = X[col].fillna(med)

    return X, usable, medians


def train_assists_model(
    *,
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = ASSISTS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> dict[str, float]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = prepare_history_df(df, norm_team_fn=norm_team)
    df = build_all_assists_features(df)
    df = _select_training_frame(df)

    train_df, valid_df = time_split(df, split_date=split_date, date_col="game_date")

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if valid_df.empty:
        raise ValueError("Validation split is empty.")

    baseline_train = make_assists_baseline(train_df)
    baseline_valid = make_assists_baseline(valid_df)

    X_train, used_features, train_feature_medians = _finalize_feature_matrix(train_df, AST_FEATURES)

    X_valid = valid_df[used_features].copy()
    for col in used_features:
        X_valid[col] = X_valid[col].fillna(train_feature_medians[col])

    y_train = train_df["ast"].astype(float).to_numpy()
    y_valid = valid_df["ast"].astype(float).to_numpy()

    model = make_assists_model()
    model.fit(X_train, y_train)

    pred_train = np.clip(model.predict(X_train), 0.0, None)
    pred_valid = np.clip(model.predict(X_valid), 0.0, None)

    dispersion_alpha_train = fit_dispersion_alpha_mom(train_df["ast"], pred_train)
    dispersion_alpha_valid = fit_dispersion_alpha_mom(valid_df["ast"], pred_valid)

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
        "baseline_valid_mape": float(safe_mape(valid_df["ast"], baseline_valid)),
        "model_valid_mape": float(safe_mape(valid_df["ast"], pred_valid)),
        "train_dispersion_alpha_mom": float(dispersion_alpha_train),
        "valid_dispersion_alpha_mom": float(dispersion_alpha_valid),
    }

    joblib.dump(model, model_dir / "ast_model.joblib")

    artifact_meta = {
        "target": "ast",
        "model_type": "HistGradientBoostingRegressor",
        "split_date": split_date,
        "used_features": used_features,
        "feature_medians": train_feature_medians,
        "dispersion_alpha_mom": float(dispersion_alpha_train),
        "baseline_name": "min_rolling_5 * ast_per_min_5 with season fallbacks",
    }

    with open(model_dir / "ast_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(model_dir / "ast_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    valid_out = valid_df[["game_date", "player", "team", "opp", "ast"]].copy()
    valid_out["baseline_pred"] = baseline_valid
    valid_out["model_pred"] = pred_valid
    valid_out["model_minus_baseline"] = valid_out["model_pred"] - valid_out["baseline_pred"]
    valid_out.to_csv(model_dir / "ast_validation_predictions.csv", index=False)

    return metrics


def main(
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = ASSISTS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> None:
    metrics = train_assists_model(
        csv_path=csv_path,
        model_dir=model_dir,
        split_date=split_date,
    )

    print("[AST] Training complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()