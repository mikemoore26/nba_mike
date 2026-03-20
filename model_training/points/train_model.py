from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from model_training.common.history_prep import prepare_history_df
from model_training.common.time_split import time_split
from model_training.config import PATH_GAMLOGS_COMBINED, POINTS_MODEL_DIR
from model_training.points.features import (
    PTS_FEATURES_FG2A,
    PTS_FEATURES_FG2RATE,
    PTS_FEATURES_FTA,
    PTS_FEATURES_FTRATE,
    build_all_points_features,
)
from model_training.points.models import (
    fit_dispersion_alpha_mom,
    make_fg2_rate_baseline,
    make_fg2_rate_model,
    make_fg2a_baseline,
    make_fg2a_model,
    make_ft_rate_baseline,
    make_ft_rate_model,
    make_fta_baseline,
    make_fta_model,
    safe_mape,
)
from model_training.utils.team_codes import norm_team


def _select_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    needed = ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "mp_minutes", "game_date"]
    out = out.dropna(subset=[c for c in needed if c in out.columns]).copy()
    out = out[(out["pts"] >= 0) & (out["mp_minutes"] >= 0)].copy()

    if "min_rolling_5" in out.columns:
        out = out[out["min_rolling_5"].notna()].copy()

    return out


def _finalize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        raise ValueError("No point feature columns found in dataframe.")

    usable = [c for c in existing if not df[c].isna().all()]
    if not usable:
        raise ValueError("All point feature columns are entirely NaN.")

    medians: dict[str, float] = {}
    X = df[usable].copy()

    for col in usable:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        medians[col] = float(med)
        X[col] = X[col].fillna(med)

    return X, usable, medians


def train_points_model(
    *,
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = POINTS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> dict[str, float]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = prepare_history_df(df, norm_team_fn=norm_team)
    df = build_all_points_features(df)
    df = _select_training_frame(df)

    train_df, valid_df = time_split(df, split_date=split_date, date_col="game_date")

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if valid_df.empty:
        raise ValueError("Validation split is empty.")

    # FG2A
    fg2a_baseline_valid = make_fg2a_baseline(valid_df)
    X_train_fg2a, used_fg2a_features, fg2a_medians = _finalize_feature_matrix(train_df, PTS_FEATURES_FG2A)
    X_valid_fg2a = valid_df[used_fg2a_features].copy()
    for col in used_fg2a_features:
        X_valid_fg2a[col] = X_valid_fg2a[col].fillna(fg2a_medians[col])

    y_train_fg2a = (train_df["fga"] - train_df["fg3a"]).astype(float).to_numpy()
    y_valid_fg2a = (valid_df["fga"] - valid_df["fg3a"]).astype(float).to_numpy()

    fg2a_model = make_fg2a_model()
    fg2a_model.fit(X_train_fg2a, y_train_fg2a)

    pred_valid_fg2a = np.clip(fg2a_model.predict(X_valid_fg2a), 0.0, None)

    # FG2 RATE
    train_fg2rate_df = train_df[(train_df["fga"] - train_df["fg3a"]) > 0].copy()
    valid_fg2rate_df = valid_df[(valid_df["fga"] - valid_df["fg3a"]) > 0].copy()

    X_train_fg2rate, used_fg2rate_features, fg2rate_medians = _finalize_feature_matrix(train_fg2rate_df, PTS_FEATURES_FG2RATE)
    X_valid_fg2rate = valid_fg2rate_df[used_fg2rate_features].copy()
    for col in used_fg2rate_features:
        X_valid_fg2rate[col] = X_valid_fg2rate[col].fillna(fg2rate_medians[col])

    y_train_fg2rate = ((train_fg2rate_df["fgm"] - train_fg2rate_df["fg3m"]) / (train_fg2rate_df["fga"] - train_fg2rate_df["fg3a"])).clip(0, 1).to_numpy()
    y_valid_fg2rate = ((valid_fg2rate_df["fgm"] - valid_fg2rate_df["fg3m"]) / (valid_fg2rate_df["fga"] - valid_fg2rate_df["fg3a"])).clip(0, 1).to_numpy()

    fg2rate_model = make_fg2_rate_model()
    fg2rate_model.fit(X_train_fg2rate, y_train_fg2rate)

    X_valid_fg2rate_all = valid_df[used_fg2rate_features].copy()
    for col in used_fg2rate_features:
        X_valid_fg2rate_all[col] = X_valid_fg2rate_all[col].fillna(fg2rate_medians[col])

    pred_valid_fg2rate_all = np.clip(fg2rate_model.predict(X_valid_fg2rate_all), 0.0, 1.0)

    # FTA
    fta_baseline_valid = make_fta_baseline(valid_df)
    X_train_fta, used_fta_features, fta_medians = _finalize_feature_matrix(train_df, PTS_FEATURES_FTA)
    X_valid_fta = valid_df[used_fta_features].copy()
    for col in used_fta_features:
        X_valid_fta[col] = X_valid_fta[col].fillna(fta_medians[col])

    y_train_fta = train_df["fta"].astype(float).to_numpy()
    y_valid_fta = valid_df["fta"].astype(float).to_numpy()

    fta_model = make_fta_model()
    fta_model.fit(X_train_fta, y_train_fta)

    pred_valid_fta = np.clip(fta_model.predict(X_valid_fta), 0.0, None)

    # FT RATE
    train_ftrate_df = train_df[train_df["fta"] > 0].copy()
    valid_ftrate_df = valid_df[valid_df["fta"] > 0].copy()

    X_train_ftrate, used_ftrate_features, ftrate_medians = _finalize_feature_matrix(train_ftrate_df, PTS_FEATURES_FTRATE)
    X_valid_ftrate = valid_ftrate_df[used_ftrate_features].copy()
    for col in used_ftrate_features:
        X_valid_ftrate[col] = X_valid_ftrate[col].fillna(ftrate_medians[col])

    y_train_ftrate = (train_ftrate_df["ftm"] / train_ftrate_df["fta"]).clip(0, 1).to_numpy()
    y_valid_ftrate = (valid_ftrate_df["ftm"] / valid_ftrate_df["fta"]).clip(0, 1).to_numpy()

    ftrate_model = make_ft_rate_model()
    ftrate_model.fit(X_train_ftrate, y_train_ftrate)

    X_valid_ftrate_all = valid_df[used_ftrate_features].copy()
    for col in used_ftrate_features:
        X_valid_ftrate_all[col] = X_valid_ftrate_all[col].fillna(ftrate_medians[col])

    pred_valid_ftrate_all = np.clip(ftrate_model.predict(X_valid_ftrate_all), 0.0, 1.0)

    # Composed PTS validation
    pts_baseline_valid = np.clip(
        2.0 * np.nan_to_num(make_fg2a_baseline(valid_df), nan=0.0) * np.nan_to_num(make_fg2_rate_baseline(valid_df), nan=0.52)
        + np.nan_to_num(make_fta_baseline(valid_df), nan=0.0) * np.nan_to_num(make_ft_rate_baseline(valid_df), nan=0.78)
        + 3.0 * np.nan_to_num(valid_df["fg3m"].to_numpy(dtype=float), nan=0.0),
        0.0,
        None,
    )

    pred_valid_pts = np.clip(
        2.0 * np.nan_to_num(pred_valid_fg2a, nan=0.0) * np.nan_to_num(pred_valid_fg2rate_all, nan=0.52)
        + np.nan_to_num(pred_valid_fta, nan=0.0) * np.nan_to_num(pred_valid_ftrate_all, nan=0.78)
        + 3.0 * np.nan_to_num(valid_df["fg3m"].to_numpy(dtype=float), nan=0.0),
        0.0,
        None,
    )

    dispersion_alpha_valid = fit_dispersion_alpha_mom(valid_df["pts"], pred_valid_pts)

    metrics = {
        "n_train": float(len(train_df)),
        "n_valid": float(len(valid_df)),
        "fg2a_model_valid_mae": float(mean_absolute_error(y_valid_fg2a, pred_valid_fg2a)),
        "fta_model_valid_mae": float(mean_absolute_error(y_valid_fta, pred_valid_fta)),
        "pts_baseline_valid_mae": float(mean_absolute_error(valid_df["pts"], pts_baseline_valid)),
        "pts_model_valid_mae": float(mean_absolute_error(valid_df["pts"], pred_valid_pts)),
        "pts_baseline_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["pts"], pts_baseline_valid))),
        "pts_model_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["pts"], pred_valid_pts))),
        "pts_model_valid_mape": float(safe_mape(valid_df["pts"], pred_valid_pts)),
        "valid_dispersion_alpha_mom": float(dispersion_alpha_valid),
    }

    joblib.dump(fg2a_model, model_dir / "fg2a_model.joblib")
    joblib.dump(fg2rate_model, model_dir / "fg2rate_model.joblib")
    joblib.dump(fta_model, model_dir / "fta_model.joblib")
    joblib.dump(ftrate_model, model_dir / "ftrate_model.joblib")

    artifact_meta = {
        "target": "pts",
        "model_type": "composed_fg2a_x_fg2rate_plus_fta_x_ftrate_plus_fg3",
        "split_date": split_date,
        "fg2a_used_features": used_fg2a_features,
        "fg2a_feature_medians": fg2a_medians,
        "fg2rate_used_features": used_fg2rate_features,
        "fg2rate_feature_medians": fg2rate_medians,
        "fta_used_features": used_fta_features,
        "fta_feature_medians": fta_medians,
        "ftrate_used_features": used_ftrate_features,
        "ftrate_feature_medians": ftrate_medians,
        "dispersion_alpha_mom": float(dispersion_alpha_valid),
        "baseline_name": "2*(fg2a baseline*fg2 rate baseline)+fta baseline*ft rate baseline+3*actual fg3m",
    }

    with open(model_dir / "pts_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(model_dir / "pts_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    valid_out = valid_df[["game_date", "player", "team", "opp", "pts", "fg3m"]].copy()
    valid_out["pred_fg2a"] = pred_valid_fg2a
    valid_out["pred_fg2rate"] = pred_valid_fg2rate_all
    valid_out["pred_fta"] = pred_valid_fta
    valid_out["pred_ftrate"] = pred_valid_ftrate_all
    valid_out["pred_pts"] = pred_valid_pts
    valid_out.to_csv(model_dir / "pts_validation_predictions.csv", index=False)

    return metrics


def main(
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = POINTS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> None:
    metrics = train_points_model(
        csv_path=csv_path,
        model_dir=model_dir,
        split_date=split_date,
    )

    print("[PTS] Training complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()