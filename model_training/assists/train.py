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


def _training_frame_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    out["diag_rows_input_to_training_frame"] = float(len(df))

    mask_base = df["ast"].notna() & df["mp_minutes"].notna() & df["game_date"].notna()
    mask_base &= (df["ast"] >= 0) & (df["mp_minutes"] >= 0)
    out["diag_rows_after_base_required"] = float(mask_base.sum())

    mask_min_floor = mask_base & (df["mp_minutes"] >= 8)
    out["diag_rows_after_min_minutes_floor"] = float(mask_min_floor.sum())

    mask_roll5 = (
        mask_min_floor & df["min_rolling_5"].notna()
        if "min_rolling_5" in df.columns
        else mask_min_floor
    )
    out["diag_rows_after_min_rolling_5_requirement"] = float(mask_roll5.sum())

    min_signal_ok = pd.Series(False, index=df.index)
    if "min_rolling_5" in df.columns:
        min_signal_ok = min_signal_ok | df["min_rolling_5"].notna()
    if "player_min_season_avg" in df.columns:
        min_signal_ok = min_signal_ok | df["player_min_season_avg"].notna()

    out["diag_rows_after_min_signal_requirement"] = float((mask_min_floor & min_signal_ok).sum())

    ast_signal_ok = (
        df["ast_per_min_5"].notna()
        | df["player_ast_per_min_season"].notna()
        | df["player_ast_season_avg"].notna()
    )
    out["diag_rows_after_ast_signal_requirement"] = float((mask_min_floor & min_signal_ok & ast_signal_ok).sum())

    return out


def _select_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["ast", "mp_minutes", "game_date"]).copy()
    out = out[(out["ast"] >= 0) & (out["mp_minutes"] >= 0)].copy()

    out = out[out["mp_minutes"] >= 8].copy()

    min_signal_ok = pd.Series(False, index=out.index)
    if "min_rolling_5" in out.columns:
        min_signal_ok = min_signal_ok | out["min_rolling_5"].notna()
    if "player_min_season_avg" in out.columns:
        min_signal_ok = min_signal_ok | out["player_min_season_avg"].notna()

    out = out[min_signal_ok].copy()

    ast_signal_ok = (
        out["ast_per_min_5"].notna()
        | out["player_ast_per_min_season"].notna()
        | out["player_ast_season_avg"].notna()
    )
    out = out[ast_signal_ok].copy()

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


def _make_recency_weights(
    df: pd.DataFrame,
    *,
    date_col: str = "game_date",
    decay_days: float = 120.0,
    min_weight: float = 0.25,
    max_weight: float = 1.0,
) -> np.ndarray:
    if date_col not in df.columns:
        return np.ones(len(df), dtype=float)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    max_date = dates.max()

    if pd.isna(max_date):
        return np.ones(len(df), dtype=float)

    days_ago = (max_date - dates).dt.days.astype(float)
    weights = np.exp(-days_ago / decay_days)
    weights = np.clip(weights, min_weight, max_weight)
    return weights.to_numpy(dtype=float)


def _choose_dispersion_alpha(
    train_alpha: float,
    valid_alpha: float,
) -> float:
    """
    Safe artifact-time alpha choice for downstream probability work.

    Rules:
    - prefer positive train alpha when available
    - if train collapses to 0 but valid shows overdispersion, use valid alpha as conservative fallback
    - otherwise leave at 0
    """
    train_alpha = float(train_alpha)
    valid_alpha = float(valid_alpha)

    if train_alpha > 1e-12:
        return train_alpha
    if valid_alpha > 1e-12:
        return valid_alpha
    return 0.0


def _add_validation_diagnostics(
    metrics: dict[str, float],
    valid_df: pd.DataFrame,
    pred_valid: np.ndarray,
) -> dict[str, float]:
    out = dict(metrics)

    diag = valid_df.copy()
    diag["pred"] = pred_valid
    diag["error"] = diag["pred"] - diag["ast"]
    diag["abs_error"] = np.abs(diag["error"])

    if "mp_minutes" in diag.columns:
        diag["min_bucket"] = pd.cut(
            diag["mp_minutes"],
            bins=[0, 15, 25, 35, 60],
            labels=["0_15", "15_25", "25_35", "35_plus"],
            include_lowest=True,
            right=False,
        )

        mae_by_min = diag.groupby("min_bucket", observed=False)["abs_error"].mean().to_dict()
        bias_by_min = diag.groupby("min_bucket", observed=False)["error"].mean().to_dict()

        out["mae_by_min_bucket"] = {
            str(k): (float(v) if pd.notna(v) else None) for k, v in mae_by_min.items()
        }
        out["bias_by_min_bucket"] = {
            str(k): (float(v) if pd.notna(v) else None) for k, v in bias_by_min.items()
        }

    if "starter_prob_10" in diag.columns:
        diag["starter_prob_bucket"] = pd.cut(
            diag["starter_prob_10"],
            bins=[-0.001, 0.2, 0.5, 0.8, 1.001],
            labels=["very_low", "low", "mid", "high"],
            include_lowest=True,
            right=False,
        )

        mae_by_role = diag.groupby("starter_prob_bucket", observed=False)["abs_error"].mean().to_dict()
        bias_by_role = diag.groupby("starter_prob_bucket", observed=False)["error"].mean().to_dict()

        out["mae_by_starter_prob_bucket"] = {
            str(k): (float(v) if pd.notna(v) else None) for k, v in mae_by_role.items()
        }
        out["bias_by_starter_prob_bucket"] = {
            str(k): (float(v) if pd.notna(v) else None) for k, v in bias_by_role.items()
        }

    pred_bins = [-0.001, 1.0, 3.0, 5.0, 7.0, 10.0, np.inf]
    pred_labels = ["lt_1", "1_3", "3_5", "5_7", "7_10", "10_plus"]
    diag["pred_bucket"] = pd.cut(diag["pred"], bins=pred_bins, labels=pred_labels)

    mae_by_pred = diag.groupby("pred_bucket", observed=False)["abs_error"].mean().to_dict()
    bias_by_pred = diag.groupby("pred_bucket", observed=False)["error"].mean().to_dict()

    out["mae_by_pred_bucket"] = {
        str(k): (float(v) if pd.notna(v) else None) for k, v in mae_by_pred.items()
    }
    out["bias_by_pred_bucket"] = {
        str(k): (float(v) if pd.notna(v) else None) for k, v in bias_by_pred.items()
    }

    return out


def train_assists_model(
    *,
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = ASSISTS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> dict[str, float]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    raw_rows = len(df)

    df = prepare_history_df(df, norm_team_fn=norm_team)
    prep_rows = len(df)

    df = build_all_assists_features(df)
    feat_rows = len(df)

    frame_diag = _training_frame_diagnostics(df)

    df = _select_training_frame(df)
    selected_rows = len(df)

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

    # raw targets for metrics / diagnostics / dispersion
    y_train_raw = train_df["ast"].astype(float).to_numpy()
    y_valid_raw = valid_df["ast"].astype(float).to_numpy()

    # transformed training target to reduce suppression of elite AST outcomes
    y_train = np.log1p(y_train_raw)

    model = make_assists_model()

    # recency weighting
    sample_weight = _make_recency_weights(train_df, date_col="game_date")

    # boost higher-assist games so rare creator ceiling rows matter more
    high_ast_boost = np.where(train_df["ast"].to_numpy(dtype=float) >= 8.0, 1.75, 1.0)
    sample_weight = sample_weight * high_ast_boost

    model.fit(X_train, y_train, sample_weight=sample_weight)

    # inverse-transform predictions back to raw AST space
    pred_train_log = model.predict(X_train)
    pred_valid_log = model.predict(X_valid)

    pred_train = np.clip(np.expm1(pred_train_log), 0.0, None)
    pred_valid = np.clip(np.expm1(pred_valid_log), 0.0, None)

    dispersion_alpha_train = fit_dispersion_alpha_mom(train_df["ast"], pred_train)
    dispersion_alpha_valid = fit_dispersion_alpha_mom(valid_df["ast"], pred_valid)
    artifact_dispersion_alpha = _choose_dispersion_alpha(
        train_alpha=dispersion_alpha_train,
        valid_alpha=dispersion_alpha_valid,
    )

    metrics = {
        "n_raw_rows": float(raw_rows),
        "n_after_prepare_history_df": float(prep_rows),
        "n_after_feature_build": float(feat_rows),
        "n_after_training_frame_selection": float(selected_rows),
        "n_train": float(len(train_df)),
        "n_valid": float(len(valid_df)),
        "train_unique_players": float(train_df["player"].nunique()),
        "valid_unique_players": float(valid_df["player"].nunique()),
        "baseline_train_mae": float(mean_absolute_error(y_train_raw, baseline_train)),
        "baseline_valid_mae": float(mean_absolute_error(y_valid_raw, baseline_valid)),
        "model_train_mae": float(mean_absolute_error(y_train_raw, pred_train)),
        "model_valid_mae": float(mean_absolute_error(y_valid_raw, pred_valid)),
        "baseline_train_rmse": float(np.sqrt(mean_squared_error(y_train_raw, baseline_train))),
        "baseline_valid_rmse": float(np.sqrt(mean_squared_error(y_valid_raw, baseline_valid))),
        "model_train_rmse": float(np.sqrt(mean_squared_error(y_train_raw, pred_train))),
        "model_valid_rmse": float(np.sqrt(mean_squared_error(y_valid_raw, pred_valid))),
        "baseline_valid_mape": float(safe_mape(valid_df["ast"], baseline_valid)),
        "model_valid_mape": float(safe_mape(valid_df["ast"], pred_valid)),
        "train_dispersion_alpha_mom": float(dispersion_alpha_train),
        "valid_dispersion_alpha_mom": float(dispersion_alpha_valid),
        "artifact_dispersion_alpha": float(artifact_dispersion_alpha),
        "sample_weight_decay_days": 120.0,
        "sample_weight_min": 0.25,
        "sample_weight_max": 1.0,
        "training_min_minutes_floor": 8.0,
        "target_transform": "log1p",
        "high_ast_weight_threshold": 8.0,
        "high_ast_weight_multiplier": 1.75,
    }

    metrics.update(frame_diag)
    metrics = _add_validation_diagnostics(metrics, valid_df, pred_valid)

    joblib.dump(model, model_dir / "ast_model.joblib")

    artifact_meta = {
        "target": "ast",
        "model_type": "HistGradientBoostingRegressor",
        "split_date": split_date,
        "used_features": used_features,
        "feature_medians": train_feature_medians,
        "dispersion_alpha_mom": float(artifact_dispersion_alpha),
        "baseline_name": "min_rolling_5 * ast_per_min_5 with season fallbacks",
        "target_transform": "log1p",
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