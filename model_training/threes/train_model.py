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
    blend_fg3m_mean,
    build_attempt_overdispersion_target,
    calibration_report,
    choose_fg3m_blend_weight,
    estimate_rate_strength,
    fit_threshold_isotonic_calibrators,
    make_fg3_rate_baseline,
    make_fg3_rate_model,
    make_fg3a_baseline,
    make_fg3a_dispersion_model,
    make_fg3a_model,
    predict_fg3m_variance,
    prob_ge_k,
    safe_mape,
    variance_to_nbinom_alpha,
)
from model_training.utils.team_codes import norm_team


CALIBRATION_THRESHOLDS = [1, 2, 3, 4, 5]
RATE_PRIOR_ATT = 80.0


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


def _fill_from_medians(
    df: pd.DataFrame,
    used_features: list[str],
    medians: dict[str, float],
) -> pd.DataFrame:
    X = df[used_features].copy()
    for col in used_features:
        X[col] = X[col].fillna(medians[col])
    return X


def _rate_feature_frame(
    df: pd.DataFrame,
    pred_fg3a_context: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    aligned = pred_fg3a_context.loc[out.index].astype(float)
    out["pred_fg3a_context"] = aligned.to_numpy()
    out["pred_fg3a_context_log1p"] = np.log1p(np.clip(out["pred_fg3a_context"], 0.0, None))
    return out


def _make_rate_feature_list(base_features: list[str]) -> list[str]:
    feats = list(base_features)
    for c in ["pred_fg3a_context", "pred_fg3a_context_log1p"]:
        if c not in feats:
            feats.append(c)
    return feats


def _stratified_fg3_metrics(valid_df: pd.DataFrame, pred_mean: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}

    tmp = valid_df[["fg3m"]].copy()
    tmp["pred_mean"] = pred_mean
    tmp["attempt_tier"] = pd.cut(
        valid_df["fg3a_rolling_5"].fillna(valid_df["player_fg3a_season_avg"]).fillna(0.0),
        bins=[-np.inf, 2.5, 5.0, np.inf],
        labels=["low_volume", "mid_volume", "high_volume"],
    )

    for tier in ["low_volume", "mid_volume", "high_volume"]:
        part = tmp[tmp["attempt_tier"] == tier]
        if len(part) == 0:
            out[f"{tier}_mae"] = np.nan
            out[f"{tier}_rmse"] = np.nan
            continue

        out[f"{tier}_mae"] = float(mean_absolute_error(part["fg3m"], part["pred_mean"]))
        out[f"{tier}_rmse"] = float(np.sqrt(mean_squared_error(part["fg3m"], part["pred_mean"])))

    return out


def _stratified_fg3a_metrics(valid_df: pd.DataFrame, pred_fg3a: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}

    tmp = valid_df[["fg3a"]].copy()
    tmp["pred_fg3a"] = pred_fg3a
    tmp["attempt_tier"] = pd.cut(
        valid_df["fg3a_rolling_5"].fillna(valid_df["player_fg3a_season_avg"]).fillna(0.0),
        bins=[-np.inf, 2.5, 5.0, np.inf],
        labels=["low_volume", "mid_volume", "high_volume"],
    )

    for tier in ["low_volume", "mid_volume", "high_volume"]:
        part = tmp[tmp["attempt_tier"] == tier]
        if len(part) == 0:
            out[f"fg3a_{tier}_mae"] = np.nan
            out[f"fg3a_{tier}_rmse"] = np.nan
            continue

        out[f"fg3a_{tier}_mae"] = float(mean_absolute_error(part["fg3a"], part["pred_fg3a"]))
        out[f"fg3a_{tier}_rmse"] = float(np.sqrt(mean_squared_error(part["fg3a"], part["pred_fg3a"])))

    return out


def _tail_accuracy_metrics(y_true: pd.Series, mu_pred: np.ndarray, alpha_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    y = np.asarray(y_true, dtype=float)

    for k in [3, 4]:
        raw_p = prob_ge_k(mu_pred, alpha_pred, k)
        target = (y >= k).astype(float)
        out[f"tail_ge_{k}_brier_raw"] = float(np.mean((raw_p - target) ** 2))

    return out


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

    print(f"[INFO] Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    print(f"[INFO] Split date: {split_date}")
    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Test rows: {len(valid_df)}")

    # -------------------------
    # FG3A baseline
    # -------------------------
    fg3a_baseline_train = make_fg3a_baseline(train_df)
    fg3a_baseline_valid = make_fg3a_baseline(valid_df)

    # -------------------------
    # FG3A residual model
    # target = fg3a - baseline
    # -------------------------
    X_train_att, used_att_features, train_att_medians = _finalize_feature_matrix(train_df, FG3_FEATURES_ATT)
    X_valid_att = _fill_from_medians(valid_df, used_att_features, train_att_medians)

    y_train_att = train_df["fg3a"].astype(float).to_numpy()
    y_valid_att = valid_df["fg3a"].astype(float).to_numpy()

    y_train_att_resid = y_train_att - fg3a_baseline_train

    fg3a_model = make_fg3a_model()
    fg3a_model.fit(X_train_att, y_train_att_resid)

    pred_train_att_resid = np.asarray(fg3a_model.predict(X_train_att), dtype=float)
    pred_valid_att_resid = np.asarray(fg3a_model.predict(X_valid_att), dtype=float)

    pred_train_att = np.clip(fg3a_baseline_train + pred_train_att_resid, 0.0, None)
    pred_valid_att = np.clip(fg3a_baseline_valid + pred_valid_att_resid, 0.0, None)

    pred_train_att_s = pd.Series(pred_train_att, index=train_df.index, dtype=float)
    pred_valid_att_s = pd.Series(pred_valid_att, index=valid_df.index, dtype=float)

    # -------------------------
    # FG3A contextual dispersion
    # -------------------------
    att_alpha_target_train = build_attempt_overdispersion_target(y_train_att, pred_train_att)

    fg3a_dispersion_model = make_fg3a_dispersion_model()
    fg3a_dispersion_model.fit(X_train_att, att_alpha_target_train)

    pred_train_att_alpha = np.clip(fg3a_dispersion_model.predict(X_train_att), 0.0, None)
    pred_valid_att_alpha = np.clip(fg3a_dispersion_model.predict(X_valid_att), 0.0, None)

    # -------------------------
    # FG3 rate model (binomial, dependency-aware)
    # -------------------------
    rate_feature_cols = _make_rate_feature_list(FG3_FEATURES_RATE)

    train_rate_df = train_df[train_df["fg3a"] > 0].copy()
    valid_rate_df = valid_df[valid_df["fg3a"] > 0].copy()

    train_rate_df = _rate_feature_frame(train_rate_df, pred_train_att_s)
    valid_rate_df = _rate_feature_frame(valid_rate_df, pred_valid_att_s)

    X_train_rate, used_rate_features, train_rate_medians = _finalize_feature_matrix(train_rate_df, rate_feature_cols)
    X_valid_rate = _fill_from_medians(valid_rate_df, used_rate_features, train_rate_medians)

    fg3_rate_model = make_fg3_rate_model(train_rate_df, used_rate_features)

    y_train_rate = (train_rate_df["fg3m"] / train_rate_df["fg3a"]).clip(0, 1).to_numpy()
    y_valid_rate = (valid_rate_df["fg3m"] / valid_rate_df["fg3a"]).clip(0, 1).to_numpy()

    pred_train_rate = np.clip(fg3_rate_model.predict_p(X_train_rate), 0.0, 1.0)
    pred_valid_rate = np.clip(fg3_rate_model.predict_p(X_valid_rate), 0.0, 1.0)

    # -------------------------
    # Compose rate over all rows
    # -------------------------
    train_rate_all_df = _rate_feature_frame(train_df, pred_train_att_s)
    valid_rate_all_df = _rate_feature_frame(valid_df, pred_valid_att_s)

    X_train_rate_all = _fill_from_medians(train_rate_all_df, used_rate_features, train_rate_medians)
    X_valid_rate_all = _fill_from_medians(valid_rate_all_df, used_rate_features, train_rate_medians)

    pred_train_rate_all = np.clip(fg3_rate_model.predict_p(X_train_rate_all), 0.0, 1.0)
    pred_valid_rate_all = np.clip(fg3_rate_model.predict_p(X_valid_rate_all), 0.0, 1.0)

    # -------------------------
    # Baseline FG3M mean
    # -------------------------
    fg3m_baseline_train = np.clip(
        np.nan_to_num(make_fg3a_baseline(train_df), nan=0.0)
        * np.nan_to_num(make_fg3_rate_baseline(train_df, prior_att=RATE_PRIOR_ATT), nan=0.36),
        0.0,
        None,
    )
    fg3m_baseline_valid = np.clip(
        np.nan_to_num(make_fg3a_baseline(valid_df), nan=0.0)
        * np.nan_to_num(make_fg3_rate_baseline(valid_df, prior_att=RATE_PRIOR_ATT), nan=0.36),
        0.0,
        None,
    )

    # -------------------------
    # Raw model FG3M mean
    # -------------------------
    pred_train_fg3m_raw = np.clip(pred_train_att * pred_train_rate_all, 0.0, None)
    pred_valid_fg3m_raw = np.clip(pred_valid_att * pred_valid_rate_all, 0.0, None)

    # -------------------------
    # Choose mean blend weight on validation
    # -------------------------
    blend_weight, blend_metrics = choose_fg3m_blend_weight(
        y_true=valid_df["fg3m"],
        baseline_mean=fg3m_baseline_valid,
        model_mean=pred_valid_fg3m_raw,
        candidate_weights=[0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.85, 1.0],
    )

    pred_train_fg3m = blend_fg3m_mean(fg3m_baseline_train, pred_train_fg3m_raw, blend_weight)
    pred_valid_fg3m = blend_fg3m_mean(fg3m_baseline_valid, pred_valid_fg3m_raw, blend_weight)

    # -------------------------
    # Contextual variance -> alpha
    # use blended mean as center, but model-derived variance shape
    # -------------------------
    train_rate_strength = estimate_rate_strength(
        train_df["player_fg3a_season_sum"] if "player_fg3a_season_sum" in train_df.columns else pd.Series(0.0, index=train_df.index),
        prior_att=RATE_PRIOR_ATT,
    )
    valid_rate_strength = estimate_rate_strength(
        valid_df["player_fg3a_season_sum"] if "player_fg3a_season_sum" in valid_df.columns else pd.Series(0.0, index=valid_df.index),
        prior_att=RATE_PRIOR_ATT,
    )

    pred_train_var = predict_fg3m_variance(
        mu_att=pred_train_att,
        p_rate=pred_train_rate_all,
        att_alpha=pred_train_att_alpha,
        rate_strength=train_rate_strength,
    )
    pred_valid_var = predict_fg3m_variance(
        mu_att=pred_valid_att,
        p_rate=pred_valid_rate_all,
        att_alpha=pred_valid_att_alpha,
        rate_strength=valid_rate_strength,
    )

    pred_train_fg3m_alpha = variance_to_nbinom_alpha(pred_train_fg3m, pred_train_var)
    pred_valid_fg3m_alpha = variance_to_nbinom_alpha(pred_valid_fg3m, pred_valid_var)

    # -------------------------
    # Calibration layer
    # -------------------------
    calibrators = fit_threshold_isotonic_calibrators(
        y_true=valid_df["fg3m"],
        mu_pred=pred_valid_fg3m,
        alpha_pred=pred_valid_fg3m_alpha,
        thresholds=CALIBRATION_THRESHOLDS,
    )

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
        "fg3m_model_raw_valid_mae": float(mean_absolute_error(valid_df["fg3m"], pred_valid_fg3m_raw)),
        "fg3m_model_valid_mae": float(mean_absolute_error(valid_df["fg3m"], pred_valid_fg3m)),

        "fg3m_baseline_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["fg3m"], fg3m_baseline_valid))),
        "fg3m_model_raw_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["fg3m"], pred_valid_fg3m_raw))),
        "fg3m_model_valid_rmse": float(np.sqrt(mean_squared_error(valid_df["fg3m"], pred_valid_fg3m))),

        "fg3m_model_valid_mape": float(safe_mape(valid_df["fg3m"], pred_valid_fg3m)),
        "fg3m_alpha_valid_mean": float(np.mean(pred_valid_fg3m_alpha)),
        "fg3m_alpha_valid_median": float(np.median(pred_valid_fg3m_alpha)),

        "fg3m_blend_weight": float(blend_weight),
    }

    metrics.update(blend_metrics)

    metrics.update(
        calibration_report(
            y_true=valid_df["fg3m"],
            mu_pred=pred_valid_fg3m,
            alpha_pred=pred_valid_fg3m_alpha,
            calibrators=calibrators,
            thresholds=CALIBRATION_THRESHOLDS,
        )
    )
    metrics.update(_tail_accuracy_metrics(valid_df["fg3m"], pred_valid_fg3m, pred_valid_fg3m_alpha))
    metrics.update(_stratified_fg3_metrics(valid_df, pred_valid_fg3m))
    metrics.update(_stratified_fg3a_metrics(valid_df, pred_valid_att))

    joblib.dump(fg3a_model, model_dir / "fg3a_model.joblib")
    joblib.dump(fg3a_dispersion_model, model_dir / "fg3a_dispersion_model.joblib")
    joblib.dump(fg3_rate_model, model_dir / "fg3_rate_model.joblib")
    joblib.dump(calibrators, model_dir / "fg3_prob_calibrators.joblib")

    artifact_meta = {
        "target": "fg3m",
        "model_type": "composed_fg3a_residual_x_binomial_rate_depaware_blended",
        "split_date": split_date,
        "rate_prior_att": RATE_PRIOR_ATT,
        "calibration_thresholds": CALIBRATION_THRESHOLDS,
        "fg3m_blend_weight": float(blend_weight),
        "fg3a_target_mode": "residual_over_baseline",
        "fg3a_used_features": used_att_features,
        "fg3a_feature_medians": train_att_medians,
        "fg3a_dispersion_used_features": used_att_features,
        "fg3a_dispersion_feature_medians": train_att_medians,
        "fg3rate_used_features": used_rate_features,
        "fg3rate_feature_medians": train_rate_medians,
        "baseline_name": "fg3a baseline * bayesian fg3 rate baseline",
    }

    with open(model_dir / "fg3_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(model_dir / "fg3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    valid_out = valid_df[["game_date", "player", "team", "opp", "fg3m", "fg3a"]].copy()
    valid_out["fg3a_baseline"] = fg3a_baseline_valid
    valid_out["pred_fg3a"] = pred_valid_att
    valid_out["pred_fg3_rate"] = pred_valid_rate_all
    valid_out["pred_fg3m_baseline"] = fg3m_baseline_valid
    valid_out["pred_fg3m_raw"] = pred_valid_fg3m_raw
    valid_out["pred_fg3m"] = pred_valid_fg3m
    valid_out["pred_fg3m_alpha"] = pred_valid_fg3m_alpha

    for k in CALIBRATION_THRESHOLDS:
        valid_out[f"p_ge_{k}_raw"] = prob_ge_k(pred_valid_fg3m, pred_valid_fg3m_alpha, k)
        if k in calibrators:
            valid_out[f"p_ge_{k}_cal"] = calibrators[k].predict(valid_out[f"p_ge_{k}_raw"])
        else:
            valid_out[f"p_ge_{k}_cal"] = np.nan

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