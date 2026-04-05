from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_training.common.feature_table import build_feature_table
from model_training.common.projection_schema import enforce_projection_schema
from model_training.config import THREES_MODEL_DIR
from model_training.threes.features import build_all_threes_features
from model_training.threes.models import (
    apply_threshold_calibrators,
    blend_fg3m_mean,
    estimate_rate_strength,
    make_fg3_rate_baseline,
    make_fg3a_baseline,
    predict_fg3m_variance,
    variance_to_nbinom_alpha,
)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_fg3_artifacts(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)

    artifacts_path = model_dir / "fg3_artifacts.json"
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Missing artifacts file: {artifacts_path}")

    artifacts = _load_json(artifacts_path)

    fg3a_model_path = model_dir / "fg3a_model.joblib"
    fg3_rate_model_path = model_dir / "fg3_rate_model.joblib"
    fg3a_dispersion_model_path = model_dir / "fg3a_dispersion_model.joblib"
    calibrators_path = model_dir / "fg3_prob_calibrators.joblib"

    if not fg3a_model_path.exists():
        raise FileNotFoundError(f"Missing fg3a model: {fg3a_model_path}")
    if not fg3_rate_model_path.exists():
        raise FileNotFoundError(f"Missing fg3 rate model: {fg3_rate_model_path}")

    bundle = {
        "fg3a_model": joblib.load(fg3a_model_path),
        "fg3_rate_model": joblib.load(fg3_rate_model_path),
        "fg3a_dispersion_model": None,
        "calibrators": {},
        "artifacts": artifacts,
        "is_v2": False,
    }

    if fg3a_dispersion_model_path.exists():
        bundle["fg3a_dispersion_model"] = joblib.load(fg3a_dispersion_model_path)
        bundle["is_v2"] = True

    if calibrators_path.exists():
        bundle["calibrators"] = joblib.load(calibrators_path)

    model_type = str(artifacts.get("model_type", ""))
    if "depaware" in model_type or "binomial" in model_type or "blended" in model_type:
        bundle["is_v2"] = True

    return bundle


def _prepare_feature_matrix(
    today_feat_df: pd.DataFrame,
    used_features: list[str],
    feature_medians: dict[str, float],
) -> pd.DataFrame:
    X = today_feat_df.copy()

    missing = [c for c in used_features if c not in X.columns]
    if missing:
        raise ValueError(f"today feature table missing required fg3 features: {missing}")

    X = X[used_features].copy()

    for col in used_features:
        X[col] = X[col].fillna(feature_medians.get(col, 0.0))

    return X


def _predict_rate_model(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_p"):
        pred = model.predict_p(X)
    elif hasattr(model, "predict"):
        pred = model.predict(X)
    else:
        raise TypeError("FG3 rate model does not expose predict_p() or predict().")

    return np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)


def _eligibility_flags(player_pred_df: pd.DataFrame) -> pd.DataFrame:
    out = player_pred_df.copy()

    reasons: list[str] = []
    eligible: list[int] = []

    for row in out.itertuples(index=False):
        row_reasons = []

        minutes_proj = row.minutes_proj if pd.notna(row.minutes_proj) else np.nan
        pred_mean = row.pred_mean if pd.notna(row.pred_mean) else np.nan
        pred_fg3a = row.pred_fg3a if pd.notna(row.pred_fg3a) else np.nan

        if pd.isna(minutes_proj) or minutes_proj < 10:
            row_reasons.append("minutes_lt_10")

        if pd.isna(pred_mean) or pred_mean < 0.5:
            row_reasons.append("pred_mean_lt_0p5")

        if pd.isna(pred_fg3a) or pred_fg3a < 1.5:
            row_reasons.append("fg3a_lt_1p5")

        if row_reasons:
            eligible.append(0)
            reasons.append("|".join(row_reasons))
        else:
            eligible.append(1)
            reasons.append("ok")

    out["is_eligible"] = eligible
    out["eligibility_reason"] = reasons
    return out


def _predict_v1(
    *,
    today_feat_df: pd.DataFrame,
    fg3a_model: object,
    fg3_rate_model: object,
    artifacts: dict,
) -> pd.DataFrame:
    fg3a_used_features = artifacts["fg3a_used_features"]
    fg3a_feature_medians = artifacts["fg3a_feature_medians"]

    fg3rate_used_features = artifacts["fg3rate_used_features"]
    fg3rate_feature_medians = artifacts["fg3rate_feature_medians"]

    dispersion = float(artifacts.get("dispersion_alpha_mom", 0.0))

    X_att = _prepare_feature_matrix(today_feat_df, fg3a_used_features, fg3a_feature_medians)
    X_rate = _prepare_feature_matrix(today_feat_df, fg3rate_used_features, fg3rate_feature_medians)

    pred_fg3a = np.clip(np.asarray(fg3a_model.predict(X_att), dtype=float), 0.0, None)
    pred_fg3_rate = _predict_rate_model(fg3_rate_model, X_rate)
    pred_mean = np.clip(pred_fg3a * pred_fg3_rate, 0.0, None)

    baseline_mean = np.clip(
        make_fg3a_baseline(today_feat_df) * make_fg3_rate_baseline(today_feat_df),
        0.0,
        None,
    )
    minutes_proj = today_feat_df["min_rolling_5"] if "min_rolling_5" in today_feat_df.columns else np.nan

    out = today_feat_df[["game_date", "player", "team", "opp"]].copy()
    out["stat"] = "fg3m"
    out["pred_fg3a"] = pred_fg3a
    out["pred_fg3_rate"] = pred_fg3_rate
    out["pred_mean"] = pred_mean
    out["baseline_mean"] = baseline_mean
    out["delta_mean"] = out["pred_mean"] - out["baseline_mean"]
    out["dist_name"] = "nbinom" if dispersion > 1e-12 else "poisson"
    out["dispersion"] = dispersion
    out["minutes_proj"] = minutes_proj
    out["model_name"] = "fg3_composed"
    out["model_version"] = "v1"

    out = _eligibility_flags(out)
    out = enforce_projection_schema(out)
    return out


def _predict_v2(
    *,
    today_feat_df: pd.DataFrame,
    fg3a_model: object,
    fg3a_dispersion_model: object | None,
    fg3_rate_model: object,
    calibrators: dict,
    artifacts: dict,
) -> pd.DataFrame:
    fg3a_used_features = artifacts["fg3a_used_features"]
    fg3a_feature_medians = artifacts["fg3a_feature_medians"]

    fg3rate_used_features = artifacts["fg3rate_used_features"]
    fg3rate_feature_medians = artifacts["fg3rate_feature_medians"]

    dispersion_feature_names = artifacts.get("fg3a_dispersion_used_features", fg3a_used_features)
    dispersion_feature_medians = artifacts.get("fg3a_dispersion_feature_medians", fg3a_feature_medians)

    thresholds = artifacts.get("calibration_thresholds", [1, 2, 3, 4, 5])
    rate_prior_att = float(artifacts.get("rate_prior_att", 80.0))
    blend_weight = float(artifacts.get("fg3m_blend_weight", 1.0))
    fg3a_target_mode = str(artifacts.get("fg3a_target_mode", "direct"))

    fg3a_baseline = make_fg3a_baseline(today_feat_df)

    X_att = _prepare_feature_matrix(today_feat_df, fg3a_used_features, fg3a_feature_medians)
    fg3a_model_pred = np.asarray(fg3a_model.predict(X_att), dtype=float)

    if fg3a_target_mode == "residual_over_baseline":
        pred_fg3a = np.clip(fg3a_baseline + fg3a_model_pred, 0.0, None)
    else:
        pred_fg3a = np.clip(fg3a_model_pred, 0.0, None)

    today_rate_df = today_feat_df.copy()
    today_rate_df["pred_fg3a_context"] = pred_fg3a
    today_rate_df["pred_fg3a_context_log1p"] = np.log1p(pred_fg3a)

    X_rate = _prepare_feature_matrix(today_rate_df, fg3rate_used_features, fg3rate_feature_medians)
    pred_fg3_rate = _predict_rate_model(fg3_rate_model, X_rate)

    if fg3a_dispersion_model is not None:
        X_disp = _prepare_feature_matrix(today_feat_df, dispersion_feature_names, dispersion_feature_medians)
        pred_att_alpha = np.clip(np.asarray(fg3a_dispersion_model.predict(X_disp), dtype=float), 0.0, None)
    else:
        pred_att_alpha = np.zeros(len(today_feat_df), dtype=float)

    pred_mean_raw = np.clip(pred_fg3a * pred_fg3_rate, 0.0, None)

    baseline_mean = np.clip(
        fg3a_baseline * make_fg3_rate_baseline(today_feat_df, prior_att=rate_prior_att),
        0.0,
        None,
    )

    pred_mean = blend_fg3m_mean(
        baseline_mean=baseline_mean,
        model_mean=pred_mean_raw,
        model_weight=blend_weight,
    )

    if "player_fg3a_season_sum" in today_feat_df.columns:
        rate_strength_source = today_feat_df["player_fg3a_season_sum"]
    else:
        rate_strength_source = pd.Series(0.0, index=today_feat_df.index)

    rate_strength = estimate_rate_strength(rate_strength_source, prior_att=rate_prior_att)

    pred_var = predict_fg3m_variance(
        mu_att=pred_fg3a,
        p_rate=pred_fg3_rate,
        att_alpha=pred_att_alpha,
        rate_strength=rate_strength,
    )
    dispersion = variance_to_nbinom_alpha(pred_mean, pred_var)

    minutes_proj = today_feat_df["min_rolling_5"] if "min_rolling_5" in today_feat_df.columns else np.nan

    out = today_feat_df[["game_date", "player", "team", "opp"]].copy()
    out["stat"] = "fg3m"
    out["pred_fg3a"] = pred_fg3a
    out["pred_fg3_rate"] = pred_fg3_rate
    out["pred_mean_raw"] = pred_mean_raw
    out["pred_mean"] = pred_mean
    out["baseline_mean"] = baseline_mean
    out["delta_mean"] = out["pred_mean"] - out["baseline_mean"]
    out["dist_name"] = "nbinom"
    out["dispersion"] = dispersion
    out["minutes_proj"] = minutes_proj
    out["model_name"] = "fg3_composed_depaware_blended"
    out["model_version"] = "v3"

    calibrated = apply_threshold_calibrators(
        mu_pred=pred_mean,
        alpha_pred=dispersion,
        calibrators=calibrators,
    )
    for k in thresholds:
        out[f"p_ge_{k}_cal"] = calibrated.get(k, np.nan)

    out = _eligibility_flags(out)
    out = enforce_projection_schema(out)
    return out


def predict_fg3_player_means(
    *,
    history_df: pd.DataFrame,
    today_df: pd.DataFrame,
    model_dir: str | Path = THREES_MODEL_DIR,
) -> pd.DataFrame:
    bundle = _load_fg3_artifacts(model_dir)

    fg3a_model = bundle["fg3a_model"]
    fg3_rate_model = bundle["fg3_rate_model"]
    fg3a_dispersion_model = bundle["fg3a_dispersion_model"]
    calibrators = bundle["calibrators"]
    artifacts = bundle["artifacts"]
    is_v2 = bool(bundle["is_v2"])

    today_feat_df = build_feature_table(
        history_df=history_df,
        today_df=today_df,
        feature_builder=build_all_threes_features,
    )

    v2_ready = (
        is_v2
        and ("fg3a_used_features" in artifacts)
        and ("fg3rate_used_features" in artifacts)
        and ("fg3a_feature_medians" in artifacts)
        and ("fg3rate_feature_medians" in artifacts)
    )

    if v2_ready:
        return _predict_v2(
            today_feat_df=today_feat_df,
            fg3a_model=fg3a_model,
            fg3a_dispersion_model=fg3a_dispersion_model,
            fg3_rate_model=fg3_rate_model,
            calibrators=calibrators,
            artifacts=artifacts,
        )

    return _predict_v1(
        today_feat_df=today_feat_df,
        fg3a_model=fg3a_model,
        fg3_rate_model=fg3_rate_model,
        artifacts=artifacts,
    )