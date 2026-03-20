from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_training.common.feature_table import build_feature_table
from model_training.common.projection_schema import enforce_projection_schema
from model_training.config import POINTS_MODEL_DIR
from model_training.points.features import build_all_points_features
from model_training.points.models import (
    make_fg2_rate_baseline,
    make_fg2a_baseline,
    make_ft_rate_baseline,
    make_fta_baseline,
)
from model_training.threes.predict import predict_fg3_player_means


def _load_pts_artifacts(model_dir: str | Path) -> tuple[object, object, object, object, dict]:
    model_dir = Path(model_dir)

    fg2a_model = joblib.load(model_dir / "fg2a_model.joblib")
    fg2rate_model = joblib.load(model_dir / "fg2rate_model.joblib")
    fta_model = joblib.load(model_dir / "fta_model.joblib")
    ftrate_model = joblib.load(model_dir / "ftrate_model.joblib")

    with open(model_dir / "pts_artifacts.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    return fg2a_model, fg2rate_model, fta_model, ftrate_model, artifacts


def _prepare_feature_matrix(
    today_feat_df: pd.DataFrame,
    used_features: list[str],
    feature_medians: dict[str, float],
) -> pd.DataFrame:
    X = today_feat_df.copy()

    missing = [c for c in used_features if c not in X.columns]
    if missing:
        raise ValueError(f"today feature table missing required point features: {missing}")

    X = X[used_features].copy()

    for col in used_features:
        X[col] = X[col].fillna(feature_medians.get(col, 0.0))

    return X


def _eligibility_flags(player_pred_df: pd.DataFrame) -> pd.DataFrame:
    out = player_pred_df.copy()

    reasons: list[str] = []
    eligible: list[int] = []

    for row in out.itertuples(index=False):
        row_reasons = []

        minutes_proj = row.minutes_proj if pd.notna(row.minutes_proj) else np.nan
        pred_mean = row.pred_mean if pd.notna(row.pred_mean) else np.nan

        if pd.isna(minutes_proj) or minutes_proj < 12:
            row_reasons.append("minutes_lt_12")

        if pd.isna(pred_mean) or pred_mean < 5.0:
            row_reasons.append("pred_mean_lt_5")

        if row_reasons:
            eligible.append(0)
            reasons.append("|".join(row_reasons))
        else:
            eligible.append(1)
            reasons.append("ok")

    out["is_eligible"] = eligible
    out["eligibility_reason"] = reasons
    return out


def predict_pts_player_means(
    *,
    history_df: pd.DataFrame,
    today_df: pd.DataFrame,
    model_dir: str | Path = POINTS_MODEL_DIR,
) -> pd.DataFrame:
    fg2a_model, fg2rate_model, fta_model, ftrate_model, artifacts = _load_pts_artifacts(model_dir)

    today_feat_df = build_feature_table(
        history_df=history_df,
        today_df=today_df,
        feature_builder=build_all_points_features,
    )

    X_fg2a = _prepare_feature_matrix(
        today_feat_df,
        artifacts["fg2a_used_features"],
        artifacts["fg2a_feature_medians"],
    )
    X_fg2rate = _prepare_feature_matrix(
        today_feat_df,
        artifacts["fg2rate_used_features"],
        artifacts["fg2rate_feature_medians"],
    )
    X_fta = _prepare_feature_matrix(
        today_feat_df,
        artifacts["fta_used_features"],
        artifacts["fta_feature_medians"],
    )
    X_ftrate = _prepare_feature_matrix(
        today_feat_df,
        artifacts["ftrate_used_features"],
        artifacts["ftrate_feature_medians"],
    )

    pred_fg2a = np.clip(fg2a_model.predict(X_fg2a), 0.0, None)
    pred_fg2rate = np.clip(fg2rate_model.predict(X_fg2rate), 0.0, 1.0)
    pred_fta = np.clip(fta_model.predict(X_fta), 0.0, None)
    pred_ftrate = np.clip(ftrate_model.predict(X_ftrate), 0.0, 1.0)

    fg3_player_df = predict_fg3_player_means(
        history_df=history_df,
        today_df=today_df,
    )
    pred_fg3m = fg3_player_df["pred_mean"].to_numpy(dtype=float)

    pred_mean = np.clip(
        2.0 * pred_fg2a * pred_fg2rate
        + pred_fta * pred_ftrate
        + 3.0 * pred_fg3m,
        0.0,
        None,
    )

    baseline_mean = np.clip(
        2.0
        * np.nan_to_num(make_fg2a_baseline(today_feat_df), nan=0.0)
        * np.nan_to_num(make_fg2_rate_baseline(today_feat_df), nan=0.52)
        + np.nan_to_num(make_fta_baseline(today_feat_df), nan=0.0)
        * np.nan_to_num(make_ft_rate_baseline(today_feat_df), nan=0.78)
        + 3.0 * np.nan_to_num(fg3_player_df["baseline_mean"].to_numpy(dtype=float), nan=0.0),
        0.0,
        None,
    )

    minutes_proj = (
        today_feat_df["min_rolling_5"]
        if "min_rolling_5" in today_feat_df.columns
        else np.nan
    )
    dispersion = float(artifacts.get("dispersion_alpha_mom", 0.0))

    out = today_feat_df[["game_date", "player", "team", "opp"]].copy()
    out["stat"] = "pts"
    out["pred_mean"] = pred_mean
    out["baseline_mean"] = baseline_mean
    out["delta_mean"] = out["pred_mean"] - out["baseline_mean"]
    out["dist_name"] = "nbinom" if dispersion > 1e-12 else "poisson"
    out["dispersion"] = dispersion
    out["minutes_proj"] = minutes_proj
    out["model_name"] = "pts_composed"
    out["model_version"] = "v1"

    out = _eligibility_flags(out)
    out = enforce_projection_schema(out)
    return out