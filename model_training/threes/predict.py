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
from model_training.threes.models import make_fg3_rate_baseline, make_fg3a_baseline


def _load_fg3_artifacts(model_dir: str | Path) -> tuple[object, object, dict]:
    model_dir = Path(model_dir)

    fg3a_model = joblib.load(model_dir / "fg3a_model.joblib")
    fg3_rate_model = joblib.load(model_dir / "fg3_rate_model.joblib")

    with open(model_dir / "fg3_artifacts.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    return fg3a_model, fg3_rate_model, artifacts


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


def predict_fg3_player_means(
    *,
    history_df: pd.DataFrame,
    today_df: pd.DataFrame,
    model_dir: str | Path = THREES_MODEL_DIR,
) -> pd.DataFrame:
    fg3a_model, fg3_rate_model, artifacts = _load_fg3_artifacts(model_dir)

    fg3a_used_features = artifacts["fg3a_used_features"]
    fg3a_feature_medians = artifacts["fg3a_feature_medians"]

    fg3rate_used_features = artifacts["fg3rate_used_features"]
    fg3rate_feature_medians = artifacts["fg3rate_feature_medians"]

    dispersion = float(artifacts.get("dispersion_alpha_mom", 0.0))

    today_feat_df = build_feature_table(
        history_df=history_df,
        today_df=today_df,
        feature_builder=build_all_threes_features,
    )

    X_att = _prepare_feature_matrix(today_feat_df, fg3a_used_features, fg3a_feature_medians)
    X_rate = _prepare_feature_matrix(today_feat_df, fg3rate_used_features, fg3rate_feature_medians)

    pred_fg3a = np.clip(fg3a_model.predict(X_att), 0.0, None)
    pred_fg3_rate = np.clip(fg3_rate_model.predict(X_rate), 0.0, 1.0)
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