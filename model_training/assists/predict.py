# model_training/assists/predict.py
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_training.common.feature_table import build_feature_table
from model_training.common.projection_schema import enforce_projection_schema
from model_training.config import ASSISTS_MODEL_DIR
from model_training.assists.features import build_all_assists_features
from model_training.assists.model import make_assists_baseline


def _load_ast_artifacts(model_dir: str | Path) -> tuple[object, dict]:
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "ast_model.joblib")

    with open(model_dir / "ast_artifacts.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    return model, artifacts


def _prepare_feature_matrix(
    today_feat_df: pd.DataFrame,
    used_features: list[str],
    feature_medians: dict[str, float],
) -> pd.DataFrame:
    X = today_feat_df.copy()

    missing = [c for c in used_features if c not in X.columns]
    if missing:
        raise ValueError(f"today feature table missing required AST features: {missing}")

    X = X[used_features].copy()

    for col in used_features:
        fill_value = feature_medians.get(col, 0.0)
        X[col] = X[col].fillna(fill_value)

    return X


def _eligibility_flags(player_pred_df: pd.DataFrame) -> pd.DataFrame:
    out = player_pred_df.copy()

    reasons: list[str] = []
    eligible: list[int] = []

    for row in out.itertuples(index=False):
        row_reasons = []

        minutes_proj = row.minutes_proj if pd.notna(row.minutes_proj) else np.nan
        pred_mean = row.pred_mean if pd.notna(row.pred_mean) else np.nan
        baseline_mean = row.baseline_mean if pd.notna(row.baseline_mean) else np.nan

        if pd.isna(minutes_proj) or minutes_proj < 10:
            row_reasons.append("minutes_lt_10")

        if pd.isna(pred_mean) or pred_mean < 1.0:
            row_reasons.append("pred_mean_lt_1")

        if (pd.isna(minutes_proj) or minutes_proj <= 0) and (pd.isna(baseline_mean) or baseline_mean <= 0):
            row_reasons.append("no_role_signal")

        if row_reasons:
            eligible.append(0)
            reasons.append("|".join(row_reasons))
        else:
            eligible.append(1)
            reasons.append("ok")

    out["is_eligible"] = eligible
    out["eligibility_reason"] = reasons
    return out


def predict_ast_player_means(
    *,
    history_df: pd.DataFrame,
    today_df: pd.DataFrame,
    model_dir: str | Path = ASSISTS_MODEL_DIR,
) -> pd.DataFrame:
    model, artifacts = _load_ast_artifacts(model_dir)
    used_features = artifacts["used_features"]
    feature_medians = artifacts["feature_medians"]
    dispersion = float(artifacts.get("dispersion_alpha_mom", 0.0))

    today_feat_df = build_feature_table(
        history_df=history_df,
        today_df=today_df,
        feature_builder=build_all_assists_features,
    )

    X_today = _prepare_feature_matrix(today_feat_df, used_features, feature_medians)

    pred_mean = np.clip(model.predict(X_today), 0.0, None)
    baseline_mean = make_assists_baseline(today_feat_df)
    minutes_proj = today_feat_df["min_rolling_5"] if "min_rolling_5" in today_feat_df.columns else np.nan

    out = today_feat_df[["game_date", "player", "team", "opp"]].copy()
    out["stat"] = "ast"
    out["pred_mean"] = pred_mean
    out["baseline_mean"] = baseline_mean
    out["delta_mean"] = out["pred_mean"] - out["baseline_mean"]
    out["dist_name"] = "nbinom" if dispersion > 1e-12 else "poisson"
    out["dispersion"] = dispersion
    out["minutes_proj"] = minutes_proj
    out["model_name"] = "ast_hgbr"
    out["model_version"] = "v1"

    out = _eligibility_flags(out)
    out = enforce_projection_schema(out)
    return out