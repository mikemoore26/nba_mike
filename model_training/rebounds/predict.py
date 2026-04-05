from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from model_training.common.feature_table import build_feature_table
from model_training.common.projection_schema import enforce_projection_schema
from model_training.config import REBOUNDS_MODEL_DIR
from model_training.rebounds.features import build_all_rebounds_features
from model_training.rebounds.model import (
    apply_low_minutes_dampener,
    conditional_dispersion_alpha,
    make_rebounds_baseline,
)
from model_training.rebounds.tail_inference import (
    compute_blended_tail_probs,
    load_tail_models,
)


def _load_reb_artifacts(model_dir: str | Path) -> tuple[object, dict]:
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / "reb_model.joblib")
    with open(model_dir / "reb_artifacts.json", "r", encoding="utf-8") as f:
        artifacts = json.load(f)

    return model, artifacts


def _prepare_feature_matrix(today_feat_df, used_features, feature_medians):
    X = today_feat_df[used_features].copy()
    for col in used_features:
        X[col] = X[col].fillna(feature_medians.get(col, 0.0))
    return X


def _prob_ge(mu, k, dist_name, dispersion):
    mu = np.clip(mu, 1e-9, None)

    if dist_name == "poisson":
        return 1 - poisson.cdf(k - 1, mu)

    alpha = np.clip(dispersion, 1e-12, None)
    r = 1.0 / alpha
    p = r / (r + mu)

    return 1 - nbinom.cdf(k - 1, r, p)


def _enforce_monotonic_tail(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["p_reb_ge_10"] = np.minimum(df["p_reb_ge_10"], df["p_reb_ge_8"])
    df["p_reb_ge_12"] = np.minimum(df["p_reb_ge_12"], df["p_reb_ge_10"])

    return df


def predict_reb_player_means(
    *,
    history_df: pd.DataFrame,
    today_df: pd.DataFrame,
    model_dir: str | Path = REBOUNDS_MODEL_DIR,
) -> pd.DataFrame:

    model, artifacts = _load_reb_artifacts(model_dir)

    used_features = artifacts["used_features"]
    feature_medians = artifacts["feature_medians"]
    base_dispersion = float(artifacts.get("dispersion_alpha_mom", 0.0))

    today_feat_df = build_feature_table(
        history_df=history_df,
        today_df=today_df,
        feature_builder=build_all_rebounds_features,
    )

    X_today = _prepare_feature_matrix(today_feat_df, used_features, feature_medians)

    raw_pred_mean = np.clip(model.predict(X_today), 0.0, None)
    baseline_mean = make_rebounds_baseline(today_feat_df)
    minutes_proj = today_feat_df.get("min_rolling_5", np.nan)

    pred_mean = apply_low_minutes_dampener(raw_pred_mean, minutes_proj)

    dispersion = conditional_dispersion_alpha(
        base_alpha=base_dispersion,
        minutes_proj=minutes_proj,
        pred_mean=pred_mean,
    )

    out = today_feat_df[["game_date", "player", "team", "opp"]].copy()

    out["stat"] = "reb"
    out["pred_mean"] = pred_mean
    out["baseline_mean"] = baseline_mean
    out["delta_mean"] = pred_mean - baseline_mean
    out["dist_name"] = "nbinom" if base_dispersion > 1e-12 else "poisson"
    out["dispersion"] = dispersion
    out["minutes_proj"] = minutes_proj
    out["model_name"] = "reb_hgbr"
    out["model_version"] = "v4"

    # -------------------------
    # 🔥 NB PROBS
    # -------------------------
    p_nb_8 = _prob_ge(pred_mean, 8, out["dist_name"].iloc[0], dispersion)
    p_nb_10 = _prob_ge(pred_mean, 10, out["dist_name"].iloc[0], dispersion)
    p_nb_12 = _prob_ge(pred_mean, 12, out["dist_name"].iloc[0], dispersion)

    p_nb_dict = {8: p_nb_8, 10: p_nb_10, 12: p_nb_12}

    # -------------------------
    # 🔥 TAIL MODELS
    # -------------------------
    tail_models = load_tail_models(Path(model_dir))

    blended, tail_raw = compute_blended_tail_probs(
        df=today_feat_df,
        p_nb_dict=p_nb_dict,
        tail_models=tail_models,
        feature_cols=used_features,
        medians=feature_medians,
    )

    # -------------------------
    # 🔥 FINAL PROBS
    # -------------------------
    out["p_reb_ge_8"] = blended[8]
    out["p_reb_ge_10"] = blended[10]
    out["p_reb_ge_12"] = blended[12]

    # cap extreme spikes
    out["p_reb_ge_12"] = np.minimum(out["p_reb_ge_12"], 0.35)

    # enforce ordering
    out = _enforce_monotonic_tail(out)

    # debug columns
    out["p_reb_ge_8_nb_raw"] = p_nb_8
    out["p_reb_ge_10_nb_raw"] = p_nb_10
    out["p_reb_ge_12_nb_raw"] = p_nb_12

    out["p_reb_ge_8_tail_raw"] = tail_raw[8]
    out["p_reb_ge_10_tail_raw"] = tail_raw[10]
    out["p_reb_ge_12_tail_raw"] = tail_raw[12]

    out = enforce_projection_schema(out)

    return out