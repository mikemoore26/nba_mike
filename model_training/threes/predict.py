# model_training/threes/predict.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from model_training.threes.today_row import build_today_rows, build_today_rows_v2
from model_training.threes.features import build_all_threes_features
from model_training.threes.probability import prob_ge_k
from model_training.common.eligibility import apply_eligibility_gate


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" not in out.columns:
        if "date" in out.columns:
            out["game_date"] = out["date"]
        else:
            raise ValueError("Expected `game_date` or legacy `date`.")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()
    return out


def _coerce_id_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["team", "opp", "player"]:
        if c in out.columns:
            out[c] = out[c].astype("string")
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(int)
    return out


def _model_expected_features(pipe) -> list[str] | None:
    try:
        return list(pipe.feature_names_in_)
    except Exception:
        pass
    try:
        return list(pipe.named_steps["model"].feature_names_in_)
    except Exception:
        pass
    return None


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def predict_game_fg3(
    *,
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    fg3a_model_path: str,
    fg3_rate_model_path: str,
    threes_features_path: str,  # kept for compatibility
    min_games_required: int = 10,
    recent_n: int = 5,
    fg3a_blend: float = 0.25,
    use_v2: bool = True,
    over_baseline_delta: float = 2.0,
) -> pd.DataFrame:

    fg3a_pipe = joblib.load(fg3a_model_path)
    rate_pipe = joblib.load(fg3_rate_model_path)

    history = _coerce_id_types(_ensure_game_date(history_df))
    history = history.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    # ----------------------------
    # Build today rows
    # ----------------------------
    if use_v2:
        today_df = build_today_rows_v2(
            history, away_team, home_team, game_date,
            min_games_required=min_games_required,
            recent_n=recent_n,
        )
    else:
        today_df = build_today_rows(
            history, away_team, home_team, game_date,
            min_games_required=min_games_required,
            recent_n=recent_n,
        )

    today_df = _coerce_id_types(_ensure_game_date(today_df))

    # ----------------------------
    # Combine + build features
    # ----------------------------
    combined = pd.concat([history, today_df], ignore_index=True)
    combined = combined.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    combined["games_played_prior"] = combined.groupby("player").cumcount()

    combined = build_all_threes_features(combined)

    X_today = combined.tail(len(today_df)).reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # ----------------------------
    # Basic gating
    # ----------------------------
    X_ok, rejects = apply_eligibility_gate(
        X_today,
        min_games_prior=min_games_required,
        expected_min_col="expected_min_10",
        require_cols=[
            "min_rolling_5",
            "player_min_season_avg",
            "player_fg3a_season_avg",
            "player_fg3_pct_season",
        ],
    )

    if X_ok.empty:
        raise ValueError(
            "No eligible players after gating.\n"
            + rejects.head(20).to_string(index=False)
        )

    # ----------------------------
    # Determine model feature lists
    # ----------------------------
    fg3a_feats = _model_expected_features(fg3a_pipe)
    rate_feats = list(getattr(rate_pipe, "feature_names", []))

    if fg3a_feats is None:
        raise RuntimeError("Could not determine FG3A model feature names.")

    if not rate_feats:
        raise RuntimeError("Could not determine RATE model feature names.")

    # ensure required columns exist
    X_ok = _ensure_cols(X_ok, fg3a_feats)
    X_ok = _ensure_cols(X_ok, rate_feats)

    # ----------------------------
    # Predictions
    # ----------------------------
    pred_fg3a_model = np.clip(fg3a_pipe.predict(X_ok[fg3a_feats]), 0, None)

    # wrapper-safe rate prediction
    if hasattr(rate_pipe, "predict_p"):
        pred_rate = rate_pipe.predict_p(X_ok[rate_feats])
    else:
        pred_rate = rate_pipe.predict(X_ok[rate_feats])

    pred_rate = np.clip(np.asarray(pred_rate, dtype=float), 0, 1)

    # heuristic attempts from today rows
    out = today_df.loc[X_ok.index, ["player", "team", "opp", "is_home", "fg3a", "game_date"]].copy()

    expected_fg3a = out["fg3a"].to_numpy(dtype=float)
    pred_fg3a = (1 - fg3a_blend) * expected_fg3a + fg3a_blend * pred_fg3a_model
    pred_fg3a = np.clip(pred_fg3a, 0, None)

    mu = pred_fg3a * pred_rate

    baseline_fg3 = (
        X_ok["player_fg3a_season_avg"].to_numpy(dtype=float)
        * X_ok["player_fg3_pct_season"].to_numpy(dtype=float)
    )
    baseline_fg3 = np.clip(baseline_fg3, 0, None)

    delta_fg3 = mu - baseline_fg3

    threshold = np.ceil(baseline_fg3 + over_baseline_delta).astype(int)
    p_over = prob_ge_k(mu, threshold)

    # ----------------------------
    # Assemble output
    # ----------------------------
    out = out.drop(columns=["fg3a"])
    out["pred_fg3a"] = pred_fg3a
    out["pred_rate"] = pred_rate
    out["pred_fg3"] = mu
    out["baseline_fg3"] = baseline_fg3
    out["delta_fg3"] = delta_fg3
    out[f"p_over_{int(over_baseline_delta)}"] = p_over

    from model_training.common.pred_schema import PredSchema, standardize_pred_df, validate_pred_schema

    pred_df = standardize_pred_df(
        out,
        schema=PredSchema(stat_name="fg3", model_name="threes", model_version="v1"),
        mean_col="pred_fg3",
        baseline_col="baseline_fg3",
        delta_col="delta_fg3",
        extra_keep=("pred_fg3a", "pred_rate"),
    )
    validate_pred_schema(pred_df)
    return pred_df.sort_values(["pred_mean"], ascending=False).reset_index(drop=True)

    