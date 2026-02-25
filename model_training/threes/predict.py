# model_training/threes/predict.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from model_training.threes.today_row import build_today_rows, build_today_rows_v2
from model_training.threes.features import build_all_threes_features
from model_training.threes.probability import prob_ge_k
from model_training.common.eligibility import apply_eligibility_gate


# ----------------------------
# Helpers
# ----------------------------
def _ensure_date_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward/forward compat:
      - canonical: game_date
      - legacy: date
    We keep BOTH if possible so older code keeps working.
    """
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if "date" not in out.columns and "game_date" in out.columns:
        out["date"] = out["game_date"]

    if "game_date" not in out.columns and "date" in out.columns:
        out["game_date"] = out["date"]

    return out


def _coerce_id_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["team", "opp", "player"]:
        if c in out.columns:
            out[c] = out[c].astype("string")
    if "is_home" in out.columns:
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(int)
    return out


def load_feature_sets(features_path: str) -> tuple[list[str], list[str]]:
    """
    Supports:
      - dict artifact: {"FG3A_FEATURES": [...], "RATE_FEATURES": [...]}
      - legacy list artifact: [...]
    """
    feats_obj = joblib.load(features_path)

    if isinstance(feats_obj, dict) and "FG3A_FEATURES" in feats_obj and "RATE_FEATURES" in feats_obj:
        return list(feats_obj["FG3A_FEATURES"]), list(feats_obj["RATE_FEATURES"])

    if isinstance(feats_obj, (list, tuple)):
        feats_list = list(feats_obj)
        try:
            from model_training.threes.features import FG3A_FEATURES, RATE_FEATURES
            fg3a_feats = [c for c in FG3A_FEATURES if c in feats_list]
            rate_feats = [c for c in RATE_FEATURES if c in feats_list]
        except Exception:
            fg3a_feats = feats_list
            rate_feats = feats_list
        return fg3a_feats, rate_feats

    raise TypeError(f"Unsupported features artifact type: {type(feats_obj)}")


# ----------------------------
# Main
# ----------------------------
def predict_game_fg3(
    *,
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    fg3a_model_path: str,
    fg3_rate_model_path: str,
    threes_features_path: str,
    min_games_required: int = 10,
    recent_n: int = 5,
    fg3a_blend: float = 0.25,
    use_v2: bool = True,
    over_baseline_delta: float = 2.0,
    # gating controls
    min_games_prior_gate: int | None = None,
    min_expected_min_gate: float | None = None,
) -> pd.DataFrame:
    fg3a_pipe = joblib.load(fg3a_model_path)
    rate_pipe = joblib.load(fg3_rate_model_path)
    FG3A_FEATS, RATE_FEATS = load_feature_sets(threes_features_path)

    # ---- canonicalize history ----
    history = _ensure_date_cols(history_df)
    history = history.dropna(subset=["game_date"]).copy()
    history = _coerce_id_types(history)
    history = history.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    # ---- build today rows ----
    if use_v2:
        today_df = build_today_rows_v2(
            history,
            away_team,
            home_team,
            game_date,
            min_games_required=min_games_required,
            recent_n=recent_n,
        )
    else:
        today_df = build_today_rows(
            history,
            away_team,
            home_team,
            game_date,
            min_games_required=min_games_required,
            recent_n=recent_n,
        )

    today_df = _ensure_date_cols(today_df)
    today_df = _coerce_id_types(today_df)

    # ---- combine + feature build ----
    combined = pd.concat([history, today_df], ignore_index=True)
    combined = _ensure_date_cols(combined)
    combined = combined.sort_values(["player", "game_date"], kind="mergesort").reset_index(drop=True)

    combined = build_all_threes_features(combined)

    # slice out today's feature rows
    X_today = combined.tail(len(today_df)).reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # ---- unified gating (history/minutes/required baselines in one place) ----
    if min_games_prior_gate is None:
        min_games_prior_gate = min_games_required

    req_cols = [
        "min_rolling_5",
        "player_min_season_avg",
        "player_fg3a_season_avg",
        "player_fg3_pct_season",
    ]

    X_ok, rejects = apply_eligibility_gate(
        X_today,
        min_games_prior=int(min_games_prior_gate),
        min_expected_min=min_expected_min_gate,
        expected_min_col="expected_min_10",
        require_cols=req_cols,
    )

    if X_ok.empty:
        raise ValueError(
            "No eligible players after gating (history/minutes/required baselines).\n"
            f"away={away_team} home={home_team} game_date={game_date}\n"
            "Sample rejects:\n" + rejects.head(25).to_string(index=False)
        )

    # Align outputs: X_ok rows correspond to subset of X_today (same order/index after reset)
    # Since we reset_index on X_today/today_df, we must subset by integer index positions.
    out = today_df.loc[X_ok.index, ["player", "team", "opp", "is_home", "fg3a"]].copy()

    # ---- feature list checks (fail loud) ----
    missing_fg3a = [c for c in FG3A_FEATS if c not in X_ok.columns]
    missing_rate = [c for c in RATE_FEATS if c not in X_ok.columns]
    if missing_fg3a or missing_rate:
        raise ValueError(
            "Feature mismatch vs artifacts.\n"
            f"Missing FG3A features ({len(missing_fg3a)}): {missing_fg3a[:20]}\n"
            f"Missing RATE features ({len(missing_rate)}): {missing_rate[:20]}\n"
            "Fix: ensure build_all_threes_features() creates these columns OR retrain/resave features.joblib."
        )

    # ---- model predictions ----
    pred_fg3a_model = np.clip(fg3a_pipe.predict(X_ok[FG3A_FEATS]), 0, None)

    pred_rate = rate_pipe.predict(X_ok[RATE_FEATS])
    pred_rate = np.clip(np.asarray(pred_rate, dtype=float), 0, 1)

    expected_fg3a = out["fg3a"].to_numpy(dtype=float)
    pred_fg3a = (1.0 - float(fg3a_blend)) * expected_fg3a + float(fg3a_blend) * pred_fg3a_model
    pred_fg3a = np.clip(pred_fg3a, 0, None)

    mu = pred_fg3a * pred_rate

    baseline_fg3 = (
        X_ok["player_fg3a_season_avg"].to_numpy(dtype=float)
        * X_ok["player_fg3_pct_season"].to_numpy(dtype=float)
    )
    baseline_fg3 = np.clip(baseline_fg3, 0, None)

    delta_fg3 = mu - baseline_fg3

    threshold = np.ceil(baseline_fg3 + float(over_baseline_delta)).astype(int)
    p_over_baseline = prob_ge_k(mu, threshold)

    out = out.drop(columns=["fg3a"], errors="ignore")
    out["pred_fg3a"] = pred_fg3a
    out["pred_rate"] = pred_rate
    out["pred_fg3"] = mu
    out["baseline_fg3"] = baseline_fg3
    out["delta_fg3"] = delta_fg3
    out[f"p_over_baseline_{int(over_baseline_delta)}"] = p_over_baseline

    return out.sort_values(
        [f"p_over_baseline_{int(over_baseline_delta)}", "delta_fg3"],
        ascending=False,
    ).reset_index(drop=True)