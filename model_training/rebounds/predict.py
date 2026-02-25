# model_training/rebounds/predict.py
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from model_training.utils.team_codes import norm_team
from model_training.rebounds.probability import prob_ge_k_nbinom
from model_training.rebounds.features import build_features_no_leak, REB_FEATURES
from model_training.rebounds.today_row import build_today_rows, build_today_rows_v2


def load_feature_set(features_path: str):
    feats_obj = joblib.load(features_path)

    if isinstance(feats_obj, dict) and "REB_FEATURES" in feats_obj:
        feats = feats_obj["REB_FEATURES"]
        alpha = float(feats_obj.get("dispersion_alpha", 0.25))
        return list(feats), alpha

    raise TypeError(f"Unsupported features artifact type: {type(feats_obj)}")


def predict_game_reb(
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    reb_model_path: str,
    features_path: str,
    min_games_required: int = 10,
    recent_n: int = 5,
    reb_blend: float = 0.25,
    use_v2: bool = True,
    over_baseline_delta: float = 2.0,
) -> pd.DataFrame:

    away_team = str(norm_team(away_team)).upper().strip()
    home_team = str(norm_team(home_team)).upper().strip()

    reb_pipe = joblib.load(reb_model_path)
    REB_FEATS, alpha = load_feature_set(features_path)

    history = history_df.copy()
    if "date" not in history.columns:
        raise ValueError("history_df missing required column: 'date'")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date", "player"]).copy()
    history = history.sort_values(["player", "date"])

    # --- build today's rows (v2 strict -> v1 fallback) ---
    if use_v2:
        try:
            today_df = build_today_rows_v2(
                history, away_team, home_team, game_date,
                min_games_required=min_games_required,
                recent_n=recent_n,
            )
        except ValueError:
            today_df = build_today_rows(
                history, away_team, home_team, game_date,
                min_games_required=max(3, min_games_required // 2),
                recent_n=recent_n,
            )
    else:
        today_df = build_today_rows(
            history, away_team, home_team, game_date,
            min_games_required=min_games_required,
            recent_n=recent_n,
        )

    # --- feature build on combined ---
    combined = pd.concat([history, today_df], ignore_index=True)
    combined = build_features_no_leak(combined)

    X_today = combined.tail(len(today_df)).reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # --- gating (primary) ---
    mask = (X_today["min_rolling_5"].notna() & X_today["reb_rolling_5"].notna()).to_numpy()

    # --- fallback gating: if everything missing, allow imputer to work ---
    if not mask.any():
        mask = X_today[REB_FEATS].notna().any(axis=1).to_numpy()

    X_ok = X_today.loc[mask].copy()
    out = today_df.loc[mask, ["player", "team", "opp", "is_home", "reb"]].copy()

    if out.empty:
        raise ValueError("No eligible players after feature gating.")

    pred_reb_model = np.clip(reb_pipe.predict(X_ok[REB_FEATS]), 0, None)

    expected_reb = pd.to_numeric(out["reb"], errors="coerce").fillna(0).to_numpy()
    pred_reb = (1.0 - reb_blend) * expected_reb + reb_blend * pred_reb_model
    pred_reb = np.clip(pred_reb, 0, None)

    baseline_reb = X_ok["player_reb_season_avg"].fillna(0).to_numpy()
    delta_reb = pred_reb - baseline_reb

    threshold = np.ceil(baseline_reb + over_baseline_delta).astype(int)
    p_over_baseline = prob_ge_k_nbinom(pred_reb, threshold, alpha=alpha)

    out["pred_reb"] = pred_reb
    out["baseline_reb"] = baseline_reb
    out["delta_reb"] = delta_reb
    out[f"p_over_baseline_{int(over_baseline_delta)}"] = p_over_baseline

    out = out.drop(columns=["reb"])

    return out.sort_values(
        [f"p_over_baseline_{int(over_baseline_delta)}", "delta_reb"],
        ascending=False
    ).reset_index(drop=True)
