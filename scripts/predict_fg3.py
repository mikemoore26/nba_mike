# model_training/threes/predict.py
# Replace your imports at top with this (safe + aligned)

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from model_training.threes.today_row import build_today_rows, build_today_rows_v2
from model_training.threes.features import build_all_threes_features  # <- use the one-stop builder
from model_training.threes.probability import prob_ge_k

from model_training.common.feature_table import load_gamelogs, build_feature_table


def load_feature_sets(features_path: str):
    feats_obj = joblib.load(features_path)

    if isinstance(feats_obj, dict) and "FG3A_FEATURES" in feats_obj and "RATE_FEATURES" in feats_obj:
        return feats_obj["FG3A_FEATURES"], feats_obj["RATE_FEATURES"]

    # backward compat: older artifacts saved a single list
    if isinstance(feats_obj, (list, tuple)):
        feats_list = list(feats_obj)
        # if you still have these lists in features.py, import them; otherwise just use feats_list for both
        try:
            from model_training.threes.features import FG3A_FEATURES, RATE_FEATURES
            fg3a_feats = [c for c in FG3A_FEATURES if c in feats_list]
            rate_feats = [c for c in RATE_FEATURES if c in feats_list]
        except Exception:
            fg3a_feats = feats_list
            rate_feats = feats_list
        return fg3a_feats, rate_feats

    raise TypeError(f"Unsupported features artifact type: {type(feats_obj)}")


def predict_game_fg3(
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    fg3a_model_path: str,
    fg3_rate_model_path: str,
    features_path: str,
    min_games_required: int = 10,
    recent_n: int = 5,
    fg3a_blend: float = 0.25,
    use_v2: bool = True,
    over_baseline_delta: float = 2.0,
) -> pd.DataFrame:
    fg3a_pipe = joblib.load(fg3a_model_path)
    rate_pipe = joblib.load(fg3_rate_model_path)
    FG3A_FEATS, RATE_FEATS = load_feature_sets(features_path)

    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history["team"] = history["team"].astype("string")
    history["opp"] = history["opp"].astype("string")
    history["player"] = history["player"].astype("string")
    history = history.sort_values(["player", "date"])

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

    combined = pd.concat([history, today_df], ignore_index=True)

    # ONE STOP: builds all rolling + baselines + stint + opp defense
    combined = build_all_threes_features(combined)

    X_today = combined.tail(len(today_df)).reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # Guard: stable baseline + minutes signal
    mask = (
        X_today["min_rolling_5"].notna()
        & X_today["player_min_season_avg"].notna()
        & X_today["player_fg3a_season_avg"].notna()
        & X_today["player_fg3_pct_season"].notna()
    ).to_numpy()

    X_ok = X_today.loc[mask].copy()
    out = today_df.loc[mask, ["player", "team", "opp", "is_home", "fg3a"]].copy()

    # Model predictions
    pred_fg3a_model = np.clip(fg3a_pipe.predict(X_ok[FG3A_FEATS]), 0, None)

    # NOTE: if your rate model is saved as LogitRateWrapper, .predict returns prob already.
    pred_rate = rate_pipe.predict(X_ok[RATE_FEATS])
    pred_rate = np.clip(np.asarray(pred_rate, dtype=float), 0, 1)

    # Blend heuristic attempts with model attempts
    expected_fg3a = out["fg3a"].to_numpy(dtype=float)
    pred_fg3a = (1.0 - fg3a_blend) * expected_fg3a + fg3a_blend * pred_fg3a_model
    pred_fg3a = np.clip(pred_fg3a, 0, None)

    mu = pred_fg3a * pred_rate

    # Baseline expectation (player normal)
    baseline_fg3 = (
        X_ok["player_fg3a_season_avg"].to_numpy(dtype=float)
        * X_ok["player_fg3_pct_season"].to_numpy(dtype=float)
    )
    baseline_fg3 = np.clip(baseline_fg3, 0, None)

    delta_fg3 = mu - baseline_fg3

    # Probability of beating baseline by +delta
    threshold = np.ceil(baseline_fg3 + float(over_baseline_delta)).astype(int)
    p_over_baseline = prob_ge_k(mu, threshold)

    out["pred_fg3a"] = pred_fg3a
    out["pred_rate"] = pred_rate
    out["pred_fg3"] = mu
    out["baseline_fg3"] = baseline_fg3
    out["delta_fg3"] = delta_fg3
    out[f"p_over_baseline_{int(over_baseline_delta)}"] = p_over_baseline

    out = out.drop(columns=["fg3a"])

    return out.sort_values(
        [f"p_over_baseline_{int(over_baseline_delta)}", "delta_fg3"],
        ascending=False
    ).reset_index(drop=True)