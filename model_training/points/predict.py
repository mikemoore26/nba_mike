# model_training/points/predict.py
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from model_training.utils.team_codes import norm_team
from model_training.threes.probability import prob_ge_k

from model_training.points.features import (
    add_derived_2pt_cols,
    build_points_features_no_leak,
    add_player_baselines_points,
    add_opp_2p_defense_features_roll,
    add_opp_ft_defense_features_roll,
)

from model_training.threes.features import (
    build_features_no_leak as build_3p_features,
    add_player_baselines as add_3p_baselines,
    add_opp_3p_defense_features_roll,
    add_team_stint_features,
)

from model_training.threes.predict import load_feature_sets as load_3p_feature_sets
from model_training.threes.today_row import build_today_rows, build_today_rows_v2


# ---------------------------------------------------------
# Feature loader (points)
# ---------------------------------------------------------
def load_points_feature_sets(features_path: str):
    feats_obj = joblib.load(features_path)

    if not isinstance(feats_obj, dict):
        raise TypeError(f"Unsupported points features artifact type: {type(feats_obj)}")

    req = ["FG2A_FEATURES", "FG2_RATE_FEATURES", "FTA_FEATURES", "FT_RATE_FEATURES"]
    missing = [k for k in req if k not in feats_obj]
    if missing:
        raise KeyError(f"Points features artifact missing keys: {missing}")

    return (
        list(feats_obj["FG2A_FEATURES"]),
        list(feats_obj["FG2_RATE_FEATURES"]),
        list(feats_obj["FTA_FEATURES"]),
        list(feats_obj["FT_RATE_FEATURES"]),
    )


def _season_from_date(game_date: pd.Timestamp) -> int:
    # season is end-year; Oct-Dec belong to next end-year
    return int(game_date.year + (1 if game_date.month >= 10 else 0))


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Ensure columns exist. Missing columns are created as NaN so imputers can handle them.
    """
    out = df.copy()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        for c in missing:
            out[c] = np.nan
    return out


def _model_feats(model, fallback: list[str]) -> list[str]:
    """
    Prefer the model's training-time feature list when available.
    This prevents 'feature names seen at fit time, yet now missing' errors.
    """
    feats = getattr(model, "feature_names_in_", None)
    if feats is None:
        return list(fallback)
    return list(feats)


def _wrapper_feats(model, fallback: list[str]) -> list[str]:
    """
    Your LogitRateWrapper stores feature_names.
    Fall back to sklearn's feature_names_in_ or the artifact list.
    """
    if hasattr(model, "feature_names") and model.feature_names is not None:
        return list(model.feature_names)
    return _model_feats(model, fallback)


# ---------------------------------------------------------
# Main Prediction Function
# ---------------------------------------------------------
def predict_game_points(
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    # --- points models ---
    fg2a_model_path: str,
    fg2_rate_model_path: str,
    fta_model_path: str,
    ft_rate_model_path: str,
    points_features_path: str,
    # --- 3P models ---
    fg3a_model_path: str,
    fg3_rate_model_path: str,
    threes_features_path: str,
    # --- controls ---
    min_games_required: int = 10,
    recent_n: int = 5,
    over_line_delta: float = 3.0,
    use_v2: bool = True,
    min_min_rolling_5: float = 12.0,
) -> pd.DataFrame:
    # ----------------------------
    # Normalize inputs
    # ----------------------------
    game_date = pd.to_datetime(game_date)

    away_team = str(norm_team(away_team)).upper().strip()
    home_team = str(norm_team(home_team)).upper().strip()

    TEAM_MAP = {"NJN": "BKN", "CHO": "CHA"}
    away_team = TEAM_MAP.get(away_team, away_team)
    home_team = TEAM_MAP.get(home_team, home_team)

    # ----------------------------
    # Prepare history
    # ----------------------------
    history = history_df.copy()
    if "date" not in history.columns:
        raise ValueError("history_df missing required column: 'date'")
    if "season" not in history.columns:
        raise ValueError("history_df missing required column: 'season'")

    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).copy()

    for c in ["team", "opp"]:
        if c in history.columns:
            history[c] = (
                history[c]
                .astype("string")
                .str.upper()
                .str.strip()
                .replace(TEAM_MAP)
                .map(norm_team)
            )

    history = history.sort_values(["player", "date"]).copy()

    # Guard: dataset date range
    max_hist_date = history["date"].max()
    if pd.isna(max_hist_date):
        raise ValueError("history_df has no valid dates after parsing.")

    if game_date > (max_hist_date + pd.Timedelta(days=1)):
        raise ValueError(
            f"game_date={game_date.date()} is beyond history max date={max_hist_date.date()}. "
            "Update your gamelogs or pick a date within the dataset."
        )

    # Season filter (prevents ghost rosters)
    season_of_date = _season_from_date(game_date)
    hist_season = history.loc[history["season"].astype(int) == int(season_of_date)].copy()
    if hist_season.empty:
        last_season = int(history["season"].astype(int).max())
        hist_season = history.loc[history["season"].astype(int) == last_season].copy()
    history = hist_season

    # ----------------------------
    # Load models
    # ----------------------------
    fg2a_pipe = joblib.load(fg2a_model_path)
    fg2_rate_model = joblib.load(fg2_rate_model_path)
    fta_pipe = joblib.load(fta_model_path)
    ft_rate_model = joblib.load(ft_rate_model_path)

    fg3a_pipe = joblib.load(fg3a_model_path)
    fg3_rate_model = joblib.load(fg3_rate_model_path)

    # ----------------------------
    # Load feature sets (artifacts) THEN override with model feature_names_in_
    # ----------------------------
    FG2A_FEATS_art, FG2_RATE_FEATS_art, FTA_FEATS_art, FT_RATE_FEATS_art = load_points_feature_sets(points_features_path)
    FG3A_FEATS_art, RATE_FEATS_art = load_3p_feature_sets(threes_features_path)

    # IMPORTANT: use model's training features when possible
    FG2A_FEATS = _model_feats(fg2a_pipe, FG2A_FEATS_art)
    FTA_FEATS = _model_feats(fta_pipe, FTA_FEATS_art)

    FG2_RATE_FEATS = _wrapper_feats(fg2_rate_model, FG2_RATE_FEATS_art)
    FT_RATE_FEATS = _wrapper_feats(ft_rate_model, FT_RATE_FEATS_art)

    # 3P: same idea
    FG3A_FEATS = _model_feats(fg3a_pipe, FG3A_FEATS_art)
    RATE_FEATS = _wrapper_feats(fg3_rate_model, RATE_FEATS_art)

    # union of all features we must materialize for sklearn checks
    ALL_FEATS = sorted(
        set(FG2A_FEATS)
        | set(FG2_RATE_FEATS)
        | set(FTA_FEATS)
        | set(FT_RATE_FEATS)
        | set(FG3A_FEATS)
        | set(RATE_FEATS)
    )

    # ----------------------------
    # Build today's rows
    # ----------------------------
    if use_v2:
        try:
            today_df = build_today_rows_v2(
                history,
                away_team,
                home_team,
                game_date,
                min_games_required=min_games_required,
                recent_n=recent_n,
            )
        except ValueError:
            today_df = build_today_rows(
                history,
                away_team,
                home_team,
                game_date,
                min_games_required=max(3, min_games_required // 2),
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

    # ----------------------------
    # Tag today rows BEFORE feature engineering (prevents sort/tail bugs)
    # ----------------------------
    history2 = history.copy()
    today2 = today_df.copy()

    history2["__is_today"] = 0
    today2["__is_today"] = 1
    today2["__today_order"] = np.arange(len(today2), dtype=int)

    combined = pd.concat([history2, today2], ignore_index=True)

    # ----------------------------
    # Feature build (points + stints + opponent + threes context)
    # ----------------------------
    combined = add_derived_2pt_cols(combined)
    combined = build_points_features_no_leak(combined)
    combined = add_player_baselines_points(combined)
    combined = add_team_stint_features(combined)
    combined = add_opp_2p_defense_features_roll(combined)
    combined = add_opp_ft_defense_features_roll(combined)

    combined = build_3p_features(combined)
    combined = add_3p_baselines(combined)
    combined = add_opp_3p_defense_features_roll(combined)

    # Ensure all expected columns exist (CRITICAL)
    combined = _ensure_columns(combined, ALL_FEATS)

    # ----------------------------
    # Extract today's slice by tag
    # ----------------------------
    X_today = (
        combined.loc[combined["__is_today"] == 1]
        .sort_values("__today_order")
        .reset_index(drop=True)
    )
    today_df = today_df.reset_index(drop=True)

    # Base gating
    mask = (
        X_today["min_rolling_5"].notna()
        & X_today["player_min_season_avg"].notna()
    ).to_numpy()

    X_ok = X_today.loc[mask].copy()
    out = today_df.loc[mask, ["player", "team", "opp", "is_home"]].copy()

    # Rotation filter
    if min_min_rolling_5 is not None:
        keep = (
            pd.to_numeric(X_ok["min_rolling_5"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
            >= float(min_min_rolling_5)
        )
        X_ok = X_ok.loc[keep].copy()
        out = out.loc[keep].copy()

    if out.empty:
        raise ValueError("No eligible players after feature gating/rotation filter.")

    # Ensure right before prediction too
    X_ok = _ensure_columns(X_ok, ALL_FEATS)

    # ----------------------------
    # Predict components
    # ----------------------------
    pred_fg2a = np.clip(fg2a_pipe.predict(X_ok[FG2A_FEATS]), 0, None)

    if hasattr(fg2_rate_model, "predict_p"):
        pred_fg2_rate = fg2_rate_model.predict_p(X_ok)
    else:
        pred_fg2_rate = np.clip(fg2_rate_model.predict(X_ok[FG2_RATE_FEATS]), 0, 1)
    pred_fg2_rate = np.clip(pred_fg2_rate, 0, 1)

    pred_fta = np.clip(fta_pipe.predict(X_ok[FTA_FEATS]), 0, None)

    if hasattr(ft_rate_model, "predict_p"):
        pred_ft_rate = ft_rate_model.predict_p(X_ok)
    else:
        pred_ft_rate = np.clip(ft_rate_model.predict(X_ok[FT_RATE_FEATS]), 0, 1)
    pred_ft_rate = np.clip(pred_ft_rate, 0, 1)

    pred_fg3a = np.clip(fg3a_pipe.predict(X_ok[FG3A_FEATS]), 0, None)

    if hasattr(fg3_rate_model, "predict_p"):
        pred_fg3_rate = fg3_rate_model.predict_p(X_ok)
    else:
        pred_fg3_rate = np.clip(fg3_rate_model.predict(X_ok[RATE_FEATS]), 0, 1)
    pred_fg3_rate = np.clip(pred_fg3_rate, 0, 1)

    # ----------------------------
    # Combine to expected points
    # ----------------------------
    mu_fg2 = pred_fg2a * pred_fg2_rate
    mu_fg3 = pred_fg3a * pred_fg3_rate
    mu_ft = pred_fta * pred_ft_rate

    mu_pts = 2 * mu_fg2 + 3 * mu_fg3 + mu_ft

    # variance approximation
    var_fg2 = pred_fg2a * pred_fg2_rate * (1 - pred_fg2_rate)
    var_fg3 = pred_fg3a * pred_fg3_rate * (1 - pred_fg3_rate)
    var_ft = pred_fta * pred_ft_rate * (1 - pred_ft_rate)

    var_pts = 4 * var_fg2 + 9 * var_fg3 + var_ft
    sd_pts = np.sqrt(np.clip(var_pts, 0, None))

    # baseline + delta
    baseline_pts = (
        2
        * pd.to_numeric(X_ok.get("player_fg2a_season_avg"), errors="coerce").fillna(0.0).to_numpy()
        * pd.to_numeric(X_ok.get("player_fg2_pct_season"), errors="coerce").fillna(0.0).to_numpy()
        + 3
        * pd.to_numeric(X_ok.get("player_fg3a_season_avg"), errors="coerce").fillna(0.0).to_numpy()
        * pd.to_numeric(X_ok.get("player_fg3_pct_season"), errors="coerce").fillna(0.0).to_numpy()
        + pd.to_numeric(X_ok.get("player_fta_season_avg"), errors="coerce").fillna(0.0).to_numpy()
        * pd.to_numeric(X_ok.get("player_ft_pct_season"), errors="coerce").fillna(0.0).to_numpy()
    )

    delta_pts = mu_pts - baseline_pts

    threshold = np.ceil(baseline_pts + float(over_line_delta)).astype(int)
    p_over = prob_ge_k(mu_pts, threshold)

    # ----------------------------
    # Output
    # ----------------------------
    out["pred_fg2a"] = pred_fg2a
    out["pred_fg2_rate"] = pred_fg2_rate
    out["pred_fta"] = pred_fta
    out["pred_ft_rate"] = pred_ft_rate
    out["pred_fg3a"] = pred_fg3a
    out["pred_fg3_rate"] = pred_fg3_rate

    out["pred_pts"] = mu_pts
    out["sd_pts"] = sd_pts
    out["baseline_pts"] = baseline_pts
    out["delta_pts"] = delta_pts
    out[f"p_over_baseline_{int(over_line_delta)}"] = p_over

    return out.sort_values(
        [f"p_over_baseline_{int(over_line_delta)}", "delta_pts"],
        ascending=False,
    ).reset_index(drop=True)
