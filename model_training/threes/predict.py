# model_training/threes/predict.py
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from model_training.utils.team_codes import norm_team
from model_training.threes.probability import prob_ge_k
from model_training.threes.today_row import build_today_rows, build_today_rows_v2

from model_training.threes.features import (
    build_features_no_leak,
    add_player_baselines,
    add_opp_3p_defense_features_roll,
    add_team_stint_features,
)


# ---------------------------------------------------------
# Feature loader (threes)
# ---------------------------------------------------------
def load_feature_sets(features_path: str) -> tuple[list[str], list[str]]:
    feats_obj = joblib.load(features_path)

    if not isinstance(feats_obj, dict):
        raise TypeError(f"Unsupported threes features artifact type: {type(feats_obj)}")

    # support either naming convention
    if "FG3A_FEATURES" in feats_obj and "RATE_FEATURES" in feats_obj:
        return list(feats_obj["FG3A_FEATURES"]), list(feats_obj["RATE_FEATURES"])

    # fallback if you saved different keys
    if "FG3A_FEATS" in feats_obj and "RATE_FEATS" in feats_obj:
        return list(feats_obj["FG3A_FEATS"]), list(feats_obj["RATE_FEATS"])

    raise KeyError("Threes features artifact missing FG3A_FEATURES/RATE_FEATURES (or FG3A_FEATS/RATE_FEATS).")


def _season_from_date(game_date: pd.Timestamp) -> int:
    # season is end-year; Oct-Dec belong to next end-year
    gd = pd.Timestamp(game_date)
    return int(gd.year + (1 if gd.month >= 10 else 0))


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    sklearn pipelines with feature_names_in_ REQUIRE these columns exist.
    Create missing columns as NaN so imputers can handle them.
    """
    out = df.copy()
    missing = [c for c in cols if c not in out.columns]
    if missing:
        for c in missing:
            out[c] = np.nan
    return out


def _normalize_team(s: str) -> str:
    TEAM_MAP = {"NJN": "BKN", "CHO": "CHA"}
    s2 = str(norm_team(s)).upper().strip()
    return TEAM_MAP.get(s2, s2)


# ---------------------------------------------------------
# Main Prediction Function (THREES)
# ---------------------------------------------------------
def predict_game_fg3(
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    *,
    fg3a_model_path: str,
    fg3_rate_model_path: str,
    threes_features_path: str,
    # controls
    min_games_required: int = 10,
    recent_n: int = 5,
    use_v2: bool = True,
    min_min_rolling_5: float = 12.0,
    # break handling
    max_allowed_gap_days: int = 21,
    # baseline over-prob
    over_baseline_delta: int = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Outputs a slate for a single matchup with:
      pred_fg3a, pred_rate, pred_fg3,
      baseline_fg3, delta_fg3, p_over_baseline_{over_baseline_delta},
      plus p_ge_2 / p_ge_3

    Key fixes:
      - allows multi-day gaps (All-Star break)
      - tags today rows BEFORE feature engineering; extracts by tag (no tail() bug)
      - ensures expected feature columns exist for sklearn
    """
    game_date = pd.to_datetime(game_date)
    away_team = _normalize_team(away_team)
    home_team = _normalize_team(home_team)

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
                history[c].astype("string")
                .str.upper().str.strip()
                .map(_normalize_team)
            )

    history = history.sort_values(["player", "date"]).copy()

    # ----------------------------
    # Guard: allow breaks, block nonsense future dates
    # ----------------------------
    max_hist_date = history["date"].max()
    if pd.isna(max_hist_date):
        raise ValueError("history_df has no valid dates after parsing.")

    gap_days = int((game_date.normalize() - max_hist_date.normalize()).days)

    if gap_days > int(max_allowed_gap_days):
        raise ValueError(
            f"game_date={game_date.date()} is beyond history max date={max_hist_date.date()} "
            f"by {gap_days} days (limit={max_allowed_gap_days}). "
            "Update your gamelogs or pick a date within the dataset."
        )

    if gap_days > 1:
        print(
            f"[WARN] Predicting {gap_days} days after last history date "
            f"({max_hist_date.date()}) — likely a break/gap. Using last available form."
        )

    # ----------------------------
    # Season filter to avoid ghost rosters
    # ----------------------------
    season_of_date = _season_from_date(game_date)
    hist_season = history.loc[history["season"].astype(int) == int(season_of_date)].copy()
    if hist_season.empty:
        last_season = int(history["season"].astype(int).max())
        hist_season = history.loc[history["season"].astype(int) == last_season].copy()
    history = hist_season

    # ----------------------------
    # Load models + feature sets (FROM ARTIFACT)
    # ----------------------------
    fg3a_pipe = joblib.load(fg3a_model_path)
    fg3_rate_model = joblib.load(fg3_rate_model_path)

    FG3A_FEATS, RATE_FEATS = load_feature_sets(threes_features_path)
    ALL_FEATS = sorted(set(FG3A_FEATS) | set(RATE_FEATS))

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

    if today_df.empty:
        raise ValueError("today_df is empty (no players returned by today row builder).")

    # ----------------------------
    # Tag today rows BEFORE feature engineering (CRITICAL)
    # ----------------------------
    history2 = history.copy()
    today2 = today_df.copy()

    history2["__is_today"] = 0
    today2["__is_today"] = 1
    today2["__today_order"] = np.arange(len(today2), dtype=int)

    combined = pd.concat([history2, today2], ignore_index=True)

    # ----------------------------
    # Feature engineering
    # ----------------------------
    combined = build_features_no_leak(combined)
    combined = add_player_baselines(combined)
    combined = add_team_stint_features(combined)
    combined = add_opp_3p_defense_features_roll(combined)

    # Ensure model columns exist
    combined = _ensure_columns(combined, ALL_FEATS)

    # ----------------------------
    # Extract today's slice SAFELY (by tag)
    # ----------------------------
    X_today = (
        combined.loc[combined["__is_today"] == 1]
        .sort_values("__today_order")
        .reset_index(drop=True)
    )
    today_df = today_df.reset_index(drop=True)

    # ----------------------------
    # Gating
    # ----------------------------
    # core role/baseline gating
    # (these are the ones that caused your error)
    must = (
        X_today["min_rolling_5"].notna()
        & X_today["player_min_season_avg"].notna()
    ).to_numpy()

    X_ok = X_today.loc[must].copy()
    out = today_df.loc[must, ["player", "team", "opp", "is_home"]].copy()

    # rotation filter
    if min_min_rolling_5 is not None:
        mins = pd.to_numeric(X_ok["min_rolling_5"], errors="coerce").fillna(0.0).to_numpy()
        keep = mins >= float(min_min_rolling_5)
        X_ok = X_ok.loc[keep].copy()
        out = out.loc[keep].copy()

    if out.empty:
        if verbose:
            # show why
            dbg = X_today[["player","team","opp","is_home","min_rolling_5","player_min_season_avg"]].copy()
            print("\n[DEBUG] Today rows gating preview:")
            print(dbg.head(25).to_string(index=False))
        raise ValueError("No eligible players after feature gating (min_rolling_5/baselines missing).")

    # final ensure
    X_ok = _ensure_columns(X_ok, ALL_FEATS)

    if verbose:
        print("\n[DEBUG] X_ok rows:", len(X_ok))
        top_nan = X_ok[ALL_FEATS].isna().mean().sort_values(ascending=False).head(15)
        print("[DEBUG] Top NaN rates:\n", top_nan.to_string())

    # ----------------------------
    # Predict components
    # ----------------------------
    pred_fg3a = np.clip(fg3a_pipe.predict(X_ok[FG3A_FEATS]), 0, None)

    if hasattr(fg3_rate_model, "predict_p"):
        pred_rate = fg3_rate_model.predict_p(X_ok)
    else:
        pred_rate = np.clip(fg3_rate_model.predict(X_ok[RATE_FEATS]), 0, 1)
    pred_rate = np.clip(pred_rate, 0, 1)

    pred_fg3 = pred_fg3a * pred_rate

    # ----------------------------
    # Probabilities for 2+ / 3+ (Poisson approx)
    # ----------------------------
    p_ge_2 = prob_ge_k(pred_fg3, 2)
    p_ge_3 = prob_ge_k(pred_fg3, 3)

    # ----------------------------
    # Baseline + delta + p(over baseline + k)
    # ----------------------------
    base_a = pd.to_numeric(X_ok.get("player_fg3a_season_avg"), errors="coerce").fillna(0.0).to_numpy()
    base_p = pd.to_numeric(X_ok.get("player_fg3_pct_season"), errors="coerce").fillna(0.0).to_numpy()
    baseline_fg3 = base_a * base_p
    delta_fg3 = pred_fg3 - baseline_fg3

    thr = np.ceil(baseline_fg3 + int(over_baseline_delta)).astype(int)
    p_over_baseline = prob_ge_k(pred_fg3, thr)

    # ----------------------------
    # Output
    # ----------------------------
    out["pred_fg3a"] = pred_fg3a
    out["pred_rate"] = pred_rate
    out["pred_fg3"] = pred_fg3
    out["p_ge_2"] = p_ge_2
    out["p_ge_3"] = p_ge_3

    out["baseline_fg3"] = baseline_fg3
    out["delta_fg3"] = delta_fg3
    out[f"p_over_baseline_{int(over_baseline_delta)}"] = p_over_baseline

    return out.sort_values(
        ["p_ge_2", "pred_fg3", "pred_fg3a"],
        ascending=False,
    ).reset_index(drop=True)