# model_training/threes/predict.py
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from model_training.threes.today_row import build_today_rows, build_today_rows_v2
from model_training.threes.features import build_all_threes_features  # one-stop builder
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
        # keep as int/bool friendly
        out["is_home"] = out["is_home"].astype(int, errors="ignore") if hasattr(out["is_home"], "astype") else out["is_home"]
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

    # backward compat: older artifacts saved a single list
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
    """
    Predicts:
      - pred_fg3a (blended attempts)
      - pred_rate (3P% rate)
      - pred_fg3 (mu = attempts * rate)
      - baseline_fg3, delta_fg3
      - p_over_baseline_{delta}

    Leakage notes:
      - history is strictly historical game logs
      - today rows are appended then features are built once on combined
      - gating uses leakage-safe prior game count (cumcount) and/or expected minutes proxy if present
    """
    fg3a_pipe = joblib.load(fg3a_model_path)
    rate_pipe = joblib.load(fg3_rate_model_path)
    FG3A_FEATS, RATE_FEATS = load_feature_sets(threes_features_path)

    # ---- canonicalize history ----
    history = _ensure_date_cols(history_df)
    history = history.dropna(subset=["date"]).copy()
    history = _coerce_id_types(history)

    # stable sort for rolling features / cumcount gating
    history = history.sort_values(["player", "date"], kind="mergesort").reset_index(drop=True)

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
    combined = combined.sort_values(["player", "date"], kind="mergesort").reset_index(drop=True)

    # ONE STOP: rolling + baselines + opp context (must be no-leak inside builder)
    combined = build_all_threes_features(combined)

    # slice out today's feature rows (same order as today_df by construction)
    X_today = combined.tail(len(today_df)).reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # ---- gating (replace fragile hand-mask with shared gate + explicit reasons) ----
    # If user didn't set a gate, keep your old behavior-ish:
    # - min_games_required already enforced in build_today_rows, but we still gate for missing baselines.
    if min_games_prior_gate is None:
        # keep consistent with earlier default min_games_required
        min_games_prior_gate = min_games_required

    X_today_gated, rejects = apply_eligibility_gate(
        X_today,
        min_games_prior=int(min_games_prior_gate),
        min_expected_min=min_expected_min_gate,
        expected_min_col="expected_min_10",
    )

    if X_today_gated.empty:
        raise ValueError(
            "No eligible players after gating.\n"
            f"away={away_team} home={home_team} game_date={game_date}\n"
            "Sample rejects:\n" + rejects.head(25).to_string(index=False)
        )

    # Also require key baselines for stable baseline math / rate model
    req_baselines = [
        "min_rolling_5",
        "player_min_season_avg",
        "player_fg3a_season_avg",
        "player_fg3_pct_season",
    ]
    miss = [c for c in req_baselines if c not in X_today_gated.columns]
    if miss:
        raise ValueError(f"Missing required baseline cols in X_today after feature build: {miss}")

    baseline_mask = (
        X_today_gated["min_rolling_5"].notna()
        & X_today_gated["player_min_season_avg"].notna()
        & X_today_gated["player_fg3a_season_avg"].notna()
        & X_today_gated["player_fg3_pct_season"].notna()
    ).to_numpy()

    X_ok = X_today_gated.loc[baseline_mask].copy()
    if X_ok.empty:
        # keep explicit; this is where you used to silently die
        sample = X_today_gated.loc[~baseline_mask, ["player", "team", "opp", "is_home"]].head(25)
        raise ValueError(
            "All eligible players failed baseline availability checks.\n"
            f"Missing/NaN in: {req_baselines}\n"
            "Sample failed rows:\n" + sample.to_string(index=False)
        )

    # Align outputs to X_ok rows
    # NOTE: today_df might be longer than X_today_gated; we map by index via original X_today row index.
    # We preserved index through apply_eligibility_gate (it returns copies with same index).
    out_cols = [c for c in ["player", "team", "opp", "is_home", "fg3a"] if c in today_df.columns]
    # safer: take from X_ok if fg3a is not present in today_df (depends on your today_row builder)
    if "fg3a" not in today_df.columns and "fg3a" in X_ok.columns:
        out_cols = [c for c in ["player", "team", "opp", "is_home"] if c in X_ok.columns] + ["fg3a"]

    if "fg3a" in out_cols and "fg3a" not in today_df.columns:
        out = X_ok[out_cols].copy()
    else:
        # reindex today_df to X_ok index where possible
        # If today_df has default RangeIndex, we fall back to X_ok for ids.
        try:
            out = today_df.loc[X_ok.index, out_cols].copy()
        except Exception:
            out = X_ok[[c for c in ["player", "team", "opp", "is_home"] if c in X_ok.columns]].copy()
            if "fg3a" in X_ok.columns:
                out["fg3a"] = X_ok["fg3a"].to_numpy()

    # ---- feature list checks (fail loud, not silent) ----
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

    # Rate model: may return prob already depending on wrapper
    pred_rate = rate_pipe.predict(X_ok[RATE_FEATS])
    pred_rate = np.clip(np.asarray(pred_rate, dtype=float), 0, 1)

    # Blend heuristic attempts (expected from today rows) with model attempts
    if "fg3a" not in out.columns:
        raise ValueError("today rows must provide 'fg3a' heuristic attempts column for blending.")
    expected_fg3a = out["fg3a"].to_numpy(dtype=float)

    pred_fg3a = (1.0 - float(fg3a_blend)) * expected_fg3a + float(fg3a_blend) * pred_fg3a_model
    pred_fg3a = np.clip(pred_fg3a, 0, None)

    mu = pred_fg3a * pred_rate

    # ---- baseline + delta + probability ----
    baseline_fg3 = (
        X_ok["player_fg3a_season_avg"].to_numpy(dtype=float)
        * X_ok["player_fg3_pct_season"].to_numpy(dtype=float)
    )
    baseline_fg3 = np.clip(baseline_fg3, 0, None)

    delta_fg3 = mu - baseline_fg3

    threshold = np.ceil(baseline_fg3 + float(over_baseline_delta)).astype(int)
    p_over_baseline = prob_ge_k(mu, threshold)

    # ---- assemble output ----
    out = out.drop(columns=["fg3a"], errors="ignore")
    out["pred_fg3a"] = pred_fg3a
    out["pred_rate"] = pred_rate
    out["pred_fg3"] = mu
    out["baseline_fg3"] = baseline_fg3
    out["delta_fg3"] = delta_fg3
    out[f"p_over_baseline_{int(over_baseline_delta)}"] = p_over_baseline

    return (
        out.sort_values(
            [f"p_over_baseline_{int(over_baseline_delta)}", "delta_fg3"],
            ascending=False,
        )
        .reset_index(drop=True)
    )