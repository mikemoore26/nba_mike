from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_num(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _pct_rank_within_stat(df: pd.DataFrame, col: str, ascending: bool = True) -> pd.Series:
    """
    Percentile rank within each stat bucket.
    Output range: [0, 1]
    """
    out = pd.Series(index=df.index, dtype=float)

    for stat, idx in df.groupby("stat").groups.items():
        vals = _safe_num(df.loc[idx, col])

        if len(vals) == 1:
            out.loc[idx] = 1.0
            continue

        ranked = vals.rank(method="average", pct=True, ascending=ascending)
        out.loc[idx] = ranked.values

    return out.fillna(0.0)


def _build_confidence_tier(df: pd.DataFrame) -> pd.Series:
    """
    Practical confidence tier from projection-support signals.
    """
    minutes_proj = _safe_num(df["minutes_proj"])
    dispersion = _safe_num(df["dispersion"])
    baseline_mean = _safe_num(df["baseline_mean"])
    pred_mean = _safe_num(df["pred_mean"])

    baseline_ok = baseline_mean > 0
    minutes_high = minutes_proj >= 28
    minutes_mid = minutes_proj >= 22
    minutes_low = minutes_proj >= 14

    low_disp = dispersion <= dispersion.quantile(0.33) if len(dispersion) > 0 else pd.Series(True, index=df.index)
    mid_disp = dispersion <= dispersion.quantile(0.66) if len(dispersion) > 0 else pd.Series(True, index=df.index)

    high_conf = minutes_high & baseline_ok & low_disp & (pred_mean > 0)
    med_conf = minutes_mid & baseline_ok & mid_disp & (pred_mean > 0)
    low_conf = minutes_low & (pred_mean > 0)

    tier = pd.Series("low_conf", index=df.index, dtype="string")
    tier.loc[med_conf] = "medium_conf"
    tier.loc[high_conf] = "high_conf"

    # if truly weak row, force low_conf
    tier.loc[(minutes_proj < 14) | (pred_mean <= 0)] = "low_conf"

    return tier


def rank_projection_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes combined projection rows from pred_pts/reb/ast/fg3 and adds:

      - percentile ranks within stat
      - confidence tier
      - upgraded ticket scores

    Expected columns:
      game_date, player, team, opp, stat,
      pred_mean, baseline_mean, delta_mean, minutes_proj,
      dispersion, is_eligible
    """
    required = [
        "game_date",
        "player",
        "team",
        "opp",
        "stat",
        "pred_mean",
        "baseline_mean",
        "delta_mean",
        "minutes_proj",
        "dispersion",
        "is_eligible",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"rank_projection_pool missing required columns: {missing}")

    out = df.copy()

    out["pred_mean"] = _safe_num(out["pred_mean"])
    out["baseline_mean"] = _safe_num(out["baseline_mean"])
    out["delta_mean"] = _safe_num(out["delta_mean"])
    out["minutes_proj"] = _safe_num(out["minutes_proj"])
    out["dispersion"] = _safe_num(out["dispersion"])
    out["is_eligible"] = _safe_num(out["is_eligible"]).astype(int)

    # within-stat percentile ranks
    out["pred_pctile_within_stat"] = _pct_rank_within_stat(out, "pred_mean", ascending=True)
    out["delta_pctile_within_stat"] = _pct_rank_within_stat(out, "delta_mean", ascending=True)
    out["minutes_pctile_within_stat"] = _pct_rank_within_stat(out, "minutes_proj", ascending=True)

    # baseline quality / missingness flags
    out["baseline_available"] = (out["baseline_mean"] > 0).astype(int)
    out["zero_baseline_flag"] = (out["baseline_mean"] <= 0).astype(int)

    # lower dispersion = better for safe/balanced
    # convert to "good" score
    if len(out) > 1:
        out["dispersion_pctile_good"] = 1.0 - _pct_rank_within_stat(out, "dispersion", ascending=True)
    else:
        out["dispersion_pctile_good"] = 1.0

    out["confidence_tier"] = _build_confidence_tier(out)

    out["confidence_score"] = 0.0
    out.loc[out["confidence_tier"] == "low_conf", "confidence_score"] = 0.33
    out.loc[out["confidence_tier"] == "medium_conf", "confidence_score"] = 0.66
    out.loc[out["confidence_tier"] == "high_conf", "confidence_score"] = 1.00

    # -------------------------------------------------
    # Upgraded scores
    # -------------------------------------------------
    # SAFE: minutes + confidence + lower dispersion + baseline-supported projection
    out["score_safe"] = (
        0.35 * out["minutes_pctile_within_stat"]
        + 0.25 * out["pred_pctile_within_stat"]
        + 0.20 * out["dispersion_pctile_good"]
        + 0.20 * out["confidence_score"]
    )

    # penalize weak floor
    out.loc[out["minutes_proj"] < 22, "score_safe"] *= 0.35
    out.loc[out["baseline_available"] == 0, "score_safe"] *= 0.50

    # BALANCED: blend prediction, upside, minutes, confidence
    out["score_balanced"] = (
        0.25 * out["minutes_pctile_within_stat"]
        + 0.25 * out["pred_pctile_within_stat"]
        + 0.25 * out["delta_pctile_within_stat"]
        + 0.15 * out["confidence_score"]
        + 0.10 * out["dispersion_pctile_good"]
    )

    # LOTTO: upside first, then raw projection, light confidence check
    out["score_lotto"] = (
        0.45 * out["delta_pctile_within_stat"]
        + 0.30 * out["pred_pctile_within_stat"]
        + 0.15 * out["minutes_pctile_within_stat"]
        + 0.10 * out["confidence_score"]
    )

    # keep only meaningful candidates from getting absurd lotto scores
    out.loc[out["minutes_proj"] < 10, "score_lotto"] *= 0.40

    return out.sort_values(
        ["score_safe", "score_balanced", "score_lotto"],
        ascending=False,
    ).reset_index(drop=True)