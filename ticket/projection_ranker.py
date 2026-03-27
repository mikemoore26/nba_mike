from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_num(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def _stat_key(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def _pct_rank_within_stat(
    df: pd.DataFrame,
    col: str,
    *,
    ascending: bool = True,
) -> pd.Series:
    """
    Percentile rank within each stat bucket.
    Output range: [0, 1]
    """
    out = pd.Series(index=df.index, dtype=float)

    for _, idx in df.groupby("stat").groups.items():
        vals = _safe_num(df.loc[idx, col])

        if len(vals) == 1:
            out.loc[idx] = 1.0
            continue

        ranked = vals.rank(method="average", pct=True, ascending=ascending)
        out.loc[idx] = ranked.values

    return out.fillna(0.0)


def _build_confidence_tier(df: pd.DataFrame) -> pd.Series:
    """
    Confidence tier from projection support signals.
    """
    minutes_proj = _safe_num(df["minutes_proj"])
    dispersion = _safe_num(df["dispersion"])
    baseline_mean = _safe_num(df["baseline_mean"])
    pred_mean = _safe_num(df["pred_mean"])

    baseline_ok = baseline_mean > 0
    minutes_high = minutes_proj >= 28
    minutes_mid = minutes_proj >= 22
    minutes_low = minutes_proj >= 14

    if len(dispersion) > 0:
        low_disp_cut = float(dispersion.quantile(0.33))
        mid_disp_cut = float(dispersion.quantile(0.66))
    else:
        low_disp_cut = np.inf
        mid_disp_cut = np.inf

    low_disp = dispersion <= low_disp_cut
    mid_disp = dispersion <= mid_disp_cut

    high_conf = minutes_high & baseline_ok & low_disp & (pred_mean > 0)
    med_conf = minutes_mid & baseline_ok & mid_disp & (pred_mean > 0)
    low_conf = minutes_low & (pred_mean > 0)

    tier = pd.Series("low_conf", index=df.index, dtype="string")
    tier.loc[low_conf] = "low_conf"
    tier.loc[med_conf] = "medium_conf"
    tier.loc[high_conf] = "high_conf"

    tier.loc[(minutes_proj < 14) | (pred_mean <= 0)] = "low_conf"
    return tier


def _confidence_score(tier: pd.Series) -> pd.Series:
    out = pd.Series(0.33, index=tier.index, dtype=float)
    out.loc[tier == "medium_conf"] = 0.66
    out.loc[tier == "high_conf"] = 1.00
    return out


def _stat_family_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backtest-informed stat-level trust.

    Interpretation from your current results:
    - AST: best broad point forecast
    - REB: strongest top-bucket / top-pick selector
    - FG3: decent but flatter
    - PTS: acceptable but less edge-rich
    """
    stat = _stat_key(df["stat"])

    broad = pd.Series(1.00, index=df.index, dtype=float)
    top_pick = pd.Series(1.00, index=df.index, dtype=float)
    variance_penalty = pd.Series(0.0, index=df.index, dtype=float)

    # AST: broad strength, slight top-tail caution
    broad.loc[stat == "ast"] = 1.08
    top_pick.loc[stat == "ast"] = 0.98
    variance_penalty.loc[stat == "ast"] = 0.03

    # REB: biggest top-pick edge
    broad.loc[stat == "reb"] = 0.98
    top_pick.loc[stat == "reb"] = 1.14
    variance_penalty.loc[stat == "reb"] = 0.02

    # FG3: okay, but flatter edge
    broad.loc[(stat == "fg3") | (stat == "fg3m")] = 0.96
    top_pick.loc[(stat == "fg3") | (stat == "fg3m")] = 0.97
    variance_penalty.loc[(stat == "fg3") | (stat == "fg3m")] = 0.06

    # PTS: useful but less selective
    broad.loc[stat == "pts"] = 0.97
    top_pick.loc[stat == "pts"] = 0.96
    variance_penalty.loc[stat == "pts"] = 0.04

    return pd.DataFrame(
        {
            "stat_broad_weight": broad,
            "stat_top_pick_weight": top_pick,
            "stat_variance_penalty": variance_penalty,
        },
        index=df.index,
    )


def rank_projection_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank combined projection rows before line-level leg scoring.

    This is a pre-line candidate prioritizer. It should answer:
      "Which projected player-stat rows are strongest inputs to ticket construction?"

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

    # Within-stat percentile ranks
    out["pred_pctile_within_stat"] = _pct_rank_within_stat(out, "pred_mean", ascending=True)
    out["delta_pctile_within_stat"] = _pct_rank_within_stat(out, "delta_mean", ascending=True)
    out["minutes_pctile_within_stat"] = _pct_rank_within_stat(out, "minutes_proj", ascending=True)

    # Lower dispersion = better
    out["dispersion_pctile_good"] = _pct_rank_within_stat(out, "dispersion", ascending=False)

    # Baseline support
    out["baseline_available"] = (out["baseline_mean"] > 0).astype(int)
    out["zero_baseline_flag"] = (out["baseline_mean"] <= 0).astype(int)

    # Relative lift over baseline helps compare "signal strength"
    out["rel_delta_vs_baseline"] = np.where(
        out["baseline_mean"] > 0,
        out["delta_mean"] / out["baseline_mean"].replace(0, np.nan),
        0.0,
    )
    out["rel_delta_vs_baseline"] = pd.to_numeric(out["rel_delta_vs_baseline"], errors="coerce").fillna(0.0)
    out["rel_delta_pctile_within_stat"] = _pct_rank_within_stat(
        out,
        "rel_delta_vs_baseline",
        ascending=True,
    )

    # Confidence
    out["confidence_tier"] = _build_confidence_tier(out)
    out["confidence_score"] = _confidence_score(out["confidence_tier"])

    # Stat-aware weights
    stat_weights = _stat_family_weights(out)
    out = pd.concat([out, stat_weights], axis=1)

    # Top-pick bucket proxy within stat
    out["top_pick_bucket"] = "core"
    out.loc[out["pred_pctile_within_stat"] >= 0.95, "top_pick_bucket"] = "elite"
    out.loc[
        (out["pred_pctile_within_stat"] >= 0.90) & (out["pred_pctile_within_stat"] < 0.95),
        "top_pick_bucket",
    ] = "strong"

    # REB gets extra value when it is truly top-end
    stat = _stat_key(out["stat"])
    out["top_bucket_bonus"] = 0.0
    out.loc[(stat == "reb") & (out["pred_pctile_within_stat"] >= 0.95), "top_bucket_bonus"] = 0.16
    out.loc[
        (stat == "reb")
        & (out["pred_pctile_within_stat"] >= 0.90)
        & (out["pred_pctile_within_stat"] < 0.95),
        "top_bucket_bonus",
    ] = 0.08

    # AST slight tail caution at extreme top
    out["ast_tail_penalty"] = 0.0
    out.loc[(stat == "ast") & (out["pred_pctile_within_stat"] >= 0.95), "ast_tail_penalty"] = 0.05

    # -------------------------------------------------
    # Scoring
    # -------------------------------------------------
    # SAFE:
    # - broad forecast quality
    # - minutes and confidence matter a lot
    # - lower dispersion matters
    # - AST broad trust is useful here
    out["score_safe"] = (
        0.34 * out["minutes_pctile_within_stat"]
        + 0.28 * out["pred_pctile_within_stat"]
        + 0.18 * out["dispersion_pctile_good"]
        + 0.14 * out["confidence_score"]
        + 0.06 * out["rel_delta_pctile_within_stat"]
    )

    out["score_safe"] = (
        out["score_safe"] * out["stat_broad_weight"]
        - out["stat_variance_penalty"]
        - out["ast_tail_penalty"]
    )

    out.loc[out["minutes_proj"] < 22, "score_safe"] *= 0.35
    out.loc[out["baseline_available"] == 0, "score_safe"] *= 0.50
    out.loc[out["is_eligible"] != 1, "score_safe"] *= 0.10

    # BALANCED:
    # - blend raw projection, delta, and confidence
    # - REB top-end gets rewarded
    out["score_balanced"] = (
        0.23 * out["minutes_pctile_within_stat"]
        + 0.24 * out["pred_pctile_within_stat"]
        + 0.22 * out["delta_pctile_within_stat"]
        + 0.16 * out["confidence_score"]
        + 0.09 * out["dispersion_pctile_good"]
        + 0.06 * out["rel_delta_pctile_within_stat"]
    )

    out["score_balanced"] = (
        out["score_balanced"] * ((out["stat_broad_weight"] + out["stat_top_pick_weight"]) / 2.0)
        + out["top_bucket_bonus"]
        - 0.5 * out["ast_tail_penalty"]
        - 0.5 * out["stat_variance_penalty"]
    )

    out.loc[out["minutes_proj"] < 18, "score_balanced"] *= 0.55
    out.loc[out["baseline_available"] == 0, "score_balanced"] *= 0.75
    out.loc[out["is_eligible"] != 1, "score_balanced"] *= 0.10

    # LOTTO:
    # - upside first
    # - top-pick signal matters more
    # - REB gets the most help here
    out["score_lotto"] = (
        0.34 * out["delta_pctile_within_stat"]
        + 0.24 * out["pred_pctile_within_stat"]
        + 0.14 * out["minutes_pctile_within_stat"]
        + 0.12 * out["confidence_score"]
        + 0.16 * out["rel_delta_pctile_within_stat"]
    )

    out["score_lotto"] = (
        out["score_lotto"] * out["stat_top_pick_weight"]
        + out["top_bucket_bonus"]
        - 0.35 * out["stat_variance_penalty"]
    )

    out.loc[out["minutes_proj"] < 10, "score_lotto"] *= 0.40
    out.loc[out["is_eligible"] != 1, "score_lotto"] *= 0.10

    # Useful global ordering proxy
    out["rank_score"] = (
        0.40 * out["score_safe"]
        + 0.35 * out["score_balanced"]
        + 0.25 * out["score_lotto"]
    )

    return out.sort_values(
        ["rank_score", "score_safe", "score_balanced", "score_lotto"],
        ascending=False,
    ).reset_index(drop=True)