from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Tunable thresholds
# ------------------------------------------------------------
SAFE_MIN_P_HIT = 0.68
BALANCED_MIN_P_HIT = 0.58
LOTTO_MIN_P_HIT = 0.52

SAFE_MIN_MINUTES = 20.0
BALANCED_MIN_MINUTES = 14.0
LOTTO_MIN_MINUTES = 10.0


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def _clip_prob(p: pd.Series) -> pd.Series:
    return p.clip(lower=1e-6, upper=1 - 1e-6)


def _safe_minutes(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _normalize_stat_side_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a direction-aware edge:
      - Over is good when pred_mean > line
      - Under is good when pred_mean < line

    edge_raw_signed > 0 means model agrees with the bet side.
    """
    out = df.copy()

    out["pred_mean"] = pd.to_numeric(out["pred_mean"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["p_hit"] = _clip_prob(pd.to_numeric(out["p_hit"], errors="coerce"))
    out["minutes_proj"] = _safe_minutes(out["minutes_proj"])

    over_mask = out["side"].astype(str).str.lower().eq("over")
    under_mask = out["side"].astype(str).str.lower().eq("under")

    out["edge_raw"] = np.nan
    out.loc[over_mask, "edge_raw"] = out.loc[over_mask, "pred_mean"] - out.loc[over_mask, "line"]
    out.loc[under_mask, "edge_raw"] = out.loc[under_mask, "line"] - out.loc[under_mask, "pred_mean"]

    # Normalize by scale of stat so large-mean stats do not dominate purely by units
    out["edge_norm"] = out["edge_raw"] / np.sqrt(out["pred_mean"].clip(lower=0.0) + 1.0)

    return out


def _minutes_confidence(minutes_proj: pd.Series) -> pd.Series:
    """
    Squashes projected minutes into [0,1].
    12 minutes should not get the same trust as 36 minutes.
    """
    x = _safe_minutes(minutes_proj)
    return (x / 36.0).clip(lower=0.0, upper=1.0)


def _stat_volatility_penalty(df: pd.DataFrame) -> pd.Series:
    """
    Small structural penalty by stat family.
    Lower is better.
    We do not want this too strong.
    """
    stat = df["stat"].astype(str).str.lower()

    penalty = pd.Series(0.0, index=df.index, dtype=float)
    penalty = np.where(stat.eq("fg3m"), 0.10, penalty)
    penalty = np.where(stat.eq("pts"), 0.06, penalty)
    penalty = np.where(stat.eq("ast"), 0.04, penalty)
    penalty = np.where(stat.eq("reb"), 0.03, penalty)

    return pd.Series(penalty, index=df.index, dtype=float)


def _extreme_line_penalty(df: pd.DataFrame) -> pd.Series:
    """
    Penalize legs that are mathematically 'safe' only because the line is far away
    from the model mean in a way that tends to create ugly fringe-leg pools.

    Example:
      pred_mean = 1.7 AST, under 6.5 looks hyper-safe, but that is not a useful core leg.
    """
    pred_mean = pd.to_numeric(df["pred_mean"], errors="coerce").fillna(0.0)
    line = pd.to_numeric(df["line"], errors="coerce").fillna(0.0)
    side = df["side"].astype(str).str.lower()

    # "distance" in raw stat units
    dist = (pred_mean - line).abs()

    # Penalize very far-away unders a little more than overs
    penalty = pd.Series(0.0, index=df.index, dtype=float)

    under_mask = side.eq("under")
    over_mask = side.eq("over")

    penalty.loc[under_mask] = np.maximum(dist.loc[under_mask] - 2.5, 0.0) * 0.06
    penalty.loc[over_mask] = np.maximum(dist.loc[over_mask] - 2.0, 0.0) * 0.03

    return penalty


def _eligibility_flag(df: pd.DataFrame) -> pd.Series:
    if "is_eligible" not in df.columns:
        return pd.Series(1, index=df.index, dtype=int)
    return pd.to_numeric(df["is_eligible"], errors="coerce").fillna(0).astype(int)


# ------------------------------------------------------------
# Ticket-specific score builders
# ------------------------------------------------------------
def _score_safe(df: pd.DataFrame) -> pd.Series:
    """
    Safe ticket:
      - high probability is king
      - stable minutes matter a lot
      - avoid overly weird/extreme lines
      - slight penalty for volatile stats
    """
    p_hit = _clip_prob(pd.to_numeric(df["p_hit"], errors="coerce"))
    prob_edge = p_hit - 0.5
    edge_norm = pd.to_numeric(df["edge_norm"], errors="coerce").fillna(0.0)
    min_conf = _minutes_confidence(df["minutes_proj"])
    stat_pen = _stat_volatility_penalty(df)
    line_pen = _extreme_line_penalty(df)

    score = (
        1.35 * prob_edge
        + 0.35 * edge_norm
        + 0.35 * min_conf
        - 0.35 * stat_pen
        - 0.45 * line_pen
    )
    return score


def _score_balanced(df: pd.DataFrame) -> pd.Series:
    """
    Balanced ticket:
      - blend probability and edge
      - still reward stable minutes
      - smaller penalty for volatility
    """
    p_hit = _clip_prob(pd.to_numeric(df["p_hit"], errors="coerce"))
    prob_edge = p_hit - 0.5
    edge_norm = pd.to_numeric(df["edge_norm"], errors="coerce").fillna(0.0)
    min_conf = _minutes_confidence(df["minutes_proj"])
    stat_pen = _stat_volatility_penalty(df)
    line_pen = _extreme_line_penalty(df)

    score = (
        0.95 * prob_edge
        + 0.80 * edge_norm
        + 0.20 * min_conf
        - 0.20 * stat_pen
        - 0.20 * line_pen
    )
    return score


def _score_lotto(df: pd.DataFrame) -> pd.Series:
    """
    Lotto-but-still-plausible:
      - edge matters most
      - probability still matters, but less
      - minutes confidence still matters
      - allow more aggressive lines, but not garbage
    """
    p_hit = _clip_prob(pd.to_numeric(df["p_hit"], errors="coerce"))
    prob_edge = p_hit - 0.5
    edge_norm = pd.to_numeric(df["edge_norm"], errors="coerce").fillna(0.0)
    min_conf = _minutes_confidence(df["minutes_proj"])
    stat_pen = _stat_volatility_penalty(df)
    line_pen = _extreme_line_penalty(df)

    score = (
        0.55 * prob_edge
        + 1.10 * edge_norm
        + 0.15 * min_conf
        - 0.15 * stat_pen
        - 0.10 * line_pen
    )
    return score


# ------------------------------------------------------------
# Risk filters
# ------------------------------------------------------------
def _passes_safe_filter(df: pd.DataFrame) -> pd.Series:
    return (
        (_eligibility_flag(df) == 1)
        & (pd.to_numeric(df["p_hit"], errors="coerce") >= SAFE_MIN_P_HIT)
        & (_safe_minutes(df["minutes_proj"]) >= SAFE_MIN_MINUTES)
        & (pd.to_numeric(df["edge_raw"], errors="coerce") > 0)
    )


def _passes_balanced_filter(df: pd.DataFrame) -> pd.Series:
    return (
        (_eligibility_flag(df) == 1)
        & (pd.to_numeric(df["p_hit"], errors="coerce") >= BALANCED_MIN_P_HIT)
        & (_safe_minutes(df["minutes_proj"]) >= BALANCED_MIN_MINUTES)
        & (pd.to_numeric(df["edge_raw"], errors="coerce") > 0)
    )


def _passes_lotto_filter(df: pd.DataFrame) -> pd.Series:
    return (
        (_eligibility_flag(df) == 1)
        & (pd.to_numeric(df["p_hit"], errors="coerce") >= LOTTO_MIN_P_HIT)
        & (_safe_minutes(df["minutes_proj"]) >= LOTTO_MIN_MINUTES)
        & (pd.to_numeric(df["edge_raw"], errors="coerce") > 0)
    )


# ------------------------------------------------------------
# Main public function
# ------------------------------------------------------------
def score_legs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:
      line-level predictions following shared prediction schema

    Output:
      same rows + scoring fields for:
        - score_safe
        - score_balanced
        - score_lotto
        - can_safe
        - can_balanced
        - can_lotto
    """
    required = [
        "game_date",
        "player",
        "team",
        "opp",
        "stat",
        "line",
        "side",
        "pred_mean",
        "p_hit",
        "minutes_proj",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"score_legs missing required columns: {missing}")

    out = df.copy()

    out = _normalize_stat_side_edge(out)

    out["minutes_conf"] = _minutes_confidence(out["minutes_proj"])
    out["prob_edge"] = _clip_prob(pd.to_numeric(out["p_hit"], errors="coerce")) - 0.5
    out["stat_vol_penalty"] = _stat_volatility_penalty(out)
    out["extreme_line_penalty"] = _extreme_line_penalty(out)

    out["score_safe"] = _score_safe(out)
    out["score_balanced"] = _score_balanced(out)
    out["score_lotto"] = _score_lotto(out)

    out["can_safe"] = _passes_safe_filter(out).astype(int)
    out["can_balanced"] = _passes_balanced_filter(out).astype(int)
    out["can_lotto"] = _passes_lotto_filter(out).astype(int)

    # optional generic view
    out["score"] = out["score_balanced"]

    return out


# ------------------------------------------------------------
# Convenience selectors
# ------------------------------------------------------------
def top_safe_legs(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    scored = score_legs(df) if "score_safe" not in df.columns else df.copy()
    return (
        scored[scored["can_safe"] == 1]
        .sort_values(["score_safe", "p_hit"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_balanced_legs(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    scored = score_legs(df) if "score_balanced" not in df.columns else df.copy()
    return (
        scored[scored["can_balanced"] == 1]
        .sort_values(["score_balanced", "p_hit"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_lotto_legs(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    scored = score_legs(df) if "score_lotto" not in df.columns else df.copy()
    return (
        scored[scored["can_lotto"] == 1]
        .sort_values(["score_lotto", "p_hit"], ascending=False)
        .head(n)
        .reset_index(drop=True)
    )