from __future__ import annotations

import pandas as pd


STAT_VARIANCE_PENALTY = {
    "fg3": 0.25,
    "fg3m": 0.25,
    "pts": 0.18,
    "ast": 0.12,
    "reb": 0.10,
}


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def add_leg_utility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts raw scored legs into more realistic betting utilities.

    Why this matters:
    - raw score_safe / score_balanced / score_lotto are still too shallow
    - this adds variance penalty, fragility penalty, and role/stability support
    - this is still leg-level only; ticket-level dependency happens elsewhere
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # safe fills
    for col, default in {
        "p_hit": 0.5,
        "edge_raw": 0.0,
        "role_score": 0.5,
        "stability_score": 0.5,
        "usage_score": 0.5,
        "fragility_score": 0.5,
        "minutes_proj": 0.0,
    }.items():
        out[col] = _safe_num(out, col, default=default)

    out["stat"] = out["stat"].astype(str).str.strip().str.lower()

    # use positive edge for bet strength; bad directional edges should not be rewarded
    edge_pos = out["edge_raw"].clip(lower=0.0)

    edge_scale = float(edge_pos.quantile(0.95)) if len(out) else 1.0
    if edge_scale <= 0:
        edge_scale = 1.0

    out["edge_pos_scaled"] = (edge_pos / edge_scale).clip(0.0, 1.5)

    out["stat_var_penalty"] = out["stat"].map(STAT_VARIANCE_PENALTY).fillna(0.15)

    out["minutes_conf"] = (
        (out["minutes_proj"] / 32.0)
        .clip(lower=0.0, upper=1.0)
    )

    # General utility
    out["leg_utility"] = (
        0.35 * out["p_hit"]
        + 0.25 * out["edge_pos_scaled"]
        + 0.15 * out["stability_score"]
        + 0.10 * out["role_score"]
        + 0.08 * out["usage_score"]
        + 0.07 * out["minutes_conf"]
        - 0.20 * out["stat_var_penalty"]
        - 0.12 * out["fragility_score"]
    )

    # tier-specific utility
    out["leg_utility_safe"] = (
        0.42 * out["p_hit"]
        + 0.18 * out["edge_pos_scaled"]
        + 0.18 * out["stability_score"]
        + 0.10 * out["role_score"]
        + 0.10 * out["minutes_conf"]
        - 0.18 * out["stat_var_penalty"]
        - 0.15 * out["fragility_score"]
    )

    out["leg_utility_balanced"] = (
        0.35 * out["p_hit"]
        + 0.25 * out["edge_pos_scaled"]
        + 0.15 * out["stability_score"]
        + 0.10 * out["role_score"]
        + 0.08 * out["usage_score"]
        + 0.07 * out["minutes_conf"]
        - 0.20 * out["stat_var_penalty"]
        - 0.12 * out["fragility_score"]
    )

    out["leg_utility_lotto"] = (
        0.22 * out["p_hit"]
        + 0.38 * out["edge_pos_scaled"]
        + 0.12 * out["usage_score"]
        + 0.10 * out["role_score"]
        + 0.08 * out["projection_rank_score"] if "projection_rank_score" in out.columns else 0.0
    )

    # lotto penalty block added separately to avoid precedence issues
    if "leg_utility_lotto" not in out.columns:
        out["leg_utility_lotto"] = 0.0

    out["leg_utility_lotto"] = (
        _safe_num(out, "leg_utility_lotto", 0.0)
        - 0.16 * out["stat_var_penalty"]
        - 0.08 * out["fragility_score"]
    )

    return out