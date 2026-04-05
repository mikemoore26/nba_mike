from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def score_legs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leg-level scoring only.

    Important:
    - DO NOT truncate the pool here.
    - DO NOT force final side here if side already exists.
    - Keep this layer focused on edge + p_hit + tier score construction.

    Final pool sizing and ticket optimization happen later.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    required = ["pred_mean", "line"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"score_legs missing required columns: {missing}")

    out["pred_mean"] = pd.to_numeric(out["pred_mean"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")

    if "side" not in out.columns:
        raw_side = np.where(out["pred_mean"] > out["line"], "over", "under")
        out["side"] = raw_side
    else:
        out["side"] = out["side"].astype(str).str.strip().str.lower()

    # -----------------------------------
    # Edge (directional + absolute)
    # -----------------------------------
    raw_edge = out["pred_mean"] - out["line"]
    out["edge_raw"] = np.where(out["side"].eq("over"), raw_edge, -raw_edge)
    out["edge_abs"] = out["edge_raw"].abs()

    # -----------------------------------
    # p_hit fallback if not present
    # -----------------------------------
    if "p_hit" not in out.columns:
        out["p_hit"] = 0.5 + np.tanh(out["edge_raw"] / 2.0) * 0.25
    else:
        out["p_hit"] = pd.to_numeric(out["p_hit"], errors="coerce")
        fallback = 0.5 + np.tanh(out["edge_raw"] / 2.0) * 0.25
        out["p_hit"] = out["p_hit"].fillna(fallback)

    out["p_hit"] = out["p_hit"].clip(lower=0.0, upper=1.0)

    # -----------------------------------
    # Slate-stable edge scaling
    # -----------------------------------
    edge_scale = float(out["edge_abs"].quantile(0.95)) if len(out) else 1.0
    if not np.isfinite(edge_scale) or edge_scale <= 0:
        edge_scale = 1.0

    out["edge_scaled"] = (out["edge_abs"] / edge_scale).clip(0.0, 1.5)

    # -----------------------------------
    # Tier scores
    # Still simple, but better than hard pure p_hit/edge mix.
    # More portfolio-aware shaping happens later in leg_utility.py
    # -----------------------------------
    out["score_safe"] = (
        0.72 * out["p_hit"]
        + 0.28 * out["edge_scaled"]
    )

    out["score_balanced"] = (
        0.58 * out["p_hit"]
        + 0.42 * out["edge_scaled"]
    )

    out["score_lotto"] = (
        0.42 * out["p_hit"]
        + 0.58 * out["edge_scaled"]
    )

    sort_cols = [c for c in ["score_balanced", "p_hit", "edge_abs"] if c in out.columns]
    out = out.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    return out


def build_ranked_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = score_legs(df)
    return out.sort_values(
        ["score_balanced", "p_hit", "edge_abs"],
        ascending=False,
    ).reset_index(drop=True)