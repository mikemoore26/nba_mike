from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _clip01(x: pd.Series) -> pd.Series:
    return x.clip(lower=0.0, upper=1.0)


def _pct_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index, dtype="float64")
    return s.rank(pct=True, method="average")


def add_role_scores(board: pd.DataFrame) -> pd.DataFrame:
    """
    Adds role/stability/usage quality fields to a player-level projection board.

    Expected input:
      projection_board.csv or projection_board_ranked.csv

    Uses only board-level pregame fields.
    """
    if board is None or board.empty:
        return pd.DataFrame()

    out = board.copy()

    minutes_proj = _safe_num(out, "minutes_proj", default=np.nan)
    minutes_conf = _safe_num(out, "minutes_conf", default=np.nan).fillna(1.0)
    confidence_score = _safe_num(out, "confidence_score", default=np.nan).fillna(1.0)

    pred_pts = _safe_num(out, "pred_pts", default=0.0).fillna(0.0)
    pred_reb = _safe_num(out, "pred_reb", default=0.0).fillna(0.0)
    pred_ast = _safe_num(out, "pred_ast", default=0.0).fillna(0.0)
    pred_fg3 = _safe_num(out, "pred_fg3", default=0.0).fillna(0.0)

    delta_pts = _safe_num(out, "delta_pts", default=0.0).fillna(0.0)
    delta_reb = _safe_num(out, "delta_reb", default=0.0).fillna(0.0)
    delta_ast = _safe_num(out, "delta_ast", default=0.0).fillna(0.0)
    delta_fg3 = _safe_num(out, "delta_fg3", default=0.0).fillna(0.0)

    # -----------------------------
    # Minutes-based role strength
    # -----------------------------
    minutes_strength = _clip01((minutes_proj - 16.0) / 20.0)
    minutes_elite = _clip01((minutes_proj - 28.0) / 10.0)

    # -----------------------------
    # Confidence / stability proxy
    # -----------------------------
    stability_score = (
        0.55 * minutes_conf.fillna(1.0)
        + 0.45 * confidence_score.fillna(1.0)
    )
    stability_score = _clip01(stability_score)

    # -----------------------------
    # Usage / self-creation proxy
    # points + assists are most relevant here
    # -----------------------------
    usage_raw = (
        0.55 * _pct_rank(pred_pts).fillna(0.0)
        + 0.35 * _pct_rank(pred_ast).fillna(0.0)
        + 0.10 * _pct_rank(pred_fg3).fillna(0.0)
    )
    usage_score = _clip01(usage_raw)

    # -----------------------------
    # Rebound role proxy
    # helps identify bigs / rebound specialists
    # -----------------------------
    rebound_role_score = _clip01(_pct_rank(pred_reb).fillna(0.0))

    # -----------------------------
    # Opportunity score
    # positive delta vs baseline matters
    # -----------------------------
    opp_raw = (
        0.35 * _pct_rank(delta_pts).fillna(0.0)
        + 0.25 * _pct_rank(delta_reb).fillna(0.0)
        + 0.25 * _pct_rank(delta_ast).fillna(0.0)
        + 0.15 * _pct_rank(delta_fg3).fillna(0.0)
    )
    opportunity_score = _clip01(opp_raw)

    # -----------------------------
    # Primary role score
    # “Can I trust this player’s path?”
    # -----------------------------
    role_score = (
        0.35 * minutes_strength.fillna(0.0)
        + 0.20 * minutes_elite.fillna(0.0)
        + 0.20 * stability_score.fillna(0.0)
        + 0.15 * usage_score.fillna(0.0)
        + 0.10 * opportunity_score.fillna(0.0)
    )
    role_score = _clip01(role_score)

    # -----------------------------
    # Fragility score
    # high if low-minute / low-stability / low-usage
    # -----------------------------
    fragility_score = _clip01(
        1.0
        - (
            0.45 * minutes_strength.fillna(0.0)
            + 0.30 * stability_score.fillna(0.0)
            + 0.25 * usage_score.fillna(0.0)
        )
    )

    # -----------------------------
    # Tier labels for downstream filtering
    # -----------------------------
    role_tier = np.select(
        [
            role_score >= 0.78,
            role_score >= 0.64,
            role_score >= 0.50,
        ],
        [
            "core",
            "solid",
            "fragile",
        ],
        default="thin",
    )

    out["minutes_strength"] = minutes_strength
    out["minutes_elite"] = minutes_elite
    out["stability_score"] = stability_score
    out["usage_score"] = usage_score
    out["rebound_role_score"] = rebound_role_score
    out["opportunity_score"] = opportunity_score
    out["role_score"] = role_score
    out["fragility_score"] = fragility_score
    out["role_tier"] = role_tier

    return out