from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom


def _safe_num(s: pd.Series, default: float | None = None) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    if default is not None:
        out = out.fillna(default)
    return out


def _normalize_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = ["game_date", "player", "team", "stat", "pred_mean"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"expand_to_pseudo_legs missing required columns: {missing}")

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["player"] = out["player"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()

    if "opp" not in out.columns:
        out["opp"] = ""

    out["opp"] = out["opp"].astype(str).str.strip().str.upper()
    out["stat"] = out["stat"].astype(str).str.strip().str.lower()
    out["pred_mean"] = _safe_num(out["pred_mean"])

    if "baseline_mean" not in out.columns:
        out["baseline_mean"] = np.nan
    out["baseline_mean"] = _safe_num(out["baseline_mean"])

    if "minutes_proj" not in out.columns:
        out["minutes_proj"] = np.nan
    out["minutes_proj"] = _safe_num(out["minutes_proj"])

    if "dist_name" not in out.columns:
        out["dist_name"] = "poisson"
    out["dist_name"] = out["dist_name"].astype(str).str.strip().str.lower()

    if "dispersion" not in out.columns:
        out["dispersion"] = np.nan
    out["dispersion"] = _safe_num(out["dispersion"])

    if "is_eligible" not in out.columns:
        out["is_eligible"] = True

    return out


def _default_half_line(mu: float) -> float:
    """
    Sportsbook-style half-point anchor around the model mean.
    """
    if pd.isna(mu):
        return np.nan
    return float(math.floor(mu) + 0.5)


def _candidate_lines(mu: float, stat: str, line_offsets: Iterable[float]) -> list[float]:
    """
    Build half-point candidate lines around the model mean.
    Example:
      mu=7.8 -> anchor=7.5 -> offsets (-1,0,1) => [6.5, 7.5, 8.5]
    """
    anchor = _default_half_line(mu)
    if pd.isna(anchor):
        return []

    lines = []
    for off in line_offsets:
        line = anchor + float(off)
        if line < 0.5:
            line = 0.5
        lines.append(float(line))

    # De-dup and sort
    return sorted(set(lines))


def _nbinom_params(mu: float, alpha: float) -> tuple[float, float] | None:
    """
    NB2 parameterization:
      var = mu + alpha * mu^2
      n = 1 / alpha
      p = n / (n + mu)
    """
    if pd.isna(mu) or mu < 0:
        return None
    if pd.isna(alpha) or alpha <= 0:
        return None

    n = 1.0 / alpha
    p = n / (n + mu)
    if n <= 0 or p <= 0 or p >= 1:
        return None
    return n, p


def _prob_over(mu: float, line: float, dist_name: str, dispersion: float | None) -> float:
    """
    For half-point line L = k + 0.5:
      P(over L) = P(X >= k+1)
    """
    if pd.isna(mu) or pd.isna(line):
        return np.nan

    k = int(math.floor(line) + 1)

    if dist_name == "nbinom":
        params = _nbinom_params(mu, dispersion if dispersion is not None else np.nan)
        if params is not None:
            n, p = params
            return float(1.0 - nbinom.cdf(k - 1, n, p))

    return float(1.0 - poisson.cdf(k - 1, mu))


def _prob_under(mu: float, line: float, dist_name: str, dispersion: float | None) -> float:
    """
    For half-point line L = k + 0.5:
      P(under L) = P(X <= k)
    """
    if pd.isna(mu) or pd.isna(line):
        return np.nan

    k = int(math.floor(line))

    if dist_name == "nbinom":
        params = _nbinom_params(mu, dispersion if dispersion is not None else np.nan)
        if params is not None:
            n, p = params
            return float(nbinom.cdf(k, n, p))

    return float(poisson.cdf(k, mu))


def expand_to_pseudo_legs(
    pred_df: pd.DataFrame,
    *,
    line_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0),
    min_prob: float = 0.50,
    keep_both_sides: bool = True,
) -> pd.DataFrame:
    """
    Expand player mean predictions into pseudo sportsbook-style over/under legs.

    Output includes the columns score_legs expects:
      - line
      - side
      - p_hit

    Notes:
    - Uses half-point lines around the model mean.
    - Uses poisson by default, nbinom when dist_name='nbinom' and dispersion is present.
    - Keeps only candidate legs with p_hit >= min_prob.
    """
    df = _normalize_pred_df(pred_df)

    rows: list[dict] = []

    for _, r in df.iterrows():
        mu = r["pred_mean"]
        stat = r["stat"]
        dist_name = r["dist_name"]
        dispersion = r["dispersion"]

        if pd.isna(mu) or mu < 0:
            continue

        candidate_lines = _candidate_lines(mu, stat, line_offsets)
        if not candidate_lines:
            continue

        for line in candidate_lines:
            p_over = _prob_over(mu, line, dist_name, dispersion)
            p_under = _prob_under(mu, line, dist_name, dispersion)

            common = {
                "game_date": r["game_date"],
                "player": r["player"],
                "team": r["team"],
                "opp": r["opp"],
                "stat": stat,
                "pred_mean": mu,
                "baseline_mean": r.get("baseline_mean", np.nan),
                "minutes_proj": r.get("minutes_proj", np.nan),
                "dist_name": dist_name,
                "dispersion": dispersion,
                "is_eligible": r.get("is_eligible", True),
                "eligibility_reason": r.get("eligibility_reason", ""),
                "model_name": r.get("model_name", ""),
                "model_version": r.get("model_version", ""),
                "line": float(line),
            }

            if keep_both_sides:
                over_row = common | {"side": "over", "p_hit": p_over}
                under_row = common | {"side": "under", "p_hit": p_under}

                if pd.notna(over_row["p_hit"]) and over_row["p_hit"] >= min_prob:
                    rows.append(over_row)
                if pd.notna(under_row["p_hit"]) and under_row["p_hit"] >= min_prob:
                    rows.append(under_row)
            else:
                # Keep only the stronger side at this line
                if p_over >= p_under:
                    row = common | {"side": "over", "p_hit": p_over}
                else:
                    row = common | {"side": "under", "p_hit": p_under}

                if pd.notna(row["p_hit"]) and row["p_hit"] >= min_prob:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # Stable sort for reproducibility
    sort_cols = [c for c in ["game_date", "player", "stat", "line", "side"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    return out