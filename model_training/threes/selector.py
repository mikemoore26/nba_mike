# model_training/threes/selector.py
# Ticket selection rules (deterministic, testable, no leakage)
# Produces:
#   - 2+ card (10 legs default)
#   - jackpot card (+2 over baseline, 2-3 legs default)

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def select_2plus_ticket(
    df: pd.DataFrame,
    *,
    n_legs: int = 10,
    min_pred_fg3a: float = 5.8,
    min_p_ge_2: float = 0.62,
    min_pred_rate: float = 0.29,   # avoid ultra-low rate outliers
    min_p_ge_3: float | None = 0.30,
    max_per_team: int = 3,
    require_unique_matchups: bool = False,
) -> pd.DataFrame:
    """
    Selects a high-floor 2+ threes ticket.
    Core idea: volume (pred_fg3a) + floor probability (p_ge_2).
    """

    out = df.copy()

    for c in ["pred_fg3a", "pred_rate", "p_ge_2", "p_ge_3", "pred_fg3"]:
        if c in out.columns:
            out[c] = _safe_num(out[c])

    needed = {"player", "team", "opp", "is_home", "pred_fg3a", "pred_rate", "p_ge_2"}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns for 2+ selector: {sorted(missing)}")

    # Basic filters
    mask = (
        (out["pred_fg3a"] >= min_pred_fg3a)
        & (out["p_ge_2"] >= min_p_ge_2)
        & (out["pred_rate"] >= min_pred_rate)
    )

    if min_p_ge_3 is not None:
        if "p_ge_3" not in out.columns:
            raise ValueError("min_p_ge_3 set but p_ge_3 column missing")
        mask &= (out["p_ge_3"] >= min_p_ge_3)

    cand = out.loc[mask].copy()

    if cand.empty:
        return cand

    # Ranking score: prioritize floor, then volume, then mean
    cand["score_2p"] = (
        cand["p_ge_2"]
        + 0.15 * cand["pred_fg3a"].rank(pct=True)
        + 0.10 * cand.get("pred_fg3", pd.Series(index=cand.index, dtype=float)).rank(pct=True)
    )

    cand = cand.sort_values(["score_2p", "p_ge_2", "pred_fg3a"], ascending=False)

    # Optional: avoid too many from same team (correlation control)
    if max_per_team is not None and max_per_team > 0:
        kept = []
        team_counts: dict[str, int] = {}
        for _, r in cand.iterrows():
            t = str(r["team"])
            team_counts.setdefault(t, 0)
            if team_counts[t] >= max_per_team:
                continue
            kept.append(r)
            team_counts[t] += 1
            if len(kept) >= n_legs:
                break
        cand = pd.DataFrame(kept)

    # Optional: require unique matchups (home/away pairs)
    if require_unique_matchups and not cand.empty:
        cand["matchup_key"] = np.where(
            cand["is_home"].astype(int) == 1,
            cand["opp"].astype(str) + "@" + cand["team"].astype(str),
            cand["team"].astype(str) + "@" + cand["opp"].astype(str),
        )
        cand = cand.drop_duplicates("matchup_key", keep="first")

    return cand.head(n_legs).reset_index(drop=True)


def select_jackpot_ticket(
    df: pd.DataFrame,
    *,
    n_legs: int = 3,
    min_pred_fg3a: float = 5.5,
    min_p_over_baseline: float = 0.18,
    min_delta_fg3: float = 0.75,
    max_per_team: int = 1,
    require_baseline: bool = True,
    over_baseline_delta: int = 2,
) -> pd.DataFrame:
    """
    Selects a small "jackpot" ticket: most likely to outshoot baseline by +delta.

    Requires columns:
      - baseline_fg3 (non-NaN if gated)
      - delta_fg3
      - p_over_baseline_{delta}
    """

    out = df.copy()

    pcol = f"p_over_baseline_{int(over_baseline_delta)}"
    needed = {"player", "team", "opp", "is_home", "pred_fg3a", "pred_rate", "pred_fg3", "delta_fg3", pcol}
    missing = needed - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns for jackpot selector: {sorted(missing)}")

    for c in ["pred_fg3a", "pred_rate", "pred_fg3", "delta_fg3", pcol, "baseline_fg3"]:
        if c in out.columns:
            out[c] = _safe_num(out[c])

    mask = (
        (out["pred_fg3a"] >= min_pred_fg3a)
        & (out[pcol] >= min_p_over_baseline)
        & (out["delta_fg3"] >= min_delta_fg3)
    )

    if require_baseline:
        if "baseline_fg3" not in out.columns:
            raise ValueError("require_baseline=True but baseline_fg3 column missing")
        mask &= out["baseline_fg3"].notna()

    cand = out.loc[mask].copy()
    if cand.empty:
        return cand

    # Score: probability first, then delta and volume
    cand["score_jackpot"] = (
        cand[pcol]
        + 0.20 * cand["delta_fg3"].rank(pct=True)
        + 0.10 * cand["pred_fg3a"].rank(pct=True)
    )

    cand = cand.sort_values(["score_jackpot", pcol, "delta_fg3", "pred_fg3a"], ascending=False)

    # Correlation control
    if max_per_team is not None and max_per_team > 0:
        kept = []
        team_counts: dict[str, int] = {}
        for _, r in cand.iterrows():
            t = str(r["team"])
            team_counts.setdefault(t, 0)
            if team_counts[t] >= max_per_team:
                continue
            kept.append(r)
            team_counts[t] += 1
            if len(kept) >= n_legs:
                break
        cand = pd.DataFrame(kept)

    return cand.head(n_legs).reset_index(drop=True)
