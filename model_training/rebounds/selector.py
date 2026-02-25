# model_training/rebounds/selectors.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# REBOUNDS TICKET SELECTORS
# expects output from predict_game_reb
# ============================================================

def _require_cols(df: pd.DataFrame, req: set[str]) -> None:
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_matchup_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["matchup_key"] = np.where(
        out["is_home"].astype(int) == 1,
        out["opp"].astype(str) + "@" + out["team"].astype(str),
        out["team"].astype(str) + "@" + out["opp"].astype(str),
    )
    return out


# ------------------------------------------------------------
# JACKPOT
# ------------------------------------------------------------
def select_jackpot_reb_ticket(
    df: pd.DataFrame,
    *,
    n_legs: int = 3,
    over_baseline_col: str = "p_over_baseline_2",
    min_pred_reb: float = 7.0,
    min_p_over_baseline: float = 0.18,
    min_delta_reb: float = 2.0,
    max_per_team: int = 1,
) -> pd.DataFrame:

    req = {"player","team","opp","is_home","pred_reb","baseline_reb","delta_reb", over_baseline_col}
    _require_cols(df, req)

    out = add_matchup_key(df)

    pool = out[
        (out["pred_reb"] >= min_pred_reb) &
        (out["delta_reb"] >= min_delta_reb) &
        (out[over_baseline_col] >= min_p_over_baseline)
    ].copy()

    if pool.empty:
        return pool

    pool = pool.sort_values(
        [over_baseline_col, "delta_reb", "pred_reb"],
        ascending=False,
    )

    if max_per_team is not None:
        pool["_team_rank"] = pool.groupby("team").cumcount()
        pool = pool[pool["_team_rank"] < max_per_team].copy()
        pool.drop(columns=["_team_rank"], inplace=True)

    return pool.head(n_legs).reset_index(drop=True)


# ------------------------------------------------------------
# COVERAGE
# ------------------------------------------------------------
def select_matchup_coverage_reb_ticket(
    df: pd.DataFrame,
    *,
    players_per_matchup: int = 2,
    insurance_per_matchup: int = 1,
    min_pred_reb: float = 6.0,
    p_col: str | None = None,
    min_prob: float | None = None,
    min_delta_reb: float = 0.0,
    max_legs: int | None = None,
) -> pd.DataFrame:

    out = add_matchup_key(df)

    if p_col is None:
        cols = [c for c in out.columns if c.startswith("p_over_baseline_")]
        if not cols:
            raise ValueError("No p_over_baseline_* column found.")
        p_col = sorted(cols)[-1]

    if min_prob is None:
        min_prob = 0.18

    req = {"player","team","opp","is_home","pred_reb","delta_reb", p_col}
    _require_cols(out, req)

    pool = out[
        (out["pred_reb"] >= min_pred_reb) &
        (out[p_col] >= min_prob) &
        (out["delta_reb"] >= min_delta_reb)
    ].copy()

    if pool.empty:
        return pool

    pool = pool.sort_values(
        ["matchup_key", p_col, "delta_reb", "pred_reb"],
        ascending=[True, False, False, False],
    )

    pool["_rank"] = pool.groupby("matchup_key").cumcount()
    keep = players_per_matchup + insurance_per_matchup

    ticket = pool[pool["_rank"] < keep].copy()

    if max_legs is not None and len(ticket) > max_legs:
        ticket = ticket.sort_values([p_col, "delta_reb", "pred_reb"], ascending=False).head(max_legs)

    return ticket.drop(columns=["_rank"]).reset_index(drop=True)


# ------------------------------------------------------------
# PENCIL LABELS
# ------------------------------------------------------------
def assign_pencil_decision_reb(
    df: pd.DataFrame,
    *,
    p_over_line_col: str | None = None,
) -> pd.DataFrame:

    out = df.copy()

    prob_col = None
    if p_over_line_col and p_over_line_col in out.columns:
        prob_col = p_over_line_col
    else:
        cols = [c for c in out.columns if c.startswith("p_over_baseline_")]
        if cols:
            prob_col = sorted(cols)[-1]

    if prob_col is None:
        out["pencil"] = "coverage_only"
        return out

    conditions = [
        (out[prob_col] >= 0.65) & (out["pred_reb"] >= 9.0),
        (out[prob_col] >= 0.58) & (out["pred_reb"] >= 7.0),
    ]
    choices = ["smash", "over"]

    out["pencil"] = np.select(conditions, choices, default="coverage_only")
    return out
