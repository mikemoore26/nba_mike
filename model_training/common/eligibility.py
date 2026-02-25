# model_training/common/eligibility.py
from __future__ import annotations
import pandas as pd

def add_prior_games_played(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prior games before the current game_date (no leakage).
    Requires df sorted by player, game_date.
    """
    out = df.copy()
    out["games_played_prior"] = out.groupby("player").cumcount()
    return out

def apply_eligibility_gate(
    df: pd.DataFrame,
    *,
    min_games_prior: int = 5,
    min_expected_min: float | None = None,
    expected_min_col: str = "expected_min_10",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (eligible_df, rejects_df) with explicit reject reasons.
    """
    work = df.copy()

    if "games_played_prior" not in work.columns:
        work = add_prior_games_played(work)

    mask_hist = work["games_played_prior"].fillna(0) >= min_games_prior

    if min_expected_min is not None:
        if expected_min_col not in work.columns:
            raise ValueError(f"min_expected_min set but {expected_min_col} missing.")
        mask_min = work[expected_min_col].fillna(0) >= min_expected_min
    else:
        mask_min = pd.Series(True, index=work.index)

    eligible_mask = mask_hist & mask_min

    id_cols = [c for c in ["game_date","player","team","opp","is_home","season"] if c in work.columns]
    rejects = work.loc[~eligible_mask, id_cols].copy()
    if len(rejects):
        rejects["reject_hist"] = (~mask_hist.loc[rejects.index]).astype(int)
        rejects["reject_min"] = (~mask_min.loc[rejects.index]).astype(int)

    eligible = work.loc[eligible_mask].copy()
    return eligible, rejects