# model_training/common/eligibility.py
from __future__ import annotations

import pandas as pd


def add_prior_games_played(df: pd.DataFrame) -> pd.DataFrame:
    """
    Leakage-safe: counts prior rows per player because df must be sorted by (player, game_date).
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
    require_cols: list[str] | None = None,
    id_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (eligible_df, rejects_df).

    Gates:
      - history: games_played_prior >= min_games_prior
      - minutes: expected_min_col >= min_expected_min (optional)
      - required non-null cols: require_cols (optional)

    Notes:
      - Assumes df is already sorted by (player, game_date) for cumcount to be valid.
      - Does not mutate input df.
    """
    work = df.copy()

    if id_cols is None:
        id_cols = [c for c in ["game_date", "season", "player", "team", "opp", "is_home"] if c in work.columns]

    if "games_played_prior" not in work.columns:
        work = add_prior_games_played(work)

    mask_hist = work["games_played_prior"].fillna(0) >= int(min_games_prior)

    if min_expected_min is not None:
        if expected_min_col not in work.columns:
            raise ValueError(f"min_expected_min set but {expected_min_col} missing.")
        mask_min = work[expected_min_col].fillna(0) >= float(min_expected_min)
    else:
        mask_min = pd.Series(True, index=work.index)

    mask_req = pd.Series(True, index=work.index)
    if require_cols:
        for c in require_cols:
            if c not in work.columns:
                raise ValueError(f"Eligibility require_cols missing from df: {c}")
            mask_req &= work[c].notna()

    eligible_mask = mask_hist & mask_min & mask_req

    rejects = work.loc[~eligible_mask, id_cols].copy()
    if len(rejects):
        rejects["reject_hist"] = (~mask_hist.loc[rejects.index]).astype(int)
        rejects["reject_min"] = (~mask_min.loc[rejects.index]).astype(int)
        rejects["reject_required_missing"] = (~mask_req.loc[rejects.index]).astype(int)

    eligible = work.loc[eligible_mask].copy()
    return eligible, rejects