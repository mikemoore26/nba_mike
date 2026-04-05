from __future__ import annotations

import pandas as pd


PROJECTION_SCHEMA = [
    "game_date",
    "player",
    "team",
    "opp",
    "stat",
    "pred_mean",
    "baseline_mean",
    "delta_mean",
    "minutes_proj",
    "dist_name",
    "dispersion",
    "is_eligible",
    "eligibility_reason",
    "model_name",
    "model_version",
]


def enforce_projection_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # -------------------------
    # ensure required columns exist
    # -------------------------
    for col in PROJECTION_SCHEMA:
        if col not in out.columns:
            out[col] = pd.NA

    # -------------------------
    # KEEP extra columns (like probabilities)
    # -------------------------
    extra_cols = [c for c in out.columns if c not in PROJECTION_SCHEMA]

    # final column order:
    # required first, then extras
    ordered_cols = PROJECTION_SCHEMA + extra_cols

    # remove duplicates while preserving order
    seen = set()
    final_cols = []
    for c in ordered_cols:
        if c not in seen:
            final_cols.append(c)
            seen.add(c)

    return out[final_cols].copy()