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

    for col in PROJECTION_SCHEMA:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[PROJECTION_SCHEMA].copy()
    return out