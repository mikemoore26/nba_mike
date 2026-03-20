from __future__ import annotations

import pandas as pd


PREDICTION_SCHEMA = [
    "game_date",
    "player",
    "team",
    "opp",
    "stat",
    "line",
    "side",
    "pred_mean",
    "baseline_mean",
    "delta_mean",
    "dist_name",
    "dispersion",
    "p_hit",
    "p_over",
    "p_under",
    "minutes_proj",
    "is_eligible",
    "eligibility_reason",
    "model_name",
    "model_version",
]


def enforce_prediction_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in PREDICTION_SCHEMA:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[PREDICTION_SCHEMA].copy()
    return out