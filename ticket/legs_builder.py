# model_training/ticketing/legs_builders.py
from __future__ import annotations
import pandas as pd
from .schema import require_leg_cols

def build_legs_from_model(
    df: pd.DataFrame,
    *,
    stat: str,
    line_col: str,
    mean_col: str,
    p_hit_col: str,
    side: str = "over",
) -> pd.DataFrame:
    out = df.copy()

    # required identity columns should already exist from your pipeline
    out["stat"] = stat
    out["side"] = side
    out["line"] = out[line_col].astype(float)
    out["pred_mean"] = out[mean_col].astype(float)
    out["p_hit"] = out[p_hit_col].astype(float)

    # optional but useful if you have them
    # out["pred_sd"] = out.get("pred_sd", pd.NA)

    require_leg_cols(out, name=f"legs_{stat}")
    return out