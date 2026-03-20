# model_training/ticketing/scoring.py
from __future__ import annotations
import numpy as np
import pandas as pd

def add_leg_scores(
    legs: pd.DataFrame,
    *,
    w_p: float = 1.0,
    w_margin: float = 0.15,
    cap_margin: float = 10.0,
) -> pd.DataFrame:
    out = legs.copy()

    margin = (out["pred_mean"] - out["line"]).clip(-cap_margin, cap_margin)
    out["margin"] = margin

    # simple linear score (you can replace w/ EV if you have odds)
    out["score"] = w_p * out["p_hit"] + w_margin * margin

    return out