from __future__ import annotations

import pandas as pd
import numpy as np


REQUIRED_COLS = ["line", "side", "p_hit", "pred_mean"]


def _series_or_default(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def score_legs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["pred_mean", "line"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # -----------------------------------
    # STEP 1: Create fallback line if missing
    # -----------------------------------
    if "line" not in df.columns:
        df["line"] = df["pred_mean"].round()

    # -----------------------------------
    # STEP 2: Side
    # -----------------------------------
    df["side"] = np.where(df["pred_mean"] > df["line"], "over", "under")

    # -----------------------------------
    # STEP 3: Edge
    # -----------------------------------
    df["edge_raw"] = df["pred_mean"] - df["line"]
    df["edge_abs"] = df["edge_raw"].abs()

    # -----------------------------------
    # STEP 4: p_hit (approx fallback)
    # -----------------------------------
    if "p_hit" not in df.columns:
        # simple approximation
        df["p_hit"] = 0.5 + np.tanh(df["edge_raw"] / 2) * 0.25

    # -----------------------------------
    # STEP 5: Scores (NO HARD FILTER)
    # -----------------------------------
    df["score_safe"] = (
        0.65 * df["p_hit"]
        + 0.35 * (df["edge_abs"] / (df["edge_abs"].max() + 1e-6))
    )

    df["score_balanced"] = (
        0.55 * df["p_hit"]
        + 0.45 * (df["edge_abs"] / (df["edge_abs"].max() + 1e-6))
    )

    df["score_lotto"] = (
        0.45 * df["p_hit"]
        + 0.55 * (df["edge_abs"] / (df["edge_abs"].max() + 1e-6))
    )

    # -----------------------------------
    # STEP 6: KEEP TOP N PER SLATE
    # -----------------------------------
    df = df.sort_values("score_balanced", ascending=False)

    # 🔥 THIS IS KEY
    df = df.head(50)

    return df.reset_index(drop=True)
def build_ranked_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = score_legs(df)
    return out.sort_values("score", ascending=False).reset_index(drop=True)