from __future__ import annotations
import pandas as pd


REQUIRED_COLS = ["line", "side", "p_hit", "pred_mean"]


def score_legs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores betting legs using:
    - probability (p_hit)
    - model edge vs line
    - minutes confidence
    - volatility penalties

    Returns dataframe with:
    - score
    - edge_rank_pct
    """

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"score_legs missing required columns: {missing}")

    df = df.copy()

    # -------------------------
    # CORE EDGE
    # -------------------------
    df["edge_raw"] = df["pred_mean"] - df["line"]

    df["edge_directional"] = df.apply(
        lambda r: r["edge_raw"] if r["side"] == "over" else -r["edge_raw"],
        axis=1,
    )

    df["edge_norm"] = df["edge_directional"] / (df["line"].abs() + 1)

    # -------------------------
    # OPTIONAL INPUTS (SAFE DEFAULTS)
    # -------------------------
    df["minutes_conf"] = df.get("minutes_conf", 1.0).fillna(1.0)
    df["vol_penalty"] = df.get("stat_vol_penalty", 0.0).fillna(0.0)
    df["line_penalty"] = df.get("extreme_line_penalty", 0.0).fillna(0.0)

    # clip to safe ranges
    df["minutes_conf"] = df["minutes_conf"].clip(0.0, 1.0)
    df["vol_penalty"] = df["vol_penalty"].clip(0.0, 1.0)
    df["line_penalty"] = df["line_penalty"].clip(0.0, 1.0)

    # -------------------------
    # FINAL SCORE (SHARP VERSION)
    # -------------------------
    df["score"] = (
        df["p_hit"]
        * (1 + df["edge_norm"])
        * df["minutes_conf"]
        * (1 - df["vol_penalty"])
        * (1 - df["line_penalty"])
    )

    # -------------------------
    # RANKING (REPLACES ranked_pool)
    # -------------------------
    df["edge_rank_pct"] = df["score"].rank(pct=True)

    return df


def build_ranked_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts scored legs into ranked pool
    """
    df = score_legs(df)
    return df.sort_values("score", ascending=False).reset_index(drop=True)