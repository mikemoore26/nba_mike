from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _stat_key(series: pd.Series) -> pd.Series:
    return _safe_str(series).str.lower()


def _ensure_scored(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "game_date",
        "player",
        "team",
        "opp",
        "stat",
        "line",
        "side",
        "pred_mean",
        "p_hit",
        "minutes_proj",
        "score_safe",
        "score_balanced",
        "score_lotto",
        "can_safe",
        "can_balanced",
        "can_lotto",
        "edge_raw",
        "edge_norm",
        "edge_rank_pct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "rank_projection_pool expected scored legs input. "
            f"Missing columns: {missing}"
        )
    return df.copy()


def _build_default_rank_score(df: pd.DataFrame) -> pd.Series:
    """
    One cross-ticket ranking score for the shared ranked pool.

    Why this construction:
    - balanced score is the best general-purpose baseline
    - safe score matters because stable legs should rise in the pool
    - lotto score matters a bit because true ceiling spots should not disappear
    - edge rank percentile matters because your backtests showed selection quality
      matters more than raw broad forecast quality, especially for REB
    """
    score_bal = _safe_numeric(df["score_balanced"])
    score_safe = _safe_numeric(df["score_safe"])
    score_lotto = _safe_numeric(df["score_lotto"])
    edge_rank_pct = _safe_numeric(df["edge_rank_pct"])
    p_hit = _safe_numeric(df["p_hit"], default=0.5)

    return (
        0.55 * score_bal
        + 0.30 * score_safe
        + 0.15 * score_lotto
        + 0.20 * edge_rank_pct
        + 0.10 * (p_hit - 0.5)
    )


def _apply_pool_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that are at least viable in one ticket family.
    This avoids polluting ranked_pool with dead legs.
    """
    out = df.copy()

    can_any = (
        (_safe_numeric(out["can_safe"]) == 1)
        | (_safe_numeric(out["can_balanced"]) == 1)
        | (_safe_numeric(out["can_lotto"]) == 1)
    )

    out = out.loc[can_any].copy()

    # require positive directional edge
    out = out.loc[_safe_numeric(out["edge_raw"]) > 0].copy()

    # require at least minimally sane probability
    out = out.loc[_safe_numeric(out["p_hit"], default=0.0) > 0.50].copy()

    return out


def _assign_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rank_score"] = _build_default_rank_score(out)

    out["overall_rank"] = (
        out["rank_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    out["overall_rank_pct"] = (
        out["rank_score"]
        .rank(method="average", pct=True)
    )

    out["stat_rank"] = (
        out.groupby("stat", dropna=False)["rank_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    out["stat_rank_pct"] = (
        out.groupby("stat", dropna=False)["rank_score"]
        .rank(method="average", pct=True)
    )

    out["team_rank"] = (
        out.groupby(["game_date", "team"], dropna=False)["rank_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    out["player_rank"] = (
        out.groupby(["game_date", "player"], dropna=False)["rank_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return out


def _assign_edge_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiers for later ticket construction.

    Tier meaning:
    - tier_1: real anchor
    - tier_2: strong support
    - tier_3: usable but lower-quality filler
    """
    out = df.copy()

    score = _safe_numeric(out["rank_score"])
    rank_pct = _safe_numeric(out["overall_rank_pct"])
    edge_raw = _safe_numeric(out["edge_raw"])
    p_hit = _safe_numeric(out["p_hit"], default=0.5)

    conditions = [
        (rank_pct >= 0.95) & (edge_raw >= 1.0) & (p_hit >= 0.60),
        (rank_pct >= 0.85) & (edge_raw >= 0.5) & (p_hit >= 0.56),
        (rank_pct >= 0.70) & (score > 0),
    ]
    labels = ["tier_1", "tier_2", "tier_3"]

    out["edge_tier"] = np.select(conditions, labels, default="tier_4")

    return out


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["player"] = _safe_str(out["player"])
    out["team"] = _safe_str(out["team"]).str.upper()
    out["opp"] = _safe_str(out["opp"]).str.upper()
    out["stat"] = _stat_key(out["stat"])
    out["side"] = _safe_str(out["side"]).str.lower()

    numeric_cols = [
        "line",
        "pred_mean",
        "p_hit",
        "minutes_proj",
        "edge_raw",
        "edge_norm",
        "edge_rank_pct",
        "score_safe",
        "score_balanced",
        "score_lotto",
        "rank_score",
        "overall_rank_pct",
        "stat_rank_pct",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def rank_projection_pool(
    df: pd.DataFrame,
    *,
    top_n: int | None = None,
    keep_tiers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Build a canonical ranked pool from scored legs.

    Parameters
    ----------
    df:
        Output of ticket.score_legs.score_legs(...)
    top_n:
        Optional global cutoff after ranking.
    keep_tiers:
        Optional subset of edge tiers to keep, e.g. ("tier_1", "tier_2")

    Returns
    -------
    pd.DataFrame
        Ranked candidate pool for debugging and ticket construction.
    """
    out = _ensure_scored(df)
    out = _standardize_columns(out)
    out = _apply_pool_filters(out)

    if out.empty:
        return out

    out = _assign_ranks(out)
    out = _assign_edge_tiers(out)

    if keep_tiers is not None:
        keep_tiers = tuple(str(x) for x in keep_tiers)
        out = out.loc[out["edge_tier"].isin(keep_tiers)].copy()

    out = out.sort_values(
        [
            "rank_score",
            "score_balanced",
            "score_safe",
            "p_hit",
            "edge_raw",
            "minutes_proj",
        ],
        ascending=False,
    ).reset_index(drop=True)

    out["overall_rank"] = np.arange(1, len(out) + 1)

    if top_n is not None:
        out = out.head(int(top_n)).copy()

    preferred_cols = [
        "game_date",
        "player",
        "team",
        "opp",
        "stat",
        "side",
        "line",
        "pred_mean",
        "p_hit",
        "minutes_proj",
        "edge_raw",
        "edge_norm",
        "edge_rank_pct",
        "score_safe",
        "score_balanced",
        "score_lotto",
        "rank_score",
        "overall_rank",
        "overall_rank_pct",
        "stat_rank",
        "stat_rank_pct",
        "team_rank",
        "player_rank",
        "edge_tier",
        "can_safe",
        "can_balanced",
        "can_lotto",
    ]

    existing_cols = [c for c in preferred_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing_cols]

    return out[existing_cols + other_cols].reset_index(drop=True)


def top_ranked_legs(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    ranked = rank_projection_pool(df)
    if ranked.empty:
        return ranked
    return ranked.head(n).reset_index(drop=True)