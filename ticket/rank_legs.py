from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _stat_key(series: pd.Series) -> pd.Series:
    return _safe_str(series).str.lower()


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
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
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "rank_projection_pool expected scored legs input. "
            f"Missing columns: {missing}"
        )

    out = df.copy()

    # numeric core
    for col in [
        "line",
        "pred_mean",
        "p_hit",
        "minutes_proj",
        "edge_raw",
        "edge_norm",
        "edge_rank_pct",
        "score",
        "score_safe",
        "score_balanced",
        "score_lotto",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # backfill score columns if only one exists
    if "score_safe" not in out.columns:
        out["score_safe"] = _safe_numeric(out["score"]) if "score" in out.columns else 0.0
    if "score_balanced" not in out.columns:
        out["score_balanced"] = _safe_numeric(out["score"]) if "score" in out.columns else 0.0
    if "score_lotto" not in out.columns:
        out["score_lotto"] = _safe_numeric(out["score"]) if "score" in out.columns else 0.0

    # backfill edge_raw if missing
    if "edge_raw" not in out.columns:
        pred = _safe_numeric(out["pred_mean"])
        line = _safe_numeric(out["line"])
        side = _safe_str(out["side"]).str.lower()
        raw = pred - line
        out["edge_raw"] = np.where(side.eq("over"), raw, -raw)

    # backfill edge_norm if missing
    if "edge_norm" not in out.columns:
        line_abs = _safe_numeric(out["line"]).abs()
        out["edge_norm"] = _safe_numeric(out["edge_raw"]) / (line_abs + 1.0)

    # backfill edge_rank_pct if missing
    if "edge_rank_pct" not in out.columns:
        base_for_rank = _safe_numeric(out["score_balanced"])
        if base_for_rank.notna().sum() == 0:
            base_for_rank = _safe_numeric(out["p_hit"])
        out["edge_rank_pct"] = base_for_rank.rank(pct=True, method="average")

    for col in ["can_safe", "can_balanced", "can_lotto"]:
        if col not in out.columns:
            out[col] = 1

    if "minutes_proj" not in out.columns:
        out["minutes_proj"] = 0.0

    return out


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["player"] = _safe_str(out["player"])
    out["team"] = _safe_str(out["team"]).str.upper()
    out["opp"] = _safe_str(out["opp"]).str.upper()
    out["stat"] = _stat_key(out["stat"])
    out["side"] = _safe_str(out["side"]).str.lower()

    return out


def _rank_score(df: pd.DataFrame) -> pd.Series:
    score_bal = _safe_numeric(df["score_balanced"])
    score_safe = _safe_numeric(df["score_safe"])
    score_lotto = _safe_numeric(df["score_lotto"])
    edge_rank_pct = _safe_numeric(df["edge_rank_pct"])
    p_hit = _safe_numeric(df["p_hit"], default=0.5)
    edge_raw = _safe_numeric(df["edge_raw"], default=0.0)
    minutes_proj = _safe_numeric(df["minutes_proj"], default=0.0)

    return (
        0.50 * score_bal
        + 0.25 * score_safe
        + 0.15 * score_lotto
        + 0.10 * edge_rank_pct
        + 0.05 * (p_hit - 0.50)
        + 0.03 * edge_raw.clip(lower=0.0)
        + 0.01 * minutes_proj.clip(lower=0.0)
    )


def _assign_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rank_score"] = _rank_score(out)

    out = out.sort_values(
        ["rank_score", "score_balanced", "p_hit", "edge_raw"],
        ascending=False,
    ).reset_index(drop=True)

    out["overall_rank"] = np.arange(1, len(out) + 1)
    out["overall_rank_pct"] = out["rank_score"].rank(method="average", pct=True)

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
    out = df.copy()

    rank_pct = _safe_numeric(out["overall_rank_pct"])
    edge_raw = _safe_numeric(out["edge_raw"])
    p_hit = _safe_numeric(out["p_hit"])

    conditions = [
        (rank_pct >= 0.95) & (edge_raw >= 1.0) & (p_hit >= 0.60),
        (rank_pct >= 0.85) & (edge_raw >= 0.5) & (p_hit >= 0.56),
        (rank_pct >= 0.70) & (p_hit >= 0.53),
    ]
    labels = ["tier_1", "tier_2", "tier_3"]

    out["edge_tier"] = np.select(conditions, labels, default="tier_4")
    return out


def rank_projection_pool(
    df: pd.DataFrame,
    *,
    top_n: int | None = None,
    keep_tiers: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    out = _ensure_cols(df)
    out = _standardize(out)

    # keep anything viable in at least one ticket type
    viable = (
        (_safe_numeric(out["can_safe"]) == 1)
        | (_safe_numeric(out["can_balanced"]) == 1)
        | (_safe_numeric(out["can_lotto"]) == 1)
    )
    out = out.loc[viable].copy()

    if out.empty:
        return out

    out = _assign_ranks(out)
    out = _assign_edge_tiers(out)

    if keep_tiers is not None:
        keep_tiers = tuple(str(x) for x in keep_tiers)
        out = out.loc[out["edge_tier"].isin(keep_tiers)].copy()

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

    existing = [c for c in preferred_cols if c in out.columns]
    other = [c for c in out.columns if c not in existing]

    return out[existing + other].reset_index(drop=True)


def top_ranked_legs(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    ranked = rank_projection_pool(df)
    if ranked.empty:
        return ranked
    return ranked.head(n).reset_index(drop=True)