from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

from ticket.leg_utility import add_leg_utility
from ticket.dependency import ticket_score


@dataclass(frozen=True)
class TicketSpec:
    name: str
    score_col: str
    utility_col: str
    min_legs: int
    max_legs: int
    preferred_legs: int
    max_same_player: int
    max_same_stat: int
    max_same_team: int
    search_iterations: int


SAFE = TicketSpec(
    name="safe",
    score_col="score_safe",
    utility_col="leg_utility_safe",
    min_legs=3,
    max_legs=5,
    preferred_legs=4,
    max_same_player=1,
    max_same_stat=2,
    max_same_team=2,
    search_iterations=300,
)

BALANCED = TicketSpec(
    name="balanced",
    score_col="score_balanced",
    utility_col="leg_utility_balanced",
    min_legs=5,
    max_legs=7,
    preferred_legs=6,
    max_same_player=1,
    max_same_stat=3,
    max_same_team=2,
    search_iterations=500,
)

LOTTO = TicketSpec(
    name="lotto",
    score_col="score_lotto",
    utility_col="leg_utility_lotto",
    min_legs=8,
    max_legs=12,
    preferred_legs=10,
    max_same_player=1,
    max_same_stat=4,
    max_same_team=3,
    search_iterations=700,
)


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _ensure_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = ["player", "team", "opp", "stat", "line", "side", "pred_mean", "p_hit"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out["player"] = out["player"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["opp"] = out["opp"].astype(str).str.strip().str.upper()
    out["stat"] = out["stat"].astype(str).str.strip().str.lower()
    out["side"] = out["side"].astype(str).str.strip().str.lower()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        out["game_date"] = pd.NA

    numeric_defaults = {
        "line": 0.0,
        "pred_mean": 0.0,
        "p_hit": 0.0,
        "edge_raw": np.nan,
        "score_safe": 0.0,
        "score_balanced": 0.0,
        "score_lotto": 0.0,
        "minutes_proj": 0.0,
        "role_score": 0.5,
        "stability_score": 0.5,
        "usage_score": 0.5,
        "fragility_score": 0.5,
        "projection_rank_score": 0.5,
    }

    for col, default in numeric_defaults.items():
        out[col] = _safe_num(out, col, default=default)

    if out["edge_raw"].isna().all():
        raw = out["pred_mean"] - out["line"]
        out["edge_raw"] = np.where(out["side"].eq("over"), raw, -raw)
    else:
        raw = out["pred_mean"] - out["line"]
        edge_fallback = np.where(out["side"].eq("over"), raw, -raw)
        out["edge_raw"] = out["edge_raw"].fillna(pd.Series(edge_fallback, index=out.index))

    return out


def _dedupe_best_expression(df: pd.DataFrame, utility_col: str) -> pd.DataFrame:
    """
    Keep only one best expression per player+stat.
    This prevents repeated line spam from dominating the pool.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    out = out.sort_values(
        ["player", "stat", utility_col, "p_hit", "edge_raw"],
        ascending=[True, True, False, False, False],
    )

    out = (
        out.groupby(["player", "stat"], as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )

    return out


def _build_ranked_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # wider pool than before; do NOT cut too early
    out["rank_score"] = (
        0.55 * _safe_num(out, "leg_utility_balanced", 0.0)
        + 0.20 * _safe_num(out, "score_balanced", 0.0)
        + 0.15 * _safe_num(out, "p_hit", 0.0)
        + 0.10 * _safe_num(out, "edge_raw", 0.0).clip(lower=0.0)
    )

    out = out.sort_values(
        ["rank_score", "leg_utility_balanced", "p_hit", "edge_raw"],
        ascending=False,
    ).reset_index(drop=True)

    out["overall_rank"] = np.arange(1, len(out) + 1)
    return out


def _choose_ticket_size(spec: TicketSpec, pool_size: int) -> int:
    if pool_size <= 0:
        return 0
    if pool_size < spec.min_legs:
        return pool_size
    return min(spec.preferred_legs, spec.max_legs, pool_size)


def _violates_constraints(candidate: pd.DataFrame, spec: TicketSpec) -> bool:
    if candidate.empty:
        return True

    player_counts = candidate["player"].value_counts()
    if (player_counts > spec.max_same_player).any():
        return True

    stat_counts = candidate["stat"].value_counts()
    if (stat_counts > spec.max_same_stat).any():
        return True

    team_counts = candidate["team"].value_counts()
    if (team_counts > spec.max_same_team).any():
        return True

    # prevent identical player+stat duplicates
    dup_player_stat = candidate.duplicated(subset=["player", "stat"]).any()
    if dup_player_stat:
        return True

    return False


def _sample_candidate(pool: pd.DataFrame, n_target: int, utility_col: str) -> pd.DataFrame:
    """
    Weighted random sample to search ticket space.
    Higher utility gets sampled more often, but we preserve exploration.
    """
    if pool.empty or n_target <= 0:
        return pd.DataFrame()

    sample_pool = pool.copy()

    weights = _safe_num(sample_pool, utility_col, 0.0)
    weights = (weights - weights.min()) + 0.05
    weights = weights.clip(lower=0.01)

    if len(sample_pool) <= n_target:
        return sample_pool.copy()

    chosen_idx = sample_pool.sample(
        n=n_target,
        replace=False,
        weights=weights,
        random_state=random.randint(1, 10_000_000),
    ).index

    return sample_pool.loc[chosen_idx].copy().reset_index(drop=True)


def _greedy_repair(pool: pd.DataFrame, candidate: pd.DataFrame, spec: TicketSpec, utility_col: str) -> pd.DataFrame:
    """
    If random sample violates constraints, rebuild from strongest rows.
    """
    if candidate.empty:
        return pd.DataFrame()

    ordered_pool = pool.sort_values(
        [utility_col, "p_hit", "edge_raw"],
        ascending=False,
    ).reset_index(drop=True)

    selected_rows: list[pd.Series] = []

    for _, row in ordered_pool.iterrows():
        tmp = pd.DataFrame(selected_rows + [row]) if selected_rows else pd.DataFrame([row])
        if _violates_constraints(tmp, spec):
            continue
        selected_rows.append(row)
        if len(selected_rows) >= len(candidate):
            break

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _search_best_ticket(pool: pd.DataFrame, spec: TicketSpec) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()

    n_target = _choose_ticket_size(spec, len(pool))
    if n_target == 0:
        return pd.DataFrame()

    best_ticket = pd.DataFrame()
    best_score = -1e9

    for _ in range(spec.search_iterations):
        candidate = _sample_candidate(pool, n_target=n_target, utility_col=spec.utility_col)
        if candidate.empty:
            continue

        if _violates_constraints(candidate, spec):
            candidate = _greedy_repair(pool, candidate, spec, spec.utility_col)

        if candidate.empty:
            continue

        if len(candidate) < spec.min_legs:
            continue

        if _violates_constraints(candidate, spec):
            continue

        score = ticket_score(
            candidate,
            ticket_type=spec.name,
            utility_col=spec.utility_col,
        )

        if score > best_score:
            best_score = score
            best_ticket = candidate.copy()

    if best_ticket.empty:
        fallback = pool.sort_values(
            [spec.utility_col, "p_hit", "edge_raw"],
            ascending=False,
        ).head(n_target).copy()

        if not fallback.empty:
            fallback = _greedy_repair(pool, fallback, spec, spec.utility_col)

        best_ticket = fallback
        best_score = ticket_score(
            best_ticket,
            ticket_type=spec.name,
            utility_col=spec.utility_col,
        ) if not best_ticket.empty else -1e9

    if best_ticket.empty:
        return pd.DataFrame()

    best_ticket = best_ticket.sort_values(
        [spec.utility_col, "p_hit", "edge_raw"],
        ascending=False,
    ).reset_index(drop=True)

    best_ticket["ticket_type"] = spec.name
    best_ticket["ticket_score"] = best_score
    best_ticket["leg_order"] = np.arange(1, len(best_ticket) + 1)

    return best_ticket


def _build_summary(tickets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for key in ["safe", "balanced", "lotto"]:
        df = tickets.get(key)
        if df is None or df.empty:
            continue

        rows.append(
            {
                "ticket_type": key,
                "n_legs": int(len(df)),
                "n_reb": int((df["stat"] == "reb").sum()),
                "n_ast": int((df["stat"] == "ast").sum()),
                "n_pts": int((df["stat"] == "pts").sum()),
                "n_fg3": int(df["stat"].isin(["fg3", "fg3m"]).sum()),
                "avg_line": float(_safe_num(df, "line").mean()),
                "avg_pred_mean": float(_safe_num(df, "pred_mean").mean()),
                "avg_p_hit": float(_safe_num(df, "p_hit").mean()),
                "avg_edge_raw": float(_safe_num(df, "edge_raw").mean()),
                "avg_leg_utility": float(_safe_num(df, "leg_utility", 0.0).mean()),
                "avg_leg_utility_safe": float(_safe_num(df, "leg_utility_safe", 0.0).mean()),
                "avg_leg_utility_balanced": float(_safe_num(df, "leg_utility_balanced", 0.0).mean()),
                "avg_leg_utility_lotto": float(_safe_num(df, "leg_utility_lotto", 0.0).mean()),
                "ticket_score": float(_safe_num(df, "ticket_score", 0.0).iloc[0]) if "ticket_score" in df.columns else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_all_tickets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = _ensure_input(df)

    # add richer utility layer
    base = add_leg_utility(base)

    # dedupe before ranking / optimization
    base = _dedupe_best_expression(base, utility_col="leg_utility_balanced")

    ranked_pool = _build_ranked_pool(base)

    # keep wider pool for search; do not overcompress
    ranked_pool = ranked_pool.head(150).copy()

    safe = _search_best_ticket(ranked_pool, SAFE)
    balanced = _search_best_ticket(ranked_pool, BALANCED)
    lotto = _search_best_ticket(ranked_pool, LOTTO)

    out = {
        "ranked_pool": ranked_pool,
        "safe": safe,
        "balanced": balanced,
        "lotto": lotto,
    }
    out["summary"] = _build_summary(out)
    return out