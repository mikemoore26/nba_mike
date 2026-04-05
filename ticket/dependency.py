from __future__ import annotations

from itertools import combinations

import pandas as pd


def pairwise_penalty(a: dict, b: dict) -> float:
    """
    Heuristic dependency penalty.

    This is NOT a true covariance model.
    It is a portfolio-risk control layer until we backtest enough
    to estimate empirical pairwise interactions.
    """
    penalty = 0.0

    a_team = str(a.get("team", "")).upper()
    b_team = str(b.get("team", "")).upper()
    a_opp = str(a.get("opp", "")).upper()
    b_opp = str(b.get("opp", "")).upper()
    a_stat = str(a.get("stat", "")).lower()
    b_stat = str(b.get("stat", "")).lower()
    a_side = str(a.get("side", "")).lower()
    b_side = str(b.get("side", "")).lower()

    a_usage = float(a.get("usage_score", 0.0) or 0.0)
    b_usage = float(b.get("usage_score", 0.0) or 0.0)

    a_role = float(a.get("role_score", 0.0) or 0.0)
    b_role = float(b.get("role_score", 0.0) or 0.0)

    a_minutes = float(a.get("minutes_proj", 0.0) or 0.0)
    b_minutes = float(b.get("minutes_proj", 0.0) or 0.0)

    same_team = a_team == b_team and a_team != ""
    same_game = (
        (a_team == b_opp and b_team == a_opp and a_team != "" and b_team != "")
        or same_team
    )
    same_stat = a_stat == b_stat and a_stat != ""

    # broad concentration
    if same_game:
        penalty += 0.04

    if same_team:
        penalty += 0.06

    if same_stat:
        penalty += 0.05

    # same-team usage collision
    if same_team and a_usage >= 0.70 and b_usage >= 0.70:
        penalty += 0.10

    # same-team primary role collision
    if same_team and a_role >= 0.75 and b_role >= 0.75:
        penalty += 0.06

    # both FG3 overs = high-variance clustering
    if a_stat in {"fg3", "fg3m"} and b_stat in {"fg3", "fg3m"}:
        penalty += 0.08

    # same-team PTS + AST can be positively linked or cannibalistic depending on role,
    # but as a first pass treat as dependency exposure that needs penalizing
    if same_team and {a_stat, b_stat} == {"pts", "ast"}:
        penalty += 0.05

    # same-team REB + REB is direct board share competition
    if same_team and a_stat == "reb" and b_stat == "reb":
        penalty += 0.08

    # same-team AST + FG3 stack can be fragile to shooting variance
    if same_team and {a_stat, b_stat} in [
        {"ast", "fg3"},
        {"ast", "fg3m"},
    ]:
        penalty += 0.05

    # both shallow-minute plays are fragile together
    if a_minutes < 24 and b_minutes < 24:
        penalty += 0.05

    # over/over same-game exposure slightly riskier than mixed side exposure
    if same_game and a_side == "over" and b_side == "over":
        penalty += 0.03

    return penalty


def ticket_dependency_penalty(ticket_df: pd.DataFrame) -> float:
    if ticket_df is None or ticket_df.empty or len(ticket_df) <= 1:
        return 0.0

    rows = ticket_df.to_dict("records")
    total = 0.0

    for a, b in combinations(rows, 2):
        total += pairwise_penalty(a, b)

    return float(total)


def stat_concentration_penalty(ticket_df: pd.DataFrame, ticket_type: str) -> float:
    """
    Soft portfolio composition control.
    We do NOT want hard quota-only behavior.
    """
    if ticket_df is None or ticket_df.empty:
        return 0.0

    counts = ticket_df["stat"].astype(str).str.lower().value_counts(normalize=True)

    fg3_share = counts.get("fg3", 0.0) + counts.get("fg3m", 0.0)
    pts_share = counts.get("pts", 0.0)
    reb_share = counts.get("reb", 0.0)
    ast_share = counts.get("ast", 0.0)

    penalty = 0.0

    if ticket_type == "safe":
        if fg3_share > 0.25:
            penalty += (fg3_share - 0.25) * 0.80
        if pts_share > 0.45:
            penalty += (pts_share - 0.45) * 0.30
    elif ticket_type == "balanced":
        if fg3_share > 0.33:
            penalty += (fg3_share - 0.33) * 0.60
        if pts_share > 0.50:
            penalty += (pts_share - 0.50) * 0.25
    else:  # lotto
        if fg3_share > 0.45:
            penalty += (fg3_share - 0.45) * 0.40

    # encourage not fully collapsing out AST/REB
    if reb_share < 0.10:
        penalty += 0.08
    if ast_share < 0.10:
        penalty += 0.08

    return float(penalty)


def ticket_score(ticket_df: pd.DataFrame, ticket_type: str, utility_col: str) -> float:
    if ticket_df is None or ticket_df.empty:
        return -1e9

    leg_value = pd.to_numeric(ticket_df[utility_col], errors="coerce").fillna(0.0).sum()
    dep_pen = ticket_dependency_penalty(ticket_df)
    stat_pen = stat_concentration_penalty(ticket_df, ticket_type=ticket_type)

    return float(leg_value - dep_pen - stat_pen)