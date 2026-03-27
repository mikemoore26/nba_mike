#ticket/build_ticket.py
# this file contains the core ticket-building logic, which is shared between live ticket generation and backtesting.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ticket.projection_ranker import rank_projection_pool
from ticket.score_legs import score_legs


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
STAT_PROB_SCALE = {
    "pts": 4.0,
    "reb": 2.0,
    "ast": 1.6,
    "fg3": 1.0,
    "fg3m": 1.0,
}


def _stat_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _sigmoid(x: pd.Series) -> pd.Series:
    x = _safe_numeric(x)
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_confidence_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "confidence_tier" in out.columns:
        return out

    minutes = _safe_numeric(out.get("minutes_proj", pd.Series(0.0, index=out.index)))
    p_hit = _safe_numeric(out.get("p_hit", pd.Series(0.5, index=out.index)), default=0.5)

    conditions = [
        (minutes >= 28) & (p_hit >= 0.62),
        (minutes >= 18) & (p_hit >= 0.55),
    ]
    values = ["high_conf", "medium_conf"]

    out["confidence_tier"] = np.select(conditions, values, default="low_conf")
    return out


def _ensure_prediction_leg_schema(preds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prediction-only rows into pseudo line-level legs so ticket building
    can work during backtesting before real sportsbook lines exist.

    Current policy:
    - line := baseline_mean when available, else rounded pred_mean
    - side := over if pred_mean >= line else under
    - p_hit := sigmoid((pred_mean - line) / stat_scale)

    This is NOT a sportsbook market model. It is a projection-selection proxy.
    """
    out = preds_df.copy()

    out["pred_mean"] = _safe_numeric(out.get("pred_mean", pd.Series(index=out.index)))
    out["baseline_mean"] = _safe_numeric(out.get("baseline_mean", pd.Series(index=out.index)), default=np.nan)
    out["minutes_proj"] = _safe_numeric(out.get("minutes_proj", pd.Series(index=out.index)))
    out["delta_mean"] = _safe_numeric(out.get("delta_mean", pd.Series(index=out.index)), default=np.nan)

    if "line" not in out.columns:
        # Use baseline as the pseudo line when available.
        # Fallback to rounded prediction if baseline is absent.
        rounded_pred = out["pred_mean"].round(0)
        out["line"] = out["baseline_mean"].where(out["baseline_mean"].notna(), rounded_pred)

    out["line"] = _safe_numeric(out["line"], default=np.nan)

    if "side" not in out.columns:
        out["side"] = np.where(out["pred_mean"] >= out["line"], "over", "under")

    if "p_hit" not in out.columns:
        stat = _stat_key(out["stat"])
        scale = stat.map(STAT_PROB_SCALE).fillna(2.0)
        edge = out["pred_mean"] - out["line"]

        # If side is under, invert edge direction so positive always means
        # model agrees with chosen side.
        under_mask = out["side"].astype(str).str.lower().eq("under")
        edge = np.where(under_mask, out["line"] - out["pred_mean"], edge)

        out["p_hit"] = _sigmoid(pd.Series(edge, index=out.index) / scale)

    out["p_hit"] = _safe_numeric(out["p_hit"], default=0.5).clip(1e-6, 1 - 1e-6)

    if "is_eligible" not in out.columns:
        out["is_eligible"] = (
            out["pred_mean"].notna()
            & out["minutes_proj"].notna()
            & (out["minutes_proj"] > 0)
        ).astype(int)
    else:
        out["is_eligible"] = pd.to_numeric(out["is_eligible"], errors="coerce").fillna(0).astype(int)

    out = _ensure_confidence_tier(out)

    return out


def _game_key(row: pd.Series) -> str:
    return "_".join(sorted([str(row["team"]), str(row["opp"])]))


# ------------------------------------------------------------
# Ticket builder
# ------------------------------------------------------------
def build_ticket(
    df: pd.DataFrame,
    score_col: str,
    *,
    min_legs: int,
    max_legs: int,
) -> pd.DataFrame:
    work = df.sort_values(score_col, ascending=False).copy()

    # Ticket-specific floor rules
    if score_col == "score_safe":
        work = work[
            (work["can_safe"] == 1)
            & (work["minutes_proj"] >= 22)
            & (work["confidence_tier"].isin(["high_conf", "medium_conf"]))
        ].copy()
    elif score_col == "score_balanced":
        work = work[
            (work["can_balanced"] == 1)
            & (work["minutes_proj"] >= 18)
            & (work["confidence_tier"].isin(["high_conf", "medium_conf"]))
        ].copy()
    elif score_col == "score_lotto":
        work = work[
            (work["can_lotto"] == 1)
            & (work["minutes_proj"] >= 10)
        ].copy()

    selected = []
    used_players = set()
    used_teams = set()
    used_stats = set()
    used_games = set()

    # pass 1: highest quality + diversity
    for _, row in work.iterrows():
        if len(selected) >= max_legs:
            break

        player = row["player"]
        team = row["team"]
        stat = row["stat"]
        game = _game_key(row)

        if player in used_players:
            continue
        if team in used_teams:
            continue
        if game in used_games:
            continue

        # prefer stat diversity while filling toward min_legs
        if stat in used_stats and len(used_stats) < min_legs:
            continue

        selected.append(row)
        used_players.add(player)
        used_teams.add(team)
        used_stats.add(stat)
        used_games.add(game)

    # pass 2: relax stat diversity
    if len(selected) < min_legs:
        for _, row in work.iterrows():
            if len(selected) >= min_legs:
                break

            player = row["player"]
            team = row["team"]
            game = _game_key(row)

            if player in used_players:
                continue
            if team in used_teams:
                continue
            if game in used_games:
                continue

            selected.append(row)
            used_players.add(player)
            used_teams.add(team)
            used_games.add(game)

    # pass 3: relax game uniqueness
    if len(selected) < min_legs:
        for _, row in work.iterrows():
            if len(selected) >= min_legs:
                break

            player = row["player"]
            team = row["team"]

            if player in used_players:
                continue
            if team in used_teams:
                continue

            selected.append(row)
            used_players.add(player)
            used_teams.add(team)

    # pass 4: final fallback = unique players only
    if len(selected) < min_legs:
        for _, row in work.iterrows():
            if len(selected) >= min_legs:
                break

            player = row["player"]
            if player in used_players:
                continue

            selected.append(row)
            used_players.add(player)

    if not selected:
        return pd.DataFrame(columns=work.columns)

    out = pd.DataFrame(selected).reset_index(drop=True)
    out["ticket_score_col"] = score_col
    return out


def build_all_tickets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = _ensure_prediction_leg_schema(df)

    work = work[work["is_eligible"] == 1].copy()
    if work.empty:
        raise ValueError("No eligible players for tickets")

    scored = score_legs(work)
    ranked = rank_projection_pool(scored)

    safe = build_ticket(
        ranked,
        "score_safe",
        min_legs=3,
        max_legs=5,
    )
    if not safe.empty:
        safe["ticket_id"] = "safe_1"

    balanced = build_ticket(
        ranked,
        "score_balanced",
        min_legs=5,
        max_legs=7,
    )
    if not balanced.empty:
        balanced["ticket_id"] = "balanced_1"

    lotto = build_ticket(
        ranked,
        "score_lotto",
        min_legs=10,
        max_legs=20,
    )
    if not lotto.empty:
        lotto["ticket_id"] = "lotto_1"

    summary = pd.DataFrame(
        [
            {
                "ticket_name": "safe",
                "n_legs": len(safe),
                "avg_pred_mean": safe["pred_mean"].mean() if not safe.empty else 0.0,
                "avg_minutes_proj": safe["minutes_proj"].mean() if not safe.empty else 0.0,
                "avg_p_hit": safe["p_hit"].mean() if not safe.empty else 0.0,
            },
            {
                "ticket_name": "balanced",
                "n_legs": len(balanced),
                "avg_pred_mean": balanced["pred_mean"].mean() if not balanced.empty else 0.0,
                "avg_minutes_proj": balanced["minutes_proj"].mean() if not balanced.empty else 0.0,
                "avg_p_hit": balanced["p_hit"].mean() if not balanced.empty else 0.0,
            },
            {
                "ticket_name": "lotto",
                "n_legs": len(lotto),
                "avg_pred_mean": lotto["pred_mean"].mean() if not lotto.empty else 0.0,
                "avg_minutes_proj": lotto["minutes_proj"].mean() if not lotto.empty else 0.0,
                "avg_p_hit": lotto["p_hit"].mean() if not lotto.empty else 0.0,
            },
        ]
    )

    return {
        "ranked_pool": ranked,
        "safe": safe,
        "balanced": balanced,
        "lotto": lotto,
        "summary": summary,
    }


def build_all_tickets_from_predictions(
    *,
    preds_df: pd.DataFrame,
    run_date: str,
    write_output: bool = False,
    results_root: str = "results_backtest",
) -> dict[str, pd.DataFrame]:
    out = build_all_tickets(preds_df)

    if write_output:
        results_dir = Path(results_root) / run_date
        results_dir.mkdir(parents=True, exist_ok=True)

        file_map = {
            "ranked_pool": "ranked_pool.csv",
            "safe": "ticket_safe.csv",
            "balanced": "ticket_balanced.csv",
            "lotto": "ticket_lotto.csv",
            "summary": "ticket_summary.csv",
        }

        for key, filename in file_map.items():
            df = out.get(key)
            if df is not None and not df.empty:
                df.to_csv(results_dir / filename, index=False)

    return out