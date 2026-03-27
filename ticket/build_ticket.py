from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TicketSpec:
    name: str
    score_col: str
    leg_count: int


SAFE = TicketSpec("safe", "score_safe", 3)
BAL = TicketSpec("balanced", "score_balanced", 4)
LOT = TicketSpec("lotto", "score_lotto", 5)


def _make_game_key(row):
    return "_".join(sorted([str(row["team"]), str(row["opp"])]))


def _build(df: pd.DataFrame, spec: TicketSpec):
    df = df.copy()

    # sort by score
    df = df.sort_values(spec.score_col, ascending=False).reset_index(drop=True)

    selected = []
    used_players = set()

    for _, row in df.iterrows():
        if len(selected) >= spec.leg_count:
            break

        # only block SAME PLAYER (keep this minimal)
        if row["player"] in used_players:
            continue

        selected.append(row)
        used_players.add(row["player"])

    # -------------------------
    # 🔥 FORCE FILL IF NEEDED
    # -------------------------
    if len(selected) < spec.leg_count:
        for _, row in df.iterrows():
            if len(selected) >= spec.leg_count:
                break

            # allow reuse if necessary
            selected.append(row)

    # hard cap
    selected = selected[:spec.leg_count]

    out = pd.DataFrame(selected).copy()
    out["ticket_type"] = spec.name
    out["leg_order"] = range(1, len(out) + 1)

    return out


def build_all_tickets(df: pd.DataFrame):
    df = df.copy()

    required = ["player", "team", "opp", "stat", "line", "side", "pred_mean", "p_hit"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["game_key"] = df.apply(_make_game_key, axis=1)

    for col in ["score_safe", "score_balanced", "score_lotto"]:
        if col not in df.columns:
            df[col] = df.get("score", 0)

    safe = _build(df, SAFE)
    balanced = _build(df, BAL)
    lotto = _build(df, LOT)

    return {
        "safe": safe,
        "balanced": balanced,
        "lotto": lotto,
    }