from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# UTF-8 SAFETY
# =========================
def _configure_utf8_output() -> None:
    os.environ["PYTHONUTF8"] = "1"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_utf8_output()


GLOBAL_PLAYER_EXPOSURE: dict[str, int] = {}
GLOBAL_GAME_EXPOSURE: dict[str, int] = {}


# =========================
# STAR + USAGE LOGIC
# =========================
def _star_tier(row: pd.Series) -> str:
    mp = float(row.get("minutes_proj", 0.0))

    if mp >= 30:
        return "alpha_star"
    elif mp >= 24:
        return "rotation_core"
    else:
        return "role_player"


def _is_high_usage_team_piece(row: pd.Series) -> bool:
    mp = float(row.get("minutes_proj", 0.0))
    usage = float(row.get("usage_score", 0.0))
    role = float(row.get("role_score", 0.0))
    rank = float(row.get("projection_rank_score", 0.0))

    return (
        (mp >= 30.0)
        and (
            usage >= 0.75
            or role >= 0.80
            or rank >= 0.80
        )
    )


# =========================
# OMIT
# =========================
def _normalize_player_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _load_omit_players() -> set[str]:
    path = Path("data") / "manual" / "omit.csv"

    if not path.exists():
        return set()

    df = pd.read_csv(path)
    col = df.columns[0]

    return {
        _normalize_player_name(x)
        for x in df[col].dropna().astype(str)
    }


# =========================
# HELPERS
# =========================
def _game_key(row: pd.Series) -> str:
    teams = sorted([str(row["team"]), str(row["opp"])])
    return f"{teams[0]}__{teams[1]}"


# =========================
# FILTER POOL
# =========================
def _filter_pool(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    omit = _load_omit_players()
    if omit:
        df = df[
            ~df["player"].astype(str).map(_normalize_player_name).isin(omit)
        ]

    df = df[
        (df["minutes_proj"] > 0) &
        (df["p_hit"] >= 0.45)
    ].copy()

    # stat floors
    df = df[
        ((df["stat"] == "reb") & (df["pred_mean"] >= 4)) |
        ((df["stat"] == "ast") & (df["pred_mean"] >= 3)) |
        ((df["stat"] == "pts") & (df["pred_mean"] >= 8)) |
        ((df["stat"] == "fg3") & (df["pred_mean"] >= 1.1))
    ].copy()

    df["game_key"] = df.apply(_game_key, axis=1)

    # STAR SYSTEM
    df["star_tier"] = df.apply(_star_tier, axis=1)
    df["is_star"] = df["star_tier"].eq("alpha_star")

    # TEAM USAGE SYSTEM
    df["is_high_usage_team_piece"] = df.apply(_is_high_usage_team_piece, axis=1)

    print("[DEBUG] star tiers:")
    print(df["star_tier"].value_counts())

    return df.reset_index(drop=True)


# =========================
# CONSTRAINT CHECK
# =========================
def _can_add(
    row,
    used_players,
    game_counts,
    team_counts,
    stat_counts,
    team_stat_counts,
    team_usage_counts,
    *,
    max_per_game,
    max_per_team,
    max_ast,
    max_fg3,
    max_reb,
    max_stars,
    star_count,
    max_high_usage_per_team,
):
    player = row["player"]
    team = row["team"]
    stat = row["stat"]
    game = row["game_key"]

    is_star = bool(row["is_star"])
    is_usage = bool(row["is_high_usage_team_piece"])

    if player in used_players:
        return False

    if game_counts.get(game, 0) >= max_per_game:
        return False

    if team_counts.get(team, 0) >= max_per_team:
        return False

    if team_stat_counts.get((team, stat), 0) >= 1:
        return False

    if is_star and star_count >= max_stars:
        return False

    if is_usage and team_usage_counts.get(team, 0) >= max_high_usage_per_team:
        return False

    if stat == "ast" and stat_counts.get("ast", 0) >= max_ast:
        return False
    if stat == "fg3" and stat_counts.get("fg3", 0) >= max_fg3:
        return False
    if stat == "reb" and stat_counts.get("reb", 0) >= max_reb:
        return False

    return True


# =========================
# ADD ROW
# =========================
def _add_row(
    row,
    ticket,
    used_players,
    game_counts,
    team_counts,
    stat_counts,
    team_stat_counts,
    team_usage_counts,
    star_count_box,
):
    ticket.append(row)

    used_players.add(row["player"])
    game_counts[row["game_key"]] = game_counts.get(row["game_key"], 0) + 1
    team_counts[row["team"]] = team_counts.get(row["team"], 0) + 1
    stat_counts[row["stat"]] = stat_counts.get(row["stat"], 0) + 1

    key = (row["team"], row["stat"])
    team_stat_counts[key] = team_stat_counts.get(key, 0) + 1

    if row["is_star"]:
        star_count_box["count"] += 1

    if row["is_high_usage_team_piece"]:
        team_usage_counts[row["team"]] = team_usage_counts.get(row["team"], 0) + 1


# =========================
# BUILD TICKET
# =========================
def _build_ticket(
    pool,
    *,
    score_col,
    min_legs,
    max_legs,
    temperature,
    max_per_game,
    max_per_team,
    max_ast,
    max_fg3,
    max_reb,
    max_stars,
    max_high_usage_per_team,
):
    if pool.empty:
        return pd.DataFrame()

    df = pool.sort_values(score_col, ascending=False).copy()

    ticket = []
    used_players = set()
    game_counts = {}
    team_counts = {}
    stat_counts = {}
    team_stat_counts = {}
    team_usage_counts = {}
    star_count_box = {"count": 0}

    for _, row in df.iterrows():
        if len(ticket) >= max_legs:
            break

        if _can_add(
            row,
            used_players,
            game_counts,
            team_counts,
            stat_counts,
            team_stat_counts,
            team_usage_counts,
            max_per_game=max_per_game,
            max_per_team=max_per_team,
            max_ast=max_ast,
            max_fg3=max_fg3,
            max_reb=max_reb,
            max_stars=max_stars,
            star_count=star_count_box["count"],
            max_high_usage_per_team=max_high_usage_per_team,
        ):
            _add_row(
                row,
                ticket,
                used_players,
                game_counts,
                team_counts,
                stat_counts,
                team_stat_counts,
                team_usage_counts,
                star_count_box,
            )

    return pd.DataFrame(ticket)


# =========================
# MAIN BUILD
# =========================
def build_projection_tickets(df: pd.DataFrame):
    df = _filter_pool(df)

    safe = _build_ticket(
        df,
        score_col="safe_score",
        min_legs=3,
        max_legs=5,
        temperature=0.4,
        max_per_game=1,
        max_per_team=2,
        max_ast=2,
        max_fg3=1,
        max_reb=2,
        max_stars=2,
        max_high_usage_per_team=1,
    )

    balanced = _build_ticket(
        df,
        score_col="balanced_score",
        min_legs=5,
        max_legs=7,
        temperature=0.6,
        max_per_game=2,
        max_per_team=2,
        max_ast=2,
        max_fg3=2,
        max_reb=3,
        max_stars=2,
        max_high_usage_per_team=1,
    )

    lotto = _build_ticket(
        df,
        score_col="lotto_score",
        min_legs=10,
        max_legs=12,
        temperature=0.9,
        max_per_game=2,
        max_per_team=2,
        max_ast=3,
        max_fg3=3,
        max_reb=4,
        max_stars=4,
        max_high_usage_per_team=2,
    )

    return safe, balanced, lotto


# =========================
# ENTRY
# =========================
def main():
    run_date = datetime.today().strftime("%Y-%m-%d")
    path = Path("results") / run_date / "projection_legs_scored.csv"

    df = pd.read_csv(path)

    safe, balanced, lotto = build_projection_tickets(df)

    out = Path("results") / run_date / "tickets"
    out.mkdir(parents=True, exist_ok=True)

    safe.to_csv(out / "ticket_safe.csv", index=False)
    balanced.to_csv(out / "ticket_balanced.csv", index=False)
    lotto.to_csv(out / "ticket_lotto.csv", index=False)

    print("[DONE] Tickets built with full constraint system")


if __name__ == "__main__":
    main()