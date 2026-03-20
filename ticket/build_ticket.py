from __future__ import annotations

import pandas as pd

from ticket.projection_ranker import rank_projection_pool


def build_ticket(
    df: pd.DataFrame,
    score_col: str,
    *,
    min_legs: int,
    max_legs: int,
) -> pd.DataFrame:
    df = df.sort_values(score_col, ascending=False).copy()

    # ticket-specific floor rules
    if score_col == "score_safe":
        df = df[(df["minutes_proj"] >= 22) & (df["confidence_tier"].isin(["high_conf", "medium_conf"]))].copy()
    elif score_col == "score_balanced":
        df = df[(df["minutes_proj"] >= 18) & (df["confidence_tier"].isin(["high_conf", "medium_conf"]))].copy()
    elif score_col == "score_lotto":
        df = df[(df["minutes_proj"] >= 10)].copy()

    selected = []
    used_players = set()
    used_teams = set()
    used_stats = set()
    used_games = set()

    for _, row in df.iterrows():
        if len(selected) >= max_legs:
            break

        player = row["player"]
        team = row["team"]
        stat = row["stat"]
        game = "_".join(sorted([str(row["team"]), str(row["opp"])]))

        if player in used_players:
            continue

        if team in used_teams:
            continue

        # prefer diversity while filling toward min_legs
        if stat in used_stats and len(used_stats) < min_legs:
            continue

        if game in used_games:
            continue

        selected.append(row)
        used_players.add(player)
        used_teams.add(team)
        used_stats.add(stat)
        used_games.add(game)

    # relax stat diversity first
    if len(selected) < min_legs:
        for _, row in df.iterrows():
            if len(selected) >= min_legs:
                break

            player = row["player"]
            team = row["team"]
            game = "_".join(sorted([str(row["team"]), str(row["opp"])]))

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

    # relax game uniqueness
    if len(selected) < min_legs:
        for _, row in df.iterrows():
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

    # final fallback: no duplicate players only
    if len(selected) < min_legs:
        for _, row in df.iterrows():
            if len(selected) >= min_legs:
                break

            player = row["player"]
            if player in used_players:
                continue

            selected.append(row)
            used_players.add(player)

    if not selected:
        return pd.DataFrame(columns=df.columns)

    return pd.DataFrame(selected).reset_index(drop=True)


def build_all_tickets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df.copy()
    df = df[df["is_eligible"] == 1].copy()

    if df.empty:
        raise ValueError("No eligible players for tickets")

    ranked = rank_projection_pool(df)

    safe = build_ticket(
        ranked,
        "score_safe",
        min_legs=3,
        max_legs=5,
    )
    balanced = build_ticket(
        ranked,
        "score_balanced",
        min_legs=5,
        max_legs=7,
    )
    lotto = build_ticket(
        ranked,
        "score_lotto",
        min_legs=10,
        max_legs=20,
    )

    summary = pd.DataFrame([
        {
            "ticket_name": "safe",
            "n_legs": len(safe),
            "avg_pred_mean": safe["pred_mean"].mean() if not safe.empty else 0.0,
            "avg_minutes_proj": safe["minutes_proj"].mean() if not safe.empty else 0.0,
        },
        {
            "ticket_name": "balanced",
            "n_legs": len(balanced),
            "avg_pred_mean": balanced["pred_mean"].mean() if not balanced.empty else 0.0,
            "avg_minutes_proj": balanced["minutes_proj"].mean() if not balanced.empty else 0.0,
        },
        {
            "ticket_name": "lotto",
            "n_legs": len(lotto),
            "avg_pred_mean": lotto["pred_mean"].mean() if not lotto.empty else 0.0,
            "avg_minutes_proj": lotto["minutes_proj"].mean() if not lotto.empty else 0.0,
        },
    ])

    return {
        "ranked_pool": ranked,
        "safe": safe,
        "balanced": balanced,
        "lotto": lotto,
        "summary": summary,
    }