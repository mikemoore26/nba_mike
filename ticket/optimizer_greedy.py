# model_training/ticketing/optimizer_greedy.py
from __future__ import annotations
import pandas as pd
from .constraints import TicketConstraints, StatMix

def select_ticket_greedy(
    legs_scored: pd.DataFrame,
    *,
    constraints: TicketConstraints,
    statmix: StatMix | None = None,
    game_penalty: float = 0.04,
    team_penalty: float = 0.02,
) -> pd.DataFrame:
    df = legs_scored.copy().reset_index(drop=True)

    # base ranking
    df = df.sort_values(["score", "p_hit"], ascending=False).reset_index(drop=True)

    picked = []
    game_ct, player_ct, team_ct, stat_ct = {}, {}, {}, {}

    min_by_stat = (statmix.min_by_stat if statmix and statmix.min_by_stat else {})
    max_by_stat = (statmix.max_by_stat if statmix and statmix.max_by_stat else {})

    for _, r in df.iterrows():
        if len(picked) >= constraints.max_legs:
            break

        gid = r["game_id"]
        ply = r["player"]
        team = r["team"]
        stat = r["stat"]

        # hard constraints
        if game_ct.get(gid, 0) >= constraints.max_per_game:
            continue
        if player_ct.get(ply, 0) >= constraints.max_per_player:
            continue
        if not constraints.allow_same_player_multi_stat and player_ct.get(ply, 0) >= 1:
            continue
        if stat in max_by_stat and stat_ct.get(stat, 0) >= max_by_stat[stat]:
            continue

        # apply soft penalty BEFORE deciding (dynamic correlation control)
        score_adj = r["score"]
        score_adj -= game_penalty * game_ct.get(gid, 0)
        score_adj -= team_penalty * team_ct.get(team, 0)

        # optional: reject if penalty makes it too weak
        # (comment out if you want always-best remaining)
        # if score_adj < some_threshold: continue

        rr = r.copy()
        rr["score_adj"] = score_adj
        picked.append(rr)

        # update counts
        game_ct[gid] = game_ct.get(gid, 0) + 1
        player_ct[ply] = player_ct.get(ply, 0) + 1
        team_ct[team] = team_ct.get(team, 0) + 1
        stat_ct[stat] = stat_ct.get(stat, 0) + 1

    ticket = pd.DataFrame(picked)

    # enforce minimum stat requirements (post-fix by swapping if needed)
    # Keep v1 simple: warn if unmet.
    for s, mn in min_by_stat.items():
        if (ticket["stat"] == s).sum() < mn:
            # don't crash; just return ticket + let caller handle
            ticket.attrs[f"warn_min_{s}"] = f"Ticket has {(ticket['stat']==s).sum()} < {mn}"

    return ticket