# model_training/ticketing/ticket_recipes.py
from __future__ import annotations
import pandas as pd

from .scoring import add_leg_scores
from .constraints import TicketConstraints, StatMix
from .optimizer_greedy import select_ticket_greedy

def filter_min_p(legs: pd.DataFrame, min_p_by_stat: dict[str, float]) -> pd.DataFrame:
    df = legs.copy()
    keep = []
    for _, r in df.iterrows():
        thr = min_p_by_stat.get(r["stat"], 0.0)
        keep.append(r["p_hit"] >= thr)
    return df[pd.Series(keep, index=df.index)]

def build_three_tickets(legs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tickets = {}

    # -----------------------
    # A) SAFE / HIGH PROB
    # -----------------------
    legs_a = filter_min_p(legs, {"fg3m": 0.60, "reb": 0.58, "pts": 0.57})
    legs_a = add_leg_scores(legs_a, w_p=1.0, w_margin=0.10)
    tickets["A_safe"] = select_ticket_greedy(
        legs_a,
        constraints=TicketConstraints(max_legs=6, max_per_game=1, max_per_player=1),
        statmix=None,
        game_penalty=0.06,
        team_penalty=0.03,
    )

    # -----------------------
    # B) BALANCED MIX
    # enforce at least 2 stats represented and min counts
    # -----------------------
    legs_b = filter_min_p(legs, {"fg3m": 0.58, "reb": 0.56, "pts": 0.55})
    legs_b = add_leg_scores(legs_b, w_p=1.0, w_margin=0.15)
    tickets["B_balanced"] = select_ticket_greedy(
        legs_b,
        constraints=TicketConstraints(max_legs=6, max_per_game=2, max_per_player=1),
        statmix=StatMix(
            min_by_stat={"fg3m": 2, "reb": 2, "pts": 1},   # tweak
            max_by_stat={"fg3m": 3, "reb": 3, "pts": 2},
        ),
        game_penalty=0.04,
        team_penalty=0.02,
    )

    # -----------------------
    # C) HIGH CEILING
    # allow a bit more stacking, favor margin more
    # -----------------------
    legs_c = filter_min_p(legs, {"fg3m": 0.55, "reb": 0.53, "pts": 0.52})
    legs_c = add_leg_scores(legs_c, w_p=0.85, w_margin=0.25)
    tickets["C_ceiling"] = select_ticket_greedy(
        legs_c,
        constraints=TicketConstraints(max_legs=6, max_per_game=2, max_per_player=1),
        statmix=None,
        game_penalty=0.02,
        team_penalty=0.01,
    )

    return tickets