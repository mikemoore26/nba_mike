# model_training/ticketing/constraints.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TicketConstraints:
    max_legs: int = 6
    max_per_game: int = 2
    max_per_player: int = 1
    allow_same_player_multi_stat: bool = False

@dataclass(frozen=True)
class StatMix:
    # min/max legs per stat (optional)
    min_by_stat: dict[str, int] | None = None
    max_by_stat: dict[str, int] | None = None