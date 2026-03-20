# model_training/utils/team_codes.py
from __future__ import annotations

TEAM_MAP: dict[str, str] = {
    # legacy / alternate abbrevs -> canonical
    "NJN": "BKN",
    "CHO": "CHA",
    # add more only if you actually encounter them
}

def norm_team(x: str) -> str:
    t = str(x).upper().strip()
    return TEAM_MAP.get(t, t)
