# model_training/ticketing/schema.py
from __future__ import annotations
from dataclasses import dataclass

LEG_REQUIRED = {
    "date", "game_id", "matchup_key",
    "player", "team", "opp", "is_home",
    "stat", "side", "line",
    "pred_mean", "p_hit",
}

def require_leg_cols(df, req=LEG_REQUIRED, name="legs"):
    missing = sorted(req - set(df.columns))
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")