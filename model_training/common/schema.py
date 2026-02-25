# model_training/common/schema.py
from __future__ import annotations

ID_COLS = [
    "game_date",
    "player",
    "team",
    "opp",
    "is_home",
]

# For joining, debugging, sorting
SORT_COLS = ["game_date", "team", "player"]

# Minimal “you can’t predict without these”
PRED_REQUIRED_COLS = set(ID_COLS)

def assert_has_cols(df, cols: set[str] | list[str], *, name: str = "df") -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")