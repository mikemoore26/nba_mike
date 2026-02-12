from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd

@dataclass
class RetrainPolicy:
    fg3a_every_days: int = 1
    rate_every_days: int = 7

def load_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def save_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))

def _days_since(last_date: str | None, as_of: pd.Timestamp) -> int | None:
    if not last_date:
        return None
    return (as_of.normalize() - pd.Timestamp(last_date).normalize()).days

def should_retrain_fg3a(meta: dict, as_of: pd.Timestamp, policy: RetrainPolicy) -> bool:
    d = _days_since(meta.get("fg3a_last_train"), as_of)
    return (d is None) or (d >= policy.fg3a_every_days)

def should_retrain_rate(meta: dict, as_of: pd.Timestamp, policy: RetrainPolicy) -> bool:
    d = _days_since(meta.get("rate_last_train"), as_of)
    return (d is None) or (d >= policy.rate_every_days)
