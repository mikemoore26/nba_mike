import pandas as pd
import numpy as np

def expected_fg3a_ceiling(ph: pd.DataFrame, recent_n: int = 5) -> float:
    """
    Better than mean(last N):
    - base = trailing mean (stable)
    - ceiling = recent 80th percentile (captures spike behavior)
    - blend them so stars can pop
    - cap at recent 95th percentile
    """
    tail = ph["fg3a"].tail(max(10, recent_n * 2)).dropna()
    if tail.empty:
        return 0.0

    base = float(tail.tail(recent_n).mean())
    ceiling = float(tail.quantile(0.80))
    exp = 0.65 * base + 0.35 * ceiling

    cap = float(tail.quantile(0.95))
    return float(np.clip(exp, 0, cap))

