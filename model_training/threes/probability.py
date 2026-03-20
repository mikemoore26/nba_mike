import pandas as pd
import numpy as np
from scipy.stats import poisson

# ----------------------------
# Probability helpers
# ----------------------------
def prob_ge_k(mu: np.ndarray, k: int) -> np.ndarray:
    """P(X >= k) for Poisson(mu)."""
    mu = np.clip(mu, 0, None)
    return 1.0 - poisson.cdf(k - 1, mu)


def add_prob_ge_k(out: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    out = out.copy()
    mu = np.clip(out["pred_fg3a"].to_numpy() * out["pred_rate"].to_numpy(), 0, None)
    out[f"p_ge_{k}"] = prob_ge_k(mu, k)
    return out
