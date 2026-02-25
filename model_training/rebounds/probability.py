# model_training/rebounds/probability.py
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom


def _nbinom_params_from_mean_alpha(mu: np.ndarray, alpha: float):
    """
    Var = mu + alpha * mu^2
    Convert to nbinom(n, p) with:
      mean = n*(1-p)/p
      var  = n*(1-p)/p^2
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-9, None)

    alpha = float(max(alpha, 0.0))
    var = mu + alpha * (mu ** 2)
    var = np.clip(var, mu + 1e-9, None)

    p = mu / var
    p = np.clip(p, 1e-6, 1 - 1e-6)
    n = mu * p / (1 - p)
    n = np.clip(n, 1e-6, None)

    return n, p


def prob_ge_k_nbinom(mu, k, *, alpha: float = 0.25) -> np.ndarray:
    """
    Returns P(X >= k) for NegBin with mean=mu and Var=mu+alpha*mu^2.
    k can be scalar or array-like ints.
    """
    mu = np.asarray(mu, dtype=float)
    k = np.asarray(k)

    n, p = _nbinom_params_from_mean_alpha(mu, alpha=alpha)

    # P(X >= k) = 1 - CDF(k-1)
    km1 = np.clip(k.astype(int) - 1, -1, None)
    out = np.where(km1 < 0, 1.0, 1.0 - nbinom.cdf(km1, n, p))
    return np.clip(out, 0.0, 1.0)
