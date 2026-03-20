from __future__ import annotations

import math

import numpy as np
from scipy.stats import nbinom, poisson


def nb_params_from_mu_alpha(mu: float, alpha: float) -> tuple[float, float]:
    """
    Convert NB2 parameterization:
        Var(Y|X) = mu + alpha * mu^2

    into scipy nbinom params:
        n = r
        p = r / (r + mu)
    """
    mu = max(float(mu), 1e-9)
    alpha = max(float(alpha), 0.0)

    if alpha <= 1e-12:
        # effectively Poisson
        r = 1e12
        p = r / (r + mu)
        return r, p

    r = 1.0 / alpha
    p = r / (r + mu)
    return r, p


def prob_over_discrete(mu: float, line: float, dist_name: str = "nbinom", alpha: float = 0.0) -> float:
    """
    For sportsbook-style line like 5.5:
        P(X > 5.5) = P(X >= 6)

    For integer-valued line like 5:
        this still treats 'over 5' as P(X >= 6)
    """
    mu = max(float(mu), 0.0)
    threshold = math.floor(float(line)) + 1

    if dist_name == "poisson" or alpha <= 1e-12:
        return float(poisson.sf(threshold - 1, mu))

    if dist_name == "nbinom":
        r, p = nb_params_from_mu_alpha(mu, alpha)
        return float(nbinom.sf(threshold - 1, r, p))

    raise ValueError(f"Unsupported dist_name: {dist_name}")


def prob_under_discrete(mu: float, line: float, dist_name: str = "nbinom", alpha: float = 0.0) -> float:
    """
    For sportsbook-style line like 5.5:
        P(X < 5.5) = P(X <= 5)
    """
    mu = max(float(mu), 0.0)
    threshold = math.floor(float(line))

    if dist_name == "poisson" or alpha <= 1e-12:
        return float(poisson.cdf(threshold, mu))

    if dist_name == "nbinom":
        r, p = nb_params_from_mu_alpha(mu, alpha)
        return float(nbinom.cdf(threshold, r, p))

    raise ValueError(f"Unsupported dist_name: {dist_name}")