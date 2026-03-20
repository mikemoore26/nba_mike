from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def american_to_implied_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def expected_value_per_unit(*, win_prob: float, american_odds: float) -> float:
    """
    EV per 1 unit staked.
    Positive means profitable in expectation.
    """
    win_prob = float(win_prob)
    lose_prob = 1.0 - win_prob
    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = decimal_odds - 1.0

    return (win_prob * profit_if_win) - (lose_prob * 1.0)


def probability_over_discrete(
    *,
    mean: float,
    line: float,
    dist_name: str,
    dispersion: float | None = None,
) -> float:
    """
    For half-lines like 27.5, 'over' means X >= 28.
    """
    mean = max(float(mean), 1e-9)
    line = float(line)
    k = math.floor(line) + 1

    dist_name = str(dist_name).lower()

    if dist_name == "poisson":
        return float(1.0 - poisson.cdf(k - 1, mean))

    if dist_name == "nbinom":
        alpha = 0.0 if dispersion is None or pd.isna(dispersion) else max(float(dispersion), 0.0)

        if alpha <= 1e-12:
            return float(1.0 - poisson.cdf(k - 1, mean))

        # NB parameterization:
        # var = mu + alpha * mu^2
        # r = 1 / alpha
        # p = r / (r + mu)
        r = 1.0 / alpha
        p = r / (r + mean)

        return float(1.0 - nbinom.cdf(k - 1, r, p))

    raise ValueError(f"Unsupported dist_name: {dist_name}")


def probability_under_discrete(
    *,
    mean: float,
    line: float,
    dist_name: str,
    dispersion: float | None = None,
) -> float:
    return 1.0 - probability_over_discrete(
        mean=mean,
        line=line,
        dist_name=dist_name,
        dispersion=dispersion,
    )


def compute_side_probability(
    *,
    mean: float,
    line: float,
    side: str,
    dist_name: str,
    dispersion: float | None = None,
) -> float:
    side = str(side).lower()
    if side == "over":
        return probability_over_discrete(
            mean=mean,
            line=line,
            dist_name=dist_name,
            dispersion=dispersion,
        )
    if side == "under":
        return probability_under_discrete(
            mean=mean,
            line=line,
            dist_name=dist_name,
            dispersion=dispersion,
        )
    raise ValueError(f"Unsupported side: {side}")


def add_market_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["model_prob"] = out.apply(
        lambda r: compute_side_probability(
            mean=r["pred_mean"],
            line=r["line"],
            side=r["side"],
            dist_name=r["dist_name"],
            dispersion=r.get("dispersion", np.nan),
        ),
        axis=1,
    )

    out["implied_prob"] = out["american_odds"].apply(american_to_implied_prob)
    out["edge"] = out["model_prob"] - out["implied_prob"]

    out["ev_per_unit"] = out.apply(
        lambda r: expected_value_per_unit(
            win_prob=r["model_prob"],
            american_odds=r["american_odds"],
        ),
        axis=1,
    )

    return out