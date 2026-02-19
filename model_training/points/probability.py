# model_training/points/probability.py
from __future__ import annotations

import numpy as np
import pandas as pd
from math import erf, sqrt


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF using erf (no SciPy dependency).
    """
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def prob_pts_ge_line_normal(mu_pts: np.ndarray, sd_pts: np.ndarray, line: float, continuity: bool = True) -> np.ndarray:
    """
    Fast probability approximation:
      P(PTS >= line) ~ Normal(mu_pts, sd_pts^2)

    continuity=True uses continuity correction:
      P(X >= L) ≈ P(N >= L - 0.5)
    """
    mu = np.asarray(mu_pts, dtype=float)
    sd = np.asarray(sd_pts, dtype=float)

    # handle degenerate sd
    sd_safe = np.where(sd <= 1e-9, 1e-9, sd)

    thresh = float(line) - (0.5 if continuity else 0.0)
    z = (mu - thresh) / sd_safe
    return _norm_cdf(z)


# -----------------------------------------------------------------------------
# Discrete distribution methods (accurate tails)
# -----------------------------------------------------------------------------
def _binom_pmf_fft(n: int, p: float) -> np.ndarray:
    """
    Binomial PMF for k=0..n. Uses stable recursion (no SciPy).
    """
    n = int(max(0, n))
    p = float(np.clip(p, 0.0, 1.0))

    pmf = np.zeros(n + 1, dtype=float)
    if n == 0:
        pmf[0] = 1.0
        return pmf

    # edge cases
    if p <= 0.0:
        pmf[0] = 1.0
        return pmf
    if p >= 1.0:
        pmf[n] = 1.0
        return pmf

    # start at k=0
    pmf[0] = (1.0 - p) ** n
    ratio = p / (1.0 - p)
    # recursion: pmf[k] = pmf[k-1] * (n-k+1)/k * p/(1-p)
    for k in range(1, n + 1):
        pmf[k] = pmf[k - 1] * (n - k + 1) / k * ratio

    # numerical normalization
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def _scale_support(pmf: np.ndarray, multiplier: int) -> np.ndarray:
    """
    Convert a PMF over counts to a PMF over points by spacing support.

    Example:
      FG3M pmf over k -> points = 3*k
      returns array where out[3*k] = pmf[k]
    """
    m = int(multiplier)
    if m <= 0:
        raise ValueError("multiplier must be >= 1")

    out = np.zeros((len(pmf) - 1) * m + 1, dtype=float)
    out[::m] = pmf
    return out


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Convolution of two PMFs.
    Uses FFT for speed when sizes are large.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    n = len(a) + len(b) - 1
    # heuristic: FFT only when big enough
    if n >= 512:
        fa = np.fft.rfft(a, n)
        fb = np.fft.rfft(b, n)
        out = np.fft.irfft(fa * fb, n)
        out = np.clip(out, 0, None)
    else:
        out = np.convolve(a, b)

    s = out.sum()
    if s > 0:
        out /= s
    return out


def pts_pmf_from_components(
    *,
    n2: float,
    p2: float,
    n3: float,
    p3: float,
    nft: float,
    pft: float,
    n_round: str = "nearest",
    max_n_cap: int | None = None,
) -> np.ndarray:
    """
    Build a discrete PMF for total points as:
      PTS = 2*FG2M + 3*FG3M + 1*FTM
    where each make component is Binomial(n, p).

    Inputs can be floats (model expected attempts).
    We convert attempts to integer n by rounding:
      - n_round="nearest" (default) -> round()
      - n_round="floor"
      - n_round="ceil"

    max_n_cap: optional cap for each n to keep PMF size bounded
              (useful when models sometimes spit huge attempts).
    """
    def to_int_n(x: float) -> int:
        x = float(np.clip(x, 0, None))
        if n_round == "floor":
            ni = int(np.floor(x))
        elif n_round == "ceil":
            ni = int(np.ceil(x))
        else:
            ni = int(np.rint(x))  # nearest
        if max_n_cap is not None:
            ni = min(ni, int(max_n_cap))
        return ni

    n2i = to_int_n(n2)
    n3i = to_int_n(n3)
    nfti = to_int_n(nft)

    pmf2m = _binom_pmf_fft(n2i, p2)
    pmf3m = _binom_pmf_fft(n3i, p3)
    pmfftm = _binom_pmf_fft(nfti, pft)

    pmf2pts = _scale_support(pmf2m, 2)
    pmf3pts = _scale_support(pmf3m, 3)

    # total = 2pts + 3pts + ft
    pmf = _convolve(_convolve(pmf2pts, pmf3pts), pmfftm)
    return pmf


def prob_pts_ge_line_discrete(
    *,
    pred_fg2a: float,
    pred_fg2_rate: float,
    pred_fg3a: float,
    pred_fg3_rate: float,
    pred_fta: float,
    pred_ft_rate: float,
    line: float,
    n_round: str = "nearest",
    max_n_cap: int | None = None,
) -> float:
    """
    Accurate discrete probability:
      P(PTS >= line)
    using Binomial -> convolution PMF.

    Note:
      This rounds attempts to integers. That is the main approximation.
    """
    pmf = pts_pmf_from_components(
        n2=pred_fg2a, p2=pred_fg2_rate,
        n3=pred_fg3a, p3=pred_fg3_rate,
        nft=pred_fta, pft=pred_ft_rate,
        n_round=n_round,
        max_n_cap=max_n_cap,
    )
    L = int(np.ceil(float(line)))
    if L <= 0:
        return 1.0
    if L >= len(pmf):
        return 0.0
    return float(pmf[L:].sum())


# -----------------------------------------------------------------------------
# Batch helpers for DataFrames
# -----------------------------------------------------------------------------
def add_prob_pts_ge_line(
    df: pd.DataFrame,
    *,
    line_col: str | None = None,
    line_value: float | None = None,
    out_col: str = "p_pts_ge_line",
    method: str = "normal",
    continuity: bool = True,
    n_round: str = "nearest",
    max_n_cap: int | None = None,
) -> pd.DataFrame:
    """
    Adds a probability column to a predictions DF.

    Requires columns:
      pred_pts, sd_pts  (for method="normal")
      pred_fg2a, pred_fg2_rate, pred_fg3a, pred_fg3_rate, pred_fta, pred_ft_rate (for discrete)

    Provide either:
      - line_col: name of the column containing the line per row
      - line_value: constant line for all rows
    """
    if (line_col is None) == (line_value is None):
        raise ValueError("Provide exactly one of line_col or line_value.")

    out = df.copy()

    if line_col is not None:
        lines = out[line_col].astype(float).to_numpy()
    else:
        lines = np.full(len(out), float(line_value), dtype=float)

    if method == "normal":
        if "pred_pts" not in out.columns or "sd_pts" not in out.columns:
            raise ValueError("method='normal' requires columns: pred_pts, sd_pts")
        probs = np.empty(len(out), dtype=float)
        for i, L in enumerate(lines):
            probs[i] = prob_pts_ge_line_normal(out["pred_pts"].to_numpy()[i], out["sd_pts"].to_numpy()[i], L, continuity=continuity)
        out[out_col] = np.clip(probs, 0, 1)
        return out

    if method == "discrete":
        req = ["pred_fg2a","pred_fg2_rate","pred_fg3a","pred_fg3_rate","pred_fta","pred_ft_rate"]
        missing = [c for c in req if c not in out.columns]
        if missing:
            raise ValueError(f"method='discrete' missing columns: {missing}")

        probs = np.empty(len(out), dtype=float)
        for i, L in enumerate(lines):
            probs[i] = prob_pts_ge_line_discrete(
                pred_fg2a=float(out["pred_fg2a"].iloc[i]),
                pred_fg2_rate=float(out["pred_fg2_rate"].iloc[i]),
                pred_fg3a=float(out["pred_fg3a"].iloc[i]),
                pred_fg3_rate=float(out["pred_fg3_rate"].iloc[i]),
                pred_fta=float(out["pred_fta"].iloc[i]),
                pred_ft_rate=float(out["pred_ft_rate"].iloc[i]),
                line=float(L),
                n_round=n_round,
                max_n_cap=max_n_cap,
            )
        out[out_col] = np.clip(probs, 0, 1)
        return out

    raise ValueError("method must be one of: 'normal', 'discrete'")
