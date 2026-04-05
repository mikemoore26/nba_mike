from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denom = np.where(np.asarray(y_true) == 0, np.nan, np.asarray(y_true))
    out = np.abs((np.asarray(y_true) - y_pred) / denom)
    return float(np.nanmean(out))


def fit_dispersion_alpha_mom(y_true: pd.Series, mu_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(mu_pred, dtype=float), 1e-6, None)

    numer = np.mean((y - mu) ** 2 - mu)
    denom = np.mean(mu ** 2)

    if denom <= 0:
        return 0.0

    alpha = numer / denom
    return float(max(alpha, 0.0))


def make_fg3a_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    Baseline attempt model:
      1) fg3a_rolling_5
      2) player_fg3a_season_avg
      3) fga_rolling_5 * 0.35
      4) league-ish fallback
    """
    baseline = df["fg3a_rolling_5"].copy()

    fallback_1 = df["player_fg3a_season_avg"] if "player_fg3a_season_avg" in df.columns else np.nan
    fallback_2 = df["fga_rolling_5"] * 0.35 if "fga_rolling_5" in df.columns else np.nan
    fallback_3 = pd.Series(3.5, index=df.index, dtype=float)

    baseline = baseline.fillna(fallback_1)
    baseline = baseline.fillna(fallback_2)
    baseline = baseline.fillna(fallback_3)

    if "expected_fg3a_ceiling" in df.columns and "fg3a_spike_ratio" in df.columns:
        spike_boost = (
            0.85 * baseline
            + 0.15 * df["expected_fg3a_ceiling"].fillna(baseline)
        )
        baseline = np.where(df["fg3a_spike_ratio"].fillna(1.0) > 1.15, spike_boost, baseline)

    return np.clip(np.asarray(baseline, dtype=float), 0.0, None)


def make_fg3_rate_baseline(df: pd.DataFrame, prior_att: float = 80.0) -> np.ndarray:
    """
    Bayesian-shrunk baseline make-rate model:
      1) player season makes/attempts with league prior
      2) fg3_pct_rolling_10 blend
      3) player_fg3_pct_season
      4) league fallback
    """
    if {"fg3m", "fg3a"}.issubset(df.columns):
        league_rate = float(df["fg3m"].sum() / max(df["fg3a"].sum(), 1))
        if not np.isfinite(league_rate) or league_rate <= 0:
            league_rate = 0.36
    else:
        league_rate = 0.36

    prior_made = prior_att * league_rate

    made = df["player_fg3m_season_sum"] if "player_fg3m_season_sum" in df.columns else pd.Series(np.nan, index=df.index)
    att = df["player_fg3a_season_sum"] if "player_fg3a_season_sum" in df.columns else pd.Series(np.nan, index=df.index)

    shrunk = (made.fillna(0.0) + prior_made) / (att.fillna(0.0) + prior_att)

    recent = df["fg3_pct_rolling_10"] if "fg3_pct_rolling_10" in df.columns else pd.Series(np.nan, index=df.index)
    season = df["player_fg3_pct_season"] if "player_fg3_pct_season" in df.columns else pd.Series(np.nan, index=df.index)

    rate = shrunk.copy()
    rate = rate.where(rate.notna(), season)
    rate = rate.where(rate.notna(), recent)

    recent_weight = np.clip(att.fillna(0.0) / 200.0, 0.0, 1.0) * 0.20 + 0.10
    base_weight = 0.80 - (recent_weight - 0.10)

    recent_filled = recent.fillna(rate).fillna(league_rate)
    rate = (base_weight * rate.fillna(league_rate)) + (recent_weight * recent_filled)

    rate = rate.replace([np.inf, -np.inf], np.nan).fillna(league_rate)
    return np.clip(np.asarray(rate, dtype=float), 0.0, 1.0)


def make_fg3a_model() -> HistGradientBoostingRegressor:
    """
    Residual model for FG3A:
      target = fg3a - fg3a_baseline
    Must support negative targets, so squared_error is required.
    """
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.75,
        random_state=42,
    )


def make_fg3a_dispersion_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.03,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


@dataclass
class BinomialGLMRateModel:
    imputer: SimpleImputer
    result_: Any
    feature_names_: list[str]

    @classmethod
    def fit_from_df(cls, df: pd.DataFrame, feature_cols: list[str]) -> "BinomialGLMRateModel":
        tr = df[df["fg3a"] > 0].copy()

        X = tr[feature_cols].copy()
        y = (tr["fg3m"] / tr["fg3a"]).astype(float)
        n = tr["fg3a"].astype(float)

        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        X_imp = sm.add_constant(X_imp, has_constant="add")

        glm = sm.GLM(y, X_imp, family=sm.families.Binomial(), var_weights=n)
        res = glm.fit()

        return cls(
            imputer=imp,
            result_=res,
            feature_names_=feature_cols,
        )

    def predict_p(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names_].copy()
        X_imp = self.imputer.transform(X)
        X_imp = sm.add_constant(X_imp, has_constant="add")
        p = self.result_.predict(X_imp)
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


def make_fg3_rate_model(train_df: pd.DataFrame, feature_cols: list[str]) -> BinomialGLMRateModel:
    return BinomialGLMRateModel.fit_from_df(train_df, feature_cols)


def build_attempt_overdispersion_target(y_true: np.ndarray, mu_pred: np.ndarray) -> np.ndarray:
    """
    Moment-style row target for alpha in Var(A) = mu + alpha * mu^2.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(mu_pred, dtype=float), 1e-6, None)

    alpha_row = ((y - mu) ** 2 - mu) / (mu ** 2)
    alpha_row = np.where(np.isfinite(alpha_row), alpha_row, 0.0)
    alpha_row = np.clip(alpha_row, 0.0, 5.0)
    return alpha_row


def predict_attempt_variance(mu_att: np.ndarray, alpha_att: np.ndarray) -> np.ndarray:
    mu_att = np.clip(np.asarray(mu_att, dtype=float), 1e-6, None)
    alpha_att = np.clip(np.asarray(alpha_att, dtype=float), 0.0, None)
    return mu_att + alpha_att * (mu_att ** 2)


def estimate_rate_strength(
    player_fg3a_season_sum: pd.Series | np.ndarray,
    prior_att: float = 80.0,
) -> np.ndarray:
    att = np.asarray(player_fg3a_season_sum, dtype=float)
    att = np.where(np.isfinite(att), att, 0.0)
    return np.clip(att + prior_att, 1.0, None)


def predict_fg3m_variance(
    mu_att: np.ndarray,
    p_rate: np.ndarray,
    att_alpha: np.ndarray,
    rate_strength: np.ndarray,
) -> np.ndarray:
    """
    Approximate total variance for makes:
      Var(M) ≈ E[ A p (1-p) ] + Var(A) * p^2 + (E[A]^2) * Var(p)

    Var(p) is approximated with Beta posterior variance using rate_strength.
    """
    mu_att = np.clip(np.asarray(mu_att, dtype=float), 1e-6, None)
    p_rate = np.clip(np.asarray(p_rate, dtype=float), 1e-6, 1.0 - 1e-6)
    att_alpha = np.clip(np.asarray(att_alpha, dtype=float), 0.0, None)
    rate_strength = np.clip(np.asarray(rate_strength, dtype=float), 1.0, None)

    var_att = predict_attempt_variance(mu_att, att_alpha)
    var_p = (p_rate * (1.0 - p_rate)) / (rate_strength + 1.0)

    shot_noise = mu_att * p_rate * (1.0 - p_rate)
    att_noise = var_att * (p_rate ** 2)
    rate_noise = (mu_att ** 2) * var_p

    var_m = shot_noise + att_noise + rate_noise
    mean_m = mu_att * p_rate
    var_m = np.maximum(var_m, mean_m + 1e-6)
    return var_m


def variance_to_nbinom_alpha(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), 1e-6, None)
    var = np.maximum(np.asarray(var, dtype=float), mu + 1e-6)
    alpha = (var - mu) / (mu ** 2)
    alpha = np.where(np.isfinite(alpha), alpha, 0.0)
    return np.clip(alpha, 0.0, 5.0)


def blend_fg3m_mean(
    baseline_mean: np.ndarray,
    model_mean: np.ndarray,
    model_weight: float,
) -> np.ndarray:
    w = float(np.clip(model_weight, 0.0, 1.0))
    baseline_mean = np.asarray(baseline_mean, dtype=float)
    model_mean = np.asarray(model_mean, dtype=float)
    return np.clip((1.0 - w) * baseline_mean + w * model_mean, 0.0, None)


def choose_fg3m_blend_weight(
    y_true: pd.Series | np.ndarray,
    baseline_mean: np.ndarray,
    model_mean: np.ndarray,
    candidate_weights: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    if candidate_weights is None:
        candidate_weights = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.85, 1.0]

    y = np.asarray(y_true, dtype=float)
    baseline_mean = np.asarray(baseline_mean, dtype=float)
    model_mean = np.asarray(model_mean, dtype=float)

    scores: dict[str, float] = {}
    best_w = 0.0
    best_mae = np.inf

    for w in candidate_weights:
        pred = blend_fg3m_mean(baseline_mean, model_mean, w)
        mae = float(np.mean(np.abs(y - pred)))
        scores[f"blend_mae_w_{w:.2f}"] = mae
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)

    return best_w, scores


def prob_ge_k(mu: np.ndarray, alpha: np.ndarray, k: int) -> np.ndarray:
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)
    alpha = np.clip(np.asarray(alpha, dtype=float), 0.0, None)

    out = np.zeros_like(mu, dtype=float)

    poisson_mask = alpha <= 1e-12
    if np.any(poisson_mask):
        out[poisson_mask] = 1.0 - poisson.cdf(k - 1, mu[poisson_mask])

    nb_mask = ~poisson_mask
    if np.any(nb_mask):
        a = alpha[nb_mask]
        m = mu[nb_mask]
        n = 1.0 / np.clip(a, 1e-12, None)
        p = n / (n + m)
        out[nb_mask] = 1.0 - nbinom.cdf(k - 1, n, p)

    return np.clip(out, 0.0, 1.0)


def fit_threshold_isotonic_calibrators(
    y_true: pd.Series | np.ndarray,
    mu_pred: np.ndarray,
    alpha_pred: np.ndarray,
    thresholds: list[int],
) -> dict[int, IsotonicRegression]:
    y = np.asarray(y_true, dtype=float)
    calibrators: dict[int, IsotonicRegression] = {}

    for k in thresholds:
        raw_p = prob_ge_k(mu_pred, alpha_pred, k)
        target = (y >= k).astype(int)

        if np.unique(target).size < 2:
            continue

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_p, target)
        calibrators[k] = iso

    return calibrators


def apply_threshold_calibrators(
    mu_pred: np.ndarray,
    alpha_pred: np.ndarray,
    calibrators: dict[int, IsotonicRegression],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for k, iso in calibrators.items():
        raw_p = prob_ge_k(mu_pred, alpha_pred, k)
        out[k] = np.clip(iso.predict(raw_p), 0.0, 1.0)
    return out


def calibration_report(
    y_true: pd.Series | np.ndarray,
    mu_pred: np.ndarray,
    alpha_pred: np.ndarray,
    calibrators: dict[int, IsotonicRegression],
    thresholds: list[int],
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    metrics: dict[str, float] = {}

    for k in thresholds:
        raw_p = prob_ge_k(mu_pred, alpha_pred, k)
        target = (y >= k).astype(int)

        if np.unique(target).size < 2:
            metrics[f"brier_ge_{k}_raw"] = np.nan
            metrics[f"brier_ge_{k}_cal"] = np.nan
            continue

        metrics[f"brier_ge_{k}_raw"] = float(brier_score_loss(target, raw_p))

        if k in calibrators:
            cal_p = np.clip(calibrators[k].predict(raw_p), 0.0, 1.0)
            metrics[f"brier_ge_{k}_cal"] = float(brier_score_loss(target, cal_p))
        else:
            metrics[f"brier_ge_{k}_cal"] = np.nan

    return metrics