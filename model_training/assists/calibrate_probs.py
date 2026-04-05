from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from model_training.config import ASSISTS_MODEL_DIR


def _load_artifacts(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    with open(model_dir / "ast_artifacts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_validation_predictions(model_dir: str | Path) -> pd.DataFrame:
    model_dir = Path(model_dir)
    path = model_dir / "ast_validation_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validation predictions file: {path}")

    df = pd.read_csv(path)
    required = ["game_date", "player", "team", "opp", "ast", "model_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Validation predictions missing required columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["ast"] = pd.to_numeric(df["ast"], errors="coerce")
    df["model_pred"] = pd.to_numeric(df["model_pred"], errors="coerce")
    df = df.dropna(subset=["ast", "model_pred"]).copy()

    df["ast"] = df["ast"].clip(lower=0)
    df["model_pred"] = df["model_pred"].clip(lower=0)
    return df


def _prob_ge_k(mu: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """
    P(X >= k) for Poisson / NB parameterization used elsewhere in project.

    NB parameterization:
      var = mu + alpha * mu^2
      n = 1 / alpha
      p = n / (n + mu)
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 0.0, None)

    if k <= 0:
        return np.ones_like(mu, dtype=float)

    if alpha <= 1e-12:
        return 1.0 - poisson.cdf(k - 1, mu)

    n = 1.0 / alpha
    p = n / (n + mu)
    return 1.0 - nbinom.cdf(k - 1, n, p)


def _ece_binary(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected calibration error for binary hits.
    """
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")

    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)

        if not np.any(mask):
            continue

        avg_pred = p_pred[mask].mean()
        avg_actual = y_true[mask].mean()
        weight = mask.mean()
        ece += weight * abs(avg_pred - avg_actual)

    return float(ece)


def _brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    return float(np.mean((p_pred - y_true) ** 2))


def _build_threshold_calibration_table(
    df: pd.DataFrame,
    *,
    alpha: float,
    k_values: list[int],
    prob_band_edges: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      threshold_summary_df
      probability_band_df
    """
    if prob_band_edges is None:
        prob_band_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    summary_rows: list[dict] = []
    band_rows: list[dict] = []

    mu = df["model_pred"].to_numpy(dtype=float)
    actual_ast = df["ast"].to_numpy(dtype=float)

    for k in k_values:
        p_ge = _prob_ge_k(mu, k, alpha=alpha)
        y_hit = (actual_ast >= k).astype(float)

        summary_rows.append(
            {
                "threshold_k": int(k),
                "n_obs": int(len(df)),
                "mean_pred_prob": float(np.mean(p_ge)),
                "empirical_hit_rate": float(np.mean(y_hit)),
                "mean_error_pred_minus_actual": float(np.mean(p_ge) - np.mean(y_hit)),
                "brier_score": _brier_score(y_hit, p_ge),
                "ece_10bin": _ece_binary(y_hit, p_ge, n_bins=10),
            }
        )

        bands = pd.cut(
            p_ge,
            bins=prob_band_edges,
            include_lowest=True,
            right=True,
        )

        tmp = pd.DataFrame(
            {
                "threshold_k": k,
                "pred_prob": p_ge,
                "actual_hit": y_hit,
                "prob_band": bands.astype(str),
            }
        )

        grp = (
            tmp.groupby("prob_band", dropna=False)
            .agg(
                n_obs=("actual_hit", "size"),
                mean_pred_prob=("pred_prob", "mean"),
                empirical_hit_rate=("actual_hit", "mean"),
            )
            .reset_index()
        )
        grp["threshold_k"] = int(k)
        grp["mean_error_pred_minus_actual"] = grp["mean_pred_prob"] - grp["empirical_hit_rate"]

        band_rows.extend(grp.to_dict(orient="records"))

    return pd.DataFrame(summary_rows), pd.DataFrame(band_rows)


def _build_mean_bucket_table(
    df: pd.DataFrame,
    *,
    alpha: float,
    k_values: list[int],
) -> pd.DataFrame:
    """
    Diagnose where calibration is breaking by model mean bucket.
    """
    out_rows: list[dict] = []

    work = df.copy()
    work["pred_bucket"] = pd.cut(
        work["model_pred"],
        bins=[-0.001, 1.0, 3.0, 5.0, 7.0, 10.0, np.inf],
        labels=["lt_1", "1_3", "3_5", "5_7", "7_10", "10_plus"],
    )

    for bucket, g in work.groupby("pred_bucket", observed=False):
        if len(g) == 0:
            continue

        mu = g["model_pred"].to_numpy(dtype=float)
        actual_ast = g["ast"].to_numpy(dtype=float)

        row = {
            "pred_bucket": str(bucket),
            "n_obs": int(len(g)),
            "mean_model_pred": float(np.mean(mu)),
            "mean_actual_ast": float(np.mean(actual_ast)),
            "mean_bias_pred_minus_actual": float(np.mean(mu) - np.mean(actual_ast)),
        }

        for k in k_values:
            p_ge = _prob_ge_k(mu, k, alpha=alpha)
            y_hit = (actual_ast >= k).astype(float)
            row[f"k_{k}_mean_pred_prob"] = float(np.mean(p_ge))
            row[f"k_{k}_empirical_hit_rate"] = float(np.mean(y_hit))
            row[f"k_{k}_prob_error"] = float(np.mean(p_ge) - np.mean(y_hit))

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _build_simple_threshold_multipliers(
    threshold_summary_df: pd.DataFrame,
    *,
    min_obs: int = 1000,
    clip_low: float = 0.80,
    clip_high: float = 1.25,
) -> pd.DataFrame:
    """
    Lightweight calibration mapping:
      multiplier = empirical_hit_rate / mean_pred_prob

    This is intentionally simple and transparent.
    It can later be applied to raw predicted probabilities:
      p_calibrated = clip(p_raw * multiplier, 0, 1)

    We keep it separate from production until validated.
    """
    out = threshold_summary_df.copy()

    ratio = np.where(
        (out["mean_pred_prob"] > 1e-9) & (out["n_obs"] >= min_obs),
        out["empirical_hit_rate"] / out["mean_pred_prob"],
        1.0,
    )
    out["calibration_multiplier"] = np.clip(ratio, clip_low, clip_high)
    return out[["threshold_k", "n_obs", "mean_pred_prob", "empirical_hit_rate", "calibration_multiplier"]]


def calibrate_ast_probabilities(
    *,
    model_dir: str | Path = ASSISTS_MODEL_DIR,
    min_threshold: int = 2,
    max_threshold: int = 12,
) -> dict[str, float]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_artifacts(model_dir)
    alpha = float(artifacts.get("dispersion_alpha_mom", 0.0))

    df = _load_validation_predictions(model_dir)

    k_values = list(range(min_threshold, max_threshold + 1))

    threshold_summary_df, prob_band_df = _build_threshold_calibration_table(
        df,
        alpha=alpha,
        k_values=k_values,
    )

    mean_bucket_df = _build_mean_bucket_table(
        df,
        alpha=alpha,
        k_values=[3, 5, 7, 10],
    )

    multiplier_df = _build_simple_threshold_multipliers(threshold_summary_df)

    threshold_summary_path = model_dir / "ast_prob_calibration_summary.csv"
    prob_band_path = model_dir / "ast_prob_calibration_bands.csv"
    mean_bucket_path = model_dir / "ast_prob_calibration_by_pred_bucket.csv"
    multiplier_path = model_dir / "ast_prob_calibration_multipliers.csv"
    metrics_path = model_dir / "ast_prob_calibration_metrics.json"

    threshold_summary_df.to_csv(threshold_summary_path, index=False)
    prob_band_df.to_csv(prob_band_path, index=False)
    mean_bucket_df.to_csv(mean_bucket_path, index=False)
    multiplier_df.to_csv(multiplier_path, index=False)

    metrics = {
        "n_validation_rows": float(len(df)),
        "dispersion_alpha_used": float(alpha),
        "min_threshold": float(min_threshold),
        "max_threshold": float(max_threshold),
        "mean_abs_threshold_prob_error": float(
            threshold_summary_df["mean_error_pred_minus_actual"].abs().mean()
        ),
        "mean_brier_score": float(threshold_summary_df["brier_score"].mean()),
        "mean_ece_10bin": float(threshold_summary_df["ece_10bin"].mean()),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main(
    model_dir: str | Path = ASSISTS_MODEL_DIR,
    min_threshold: int = 2,
    max_threshold: int = 12,
) -> None:
    metrics = calibrate_ast_probabilities(
        model_dir=model_dir,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )

    print("[AST] Probability calibration complete.")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()