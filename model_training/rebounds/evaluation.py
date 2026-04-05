from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_reb_validation_diagnostics(
    *,
    valid_df: pd.DataFrame,
    pred_valid: np.ndarray,
    baseline_valid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | dict]]:
    """
    Build row-level and aggregate validation diagnostics.
    Safe additive utility: does not modify training schema/artifacts.
    """

    diag = valid_df.copy()

    diag["model_pred"] = np.asarray(pred_valid, dtype=float)
    diag["baseline_pred"] = np.asarray(baseline_valid, dtype=float)

    diag["model_minus_baseline"] = diag["model_pred"] - diag["baseline_pred"]

    diag["model_resid"] = diag["reb"] - diag["model_pred"]
    diag["baseline_resid"] = diag["reb"] - diag["baseline_pred"]

    diag["model_abs_err"] = np.abs(diag["model_resid"])
    diag["baseline_abs_err"] = np.abs(diag["baseline_resid"])

    diag["model_sq_err"] = diag["model_resid"] ** 2
    diag["baseline_sq_err"] = diag["baseline_resid"] ** 2

    diag["model_overpred"] = (diag["model_resid"] < 0).astype(int)
    diag["model_underpred"] = (diag["model_resid"] > 0).astype(int)

    # -------------------------
    # Minutes buckets
    # -------------------------
    if "mp_minutes" in diag.columns:
        diag["minutes_bucket"] = pd.cut(
            diag["mp_minutes"],
            bins=[0, 12, 20, 28, 36, 60],
            labels=["0-12", "12-20", "20-28", "28-36", "36+"],
            include_lowest=True,
        )
    else:
        diag["minutes_bucket"] = "unknown"

    minutes_summary = (
        diag.groupby("minutes_bucket", observed=False)
        .agg(
            n=("reb", "size"),
            actual_mean=("reb", "mean"),
            model_mean=("model_pred", "mean"),
            baseline_mean=("baseline_pred", "mean"),
            model_mae=("model_abs_err", "mean"),
            baseline_mae=("baseline_abs_err", "mean"),
            model_rmse=("model_sq_err", lambda x: float(np.sqrt(np.mean(x)))),
            baseline_rmse=("baseline_sq_err", lambda x: float(np.sqrt(np.mean(x)))),
            model_bias=("model_resid", "mean"),
            baseline_bias=("baseline_resid", "mean"),
        )
        .reset_index()
    )
    minutes_summary["mae_lift_vs_baseline"] = (
        minutes_summary["baseline_mae"] - minutes_summary["model_mae"]
    )

    # -------------------------
    # Prediction buckets
    # -------------------------
    diag["pred_bucket"] = pd.cut(
        diag["model_pred"],
        bins=[0, 3, 5, 7, 9, 12, 20],
        labels=["0-3", "3-5", "5-7", "7-9", "9-12", "12+"],
        include_lowest=True,
    )

    pred_summary = (
        diag.groupby("pred_bucket", observed=False)
        .agg(
            n=("reb", "size"),
            actual_mean=("reb", "mean"),
            pred_mean=("model_pred", "mean"),
            mae=("model_abs_err", "mean"),
            rmse=("model_sq_err", lambda x: float(np.sqrt(np.mean(x)))),
            bias=("model_resid", "mean"),
        )
        .reset_index()
    )

    # -------------------------
    # Tail summaries
    # -------------------------
    tail_rows = []
    for k in [8, 10, 12, 14]:
        actual_rate = float((diag["reb"] >= k).mean())
        pred_rate = float((diag["model_pred"] >= k).mean())
        baseline_rate = float((diag["baseline_pred"] >= k).mean())

        tail_rows.append(
            {
                "threshold": k,
                "actual_rate": actual_rate,
                "model_rate_from_mean": pred_rate,
                "baseline_rate_from_mean": baseline_rate,
                "model_minus_actual": pred_rate - actual_rate,
                "baseline_minus_actual": baseline_rate - actual_rate,
            }
        )

    tail_summary = pd.DataFrame(tail_rows)

    # -------------------------
    # Player bias summary
    # -------------------------
    player_summary = (
        diag.groupby("player", observed=False)
        .agg(
            n=("reb", "size"),
            actual_mean=("reb", "mean"),
            pred_mean=("model_pred", "mean"),
            mae=("model_abs_err", "mean"),
            bias=("model_resid", "mean"),
        )
        .reset_index()
        .sort_values(["n", "mae"], ascending=[False, False])
    )

    # -------------------------
    # Opponent bias summary
    # -------------------------
    if "opp" in diag.columns:
        opp_summary = (
            diag.groupby("opp", observed=False)
            .agg(
                n=("reb", "size"),
                actual_mean=("reb", "mean"),
                pred_mean=("model_pred", "mean"),
                mae=("model_abs_err", "mean"),
                bias=("model_resid", "mean"),
            )
            .reset_index()
            .sort_values(["n", "mae"], ascending=[False, False])
        )
    else:
        opp_summary = pd.DataFrame(columns=["opp", "n", "actual_mean", "pred_mean", "mae", "bias"])

    # -------------------------
    # Global summary
    # -------------------------
    global_summary = {
        "n_valid": int(len(diag)),
        "model_mae": float(diag["model_abs_err"].mean()),
        "baseline_mae": float(diag["baseline_abs_err"].mean()),
        "model_rmse": float(np.sqrt(diag["model_sq_err"].mean())),
        "baseline_rmse": float(np.sqrt(diag["baseline_sq_err"].mean())),
        "model_bias": float(diag["model_resid"].mean()),
        "baseline_bias": float(diag["baseline_resid"].mean()),
        "model_overpredict_rate": float(diag["model_overpred"].mean()),
        "model_underpredict_rate": float(diag["model_underpred"].mean()),
    }

    outputs = {
        "global_summary": global_summary,
        "minutes_summary": minutes_summary,
        "pred_summary": pred_summary,
        "tail_summary": tail_summary,
        "player_summary": player_summary,
        "opp_summary": opp_summary,
    }

    return diag, outputs


def save_reb_validation_diagnostics(
    *,
    model_dir: str | Path,
    diag_df: pd.DataFrame,
    outputs: dict[str, pd.DataFrame | dict],
) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    diag_df.to_csv(model_dir / "reb_validation_diagnostics.csv", index=False)

    if "minutes_summary" in outputs:
        outputs["minutes_summary"].to_csv(model_dir / "reb_minutes_summary.csv", index=False)

    if "pred_summary" in outputs:
        outputs["pred_summary"].to_csv(model_dir / "reb_prediction_bucket_summary.csv", index=False)

    if "tail_summary" in outputs:
        outputs["tail_summary"].to_csv(model_dir / "reb_tail_summary.csv", index=False)

    if "player_summary" in outputs:
        outputs["player_summary"].to_csv(model_dir / "reb_player_summary.csv", index=False)

    if "opp_summary" in outputs:
        outputs["opp_summary"].to_csv(model_dir / "reb_opp_summary.csv", index=False)

    payload = {}
    for k, v in outputs.items():
        if isinstance(v, pd.DataFrame):
            payload[k] = v.to_dict(orient="records")
        else:
            payload[k] = v

    with open(model_dir / "reb_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)