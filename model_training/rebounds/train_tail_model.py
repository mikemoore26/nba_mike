from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_training.common.history_prep import prepare_history_df
from model_training.common.time_split import time_split
from model_training.config import PATH_GAMLOGS_COMBINED, REBOUNDS_MODEL_DIR
from model_training.rebounds.features import REBOUND_FEATURES, build_all_rebounds_features
from model_training.rebounds.tail_model import (
    TAIL_THRESHOLDS,
    evaluate_tail_model,
    make_rebounds_tail_model,
    make_tail_target,
    tail_sample_weights,
)
from model_training.utils.team_codes import norm_team


def _select_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=["reb", "mp_minutes", "game_date"]).copy()
    out = out[(out["reb"] >= 0) & (out["mp_minutes"] >= 0)].copy()

    if "min_rolling_5" in out.columns:
        out = out[out["min_rolling_5"].notna()].copy()

    signal_ok = (
        out["reb_per_min_5"].notna()
        | out["player_reb_per_min_season"].notna()
        | out["player_reb_season_avg"].notna()
    )
    out = out[signal_ok].copy()

    return out


def _finalize_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        raise ValueError("No rebound feature columns found in dataframe.")

    usable = [c for c in existing if not df[c].isna().all()]
    if not usable:
        raise ValueError("All rebound feature columns are entirely NaN.")

    medians: dict[str, float] = {}
    X = df[usable].copy()

    for col in usable:
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        medians[col] = float(med)
        X[col] = X[col].fillna(med)

    return X, usable, medians


def _make_calibration_summary(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
) -> dict[str, float]:
    mean_pred = float(df[pred_col].mean())
    actual_rate = float(df[actual_col].mean())
    return {
        "n": int(len(df)),
        "mean_pred": mean_pred,
        "actual_rate": actual_rate,
        "diff_pred_minus_actual": mean_pred - actual_rate,
    }


def _make_probability_bucket_summary(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
) -> pd.DataFrame:
    out = df.copy()

    out["prob_bucket"] = pd.cut(
        out[pred_col],
        bins=[0.0, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0],
        labels=["0-.01", ".01-.03", ".03-.05", ".05-.10", ".10-.20", ".20-.35", ".35-.50", ".50+"],
        include_lowest=True,
    )

    summary = (
        out.groupby("prob_bucket", observed=False)
        .agg(
            n=(actual_col, "size"),
            mean_pred=(pred_col, "mean"),
            actual_rate=(actual_col, "mean"),
        )
        .reset_index()
    )
    summary["diff_pred_minus_actual"] = summary["mean_pred"] - summary["actual_rate"]
    return summary


def _make_minutes_bucket_summary(
    df: pd.DataFrame,
    *,
    pred_col: str,
    actual_col: str,
) -> pd.DataFrame:
    out = df.copy()

    if "mp_minutes" not in out.columns:
        out["minutes_bucket"] = "unknown"
    else:
        out["minutes_bucket"] = pd.cut(
            out["mp_minutes"],
            bins=[0, 12, 20, 28, 36, 60],
            labels=["0-12", "12-20", "20-28", "28-36", "36+"],
            include_lowest=True,
        )

    summary = (
        out.groupby("minutes_bucket", observed=False)
        .agg(
            n=(actual_col, "size"),
            mean_pred=(pred_col, "mean"),
            actual_rate=(actual_col, "mean"),
        )
        .reset_index()
    )
    summary["diff_pred_minus_actual"] = summary["mean_pred"] - summary["actual_rate"]
    return summary


def _json_safe_value(x):
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x.isoformat()
    if pd.isna(x):
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_json_safe_value)
    return out.to_dict(orient="records")


def _save_tail_debug_outputs(
    *,
    model_dir: Path,
    threshold: int,
    valid_df: pd.DataFrame,
    pred_prob: np.ndarray,
    y_valid: np.ndarray,
) -> dict:
    threshold_key = f"reb_ge_{threshold}"

    out = valid_df.copy()
    out[f"{threshold_key}_actual"] = y_valid
    out[f"{threshold_key}_pred_prob"] = pred_prob

    keep_cols = ["game_date", "player", "team", "opp", "reb"]
    extra_cols = [c for c in ["mp_minutes", "min_rolling_5", "reb_per_min_5", "minutes_bucket_code"] if c in out.columns]
    out = out[keep_cols + extra_cols + [f"{threshold_key}_actual", f"{threshold_key}_pred_prob"]].copy()

    out_path = model_dir / f"{threshold_key}_validation.csv"
    out.to_csv(out_path, index=False)

    calibration_summary = _make_calibration_summary(
        out,
        pred_col=f"{threshold_key}_pred_prob",
        actual_col=f"{threshold_key}_actual",
    )

    prob_bucket_summary = _make_probability_bucket_summary(
        out,
        pred_col=f"{threshold_key}_pred_prob",
        actual_col=f"{threshold_key}_actual",
    )
    prob_bucket_summary.to_csv(model_dir / f"{threshold_key}_prob_bucket_summary.csv", index=False)

    minutes_bucket_summary = _make_minutes_bucket_summary(
        out,
        pred_col=f"{threshold_key}_pred_prob",
        actual_col=f"{threshold_key}_actual",
    )
    minutes_bucket_summary.to_csv(model_dir / f"{threshold_key}_minutes_bucket_summary.csv", index=False)

    sample_rows = _json_safe_records(out.head(25))
    prob_bucket_records = _json_safe_records(prob_bucket_summary)
    minutes_bucket_records = _json_safe_records(minutes_bucket_summary)

    return {
        "calibration_summary": calibration_summary,
        "prob_bucket_summary": prob_bucket_records,
        "minutes_bucket_summary": minutes_bucket_records,
        "sample_rows": sample_rows,
    }


def train_rebounds_tail_models(
    *,
    csv_path: str | Path = PATH_GAMLOGS_COMBINED,
    model_dir: str | Path = REBOUNDS_MODEL_DIR,
    split_date: str = "2025-01-01",
) -> dict[str, dict]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)
    df = prepare_history_df(df, norm_team_fn=norm_team)
    df = build_all_rebounds_features(df)
    df = _select_training_frame(df)

    train_df, valid_df = time_split(df, split_date=split_date, date_col="game_date")

    print(f"[INFO] Date range: {df['game_date'].min().date()} → {df['game_date'].max().date()}")
    print(f"[INFO] Split date: {split_date}")
    print(f"[INFO] Train rows: {len(train_df)}")
    print(f"[INFO] Test rows: {len(valid_df)}")

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if valid_df.empty:
        raise ValueError("Validation split is empty.")

    X_train, used_features, train_feature_medians = _finalize_feature_matrix(train_df, REBOUND_FEATURES)
    X_valid = valid_df[used_features].copy()
    for col in used_features:
        X_valid[col] = X_valid[col].fillna(train_feature_medians[col])

    metrics_by_threshold: dict[str, dict] = {}
    diagnostics_by_threshold: dict[str, dict] = {}

    for threshold in TAIL_THRESHOLDS:
        y_train = make_tail_target(train_df["reb"], threshold)
        y_valid = make_tail_target(valid_df["reb"], threshold)

        sample_weight = tail_sample_weights(train_df, threshold)

        model = make_rebounds_tail_model()
        model.fit(X_train, y_train, sample_weight=sample_weight)

        train_prob = model.predict_proba(X_train)[:, 1]
        valid_prob = model.predict_proba(X_valid)[:, 1]

        train_metrics = evaluate_tail_model(y_true=y_train, y_prob=train_prob)
        valid_metrics = evaluate_tail_model(y_true=y_valid, y_prob=valid_prob)

        threshold_key = f"reb_ge_{threshold}"

        metrics_by_threshold[threshold_key] = {
            "train": train_metrics,
            "valid": valid_metrics,
        }

        diagnostics_by_threshold[threshold_key] = _save_tail_debug_outputs(
            model_dir=model_dir,
            threshold=threshold,
            valid_df=valid_df,
            pred_prob=valid_prob,
            y_valid=y_valid,
        )

        joblib.dump(model, model_dir / f"{threshold_key}_model.joblib")

    artifact_meta = {
        "target": "reb_tail",
        "split_date": split_date,
        "used_features": used_features,
        "feature_medians": train_feature_medians,
        "thresholds": TAIL_THRESHOLDS,
    }

    with open(model_dir / "reb_tail_artifacts.json", "w", encoding="utf-8") as f:
        json.dump(artifact_meta, f, indent=2)

    with open(model_dir / "reb_tail_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_by_threshold, f, indent=2)

    with open(model_dir / "reb_tail_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics_by_threshold, f, indent=2)

    return metrics_by_threshold


def main():
    metrics = train_rebounds_tail_models(
        csv_path=PATH_GAMLOGS_COMBINED,
        model_dir=REBOUNDS_MODEL_DIR,
    )

    print("[REB TAIL] Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()