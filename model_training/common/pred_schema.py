# model_training/common/pred_schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


CORE_ID_COLS = ["player", "team", "opp", "is_home", "game_date"]


@dataclass(frozen=True)
class PredSchema:
    """
    Unified prediction output contract.

    Required output columns:
      - player, team, opp, is_home, game_date
      - stat_name
      - pred_mean
    Optional:
      - pred_sd (later)
      - baseline_mean
      - delta_mean
      - p_ge_*, p_over_*
      - model_name, model_version
    """
    stat_name: str
    model_name: str = ""
    model_version: str = ""


def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" not in out.columns:
        if "date" in out.columns:
            out["game_date"] = out["date"]
        else:
            raise ValueError("pred df missing game_date (or legacy date).")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date"]).copy()
    return out


def standardize_pred_df(
    df: pd.DataFrame,
    *,
    schema: PredSchema,
    mean_col: str,
    baseline_col: str | None = None,
    delta_col: str | None = None,
    sd_col: str | None = None,
    extra_keep: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Convert a model-specific output df into a unified contract.

    - Renames mean_col -> pred_mean
    - Adds stat_name, model_name, model_version
    - Keeps any probability columns starting with p_ge_ or p_over_
    - Keeps optional baseline/delta/sd as baseline_mean/delta_mean/pred_sd

    This is what makes mixed-stat ticket building model-agnostic.
    """
    out = _ensure_game_date(df)

    missing_ids = [c for c in CORE_ID_COLS if c not in out.columns]
    if missing_ids:
        raise ValueError(f"Prediction df missing required id cols: {missing_ids}")

    if mean_col not in out.columns:
        raise ValueError(f"mean_col '{mean_col}' not in df columns")

    out = out.copy()
    out["stat_name"] = schema.stat_name
    out["model_name"] = schema.model_name
    out["model_version"] = schema.model_version

    out["pred_mean"] = out[mean_col]

    if baseline_col:
        if baseline_col not in out.columns:
            raise ValueError(f"baseline_col '{baseline_col}' not in df columns")
        out["baseline_mean"] = out[baseline_col]

    if delta_col:
        if delta_col not in out.columns:
            raise ValueError(f"delta_col '{delta_col}' not in df columns")
        out["delta_mean"] = out[delta_col]

    if sd_col:
        if sd_col not in out.columns:
            raise ValueError(f"sd_col '{sd_col}' not in df columns")
        out["pred_sd"] = out[sd_col]

    prob_cols = [c for c in out.columns if c.startswith("p_ge_") or c.startswith("p_over_")]
    keep = CORE_ID_COLS + [
        "stat_name",
        "pred_mean",
        "model_name",
        "model_version",
    ]
    for c in ["baseline_mean", "delta_mean", "pred_sd"]:
        if c in out.columns:
            keep.append(c)

    keep += prob_cols

    # keep any requested extras that exist
    for c in extra_keep:
        if c in out.columns and c not in keep:
            keep.append(c)

    return out[keep].copy()


def validate_pred_schema(df: pd.DataFrame) -> None:
    """
    Fail loud if the unified schema is violated.
    """
    required = set(CORE_ID_COLS + ["stat_name", "pred_mean", "model_name", "model_version"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unified pred schema missing: {sorted(missing)}")