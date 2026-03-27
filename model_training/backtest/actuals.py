from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATE_CANDIDATES = ["game_date", "date", "game_dt"]
PLAYER_CANDIDATES = ["player", "player_name", "name"]
TEAM_CANDIDATES = ["team", "team_abbr", "tm"]
MINUTES_CANDIDATES = ["mp_minutes", "minutes", "min", "mp"]

STAT_TARGET_CANDIDATES = {
    "pts": ["pts"],
    "reb": ["reb", "trb"],
    "ast": ["ast"],
    "fg3": ["fg3m", "fg3"],
    "fg3m": ["fg3m", "fg3"],
}


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}")


def _normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def _resolve_actual_source_col(stat: str, available_cols: set[str]) -> str | None:
    stat = str(stat).lower().strip()
    candidates = STAT_TARGET_CANDIDATES.get(stat, [stat])

    for col in candidates:
        if col in available_cols:
            return col
    return None


def load_actuals(gamelog_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(gamelog_csv, low_memory=False)

    date_col = _first_existing(df, DATE_CANDIDATES)
    player_col = _first_existing(df, PLAYER_CANDIDATES)
    team_col = _first_existing(df, TEAM_CANDIDATES)

    df = df.copy()
    df["game_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    df["player_norm"] = _normalize_text(df[player_col])
    df["team_norm"] = _normalize_text(df[team_col])

    minutes_col = next((c for c in MINUTES_CANDIDATES if c in df.columns), None)
    if minutes_col is None:
        df["actual_minutes"] = np.nan
    else:
        df["actual_minutes"] = pd.to_numeric(df[minutes_col], errors="coerce")

    keep_cols = ["game_date", "player_norm", "team_norm", "actual_minutes"]
    for col in ["pts", "reb", "trb", "ast", "fg3", "fg3m"]:
        if col in df.columns:
            keep_cols.append(col)

    out = (
        df[keep_cols]
        .dropna(subset=["game_date", "player_norm", "team_norm"])
        .drop_duplicates(subset=["game_date", "player_norm", "team_norm"])
        .reset_index(drop=True)
    )

    return out


def attach_actuals(preds: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["game_date", "player", "team", "stat", "pred_mean"]
    missing = [c for c in required_cols if c not in preds.columns]
    if missing:
        raise KeyError(f"Prediction frame missing required cols: {missing}")

    df = preds.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["player_norm"] = _normalize_text(df["player"])
    df["team_norm"] = _normalize_text(df["team"])

    merged = df.merge(
        actuals,
        on=["game_date", "player_norm", "team_norm"],
        how="left",
        validate="m:1",
    )

    merged["pred_mean"] = pd.to_numeric(merged["pred_mean"], errors="coerce")
    merged["baseline_mean"] = pd.to_numeric(merged.get("baseline_mean"), errors="coerce")
    merged["minutes_proj"] = pd.to_numeric(merged.get("minutes_proj"), errors="coerce")
    merged["minutes_actual"] = pd.to_numeric(merged.get("actual_minutes"), errors="coerce")

    available_cols = set(merged.columns)
    merged["actual_source_col"] = merged["stat"].apply(
        lambda s: _resolve_actual_source_col(s, available_cols)
    )

    def _actual_from_row(row: pd.Series):
        source_col = row.get("actual_source_col")
        if source_col is None or pd.isna(source_col):
            return np.nan
        return row.get(source_col, np.nan)

    merged["actual_value"] = merged.apply(_actual_from_row, axis=1)
    merged["actual_value"] = pd.to_numeric(merged["actual_value"], errors="coerce")

    merged["has_prediction"] = merged["pred_mean"].notna().astype(int)
    merged["has_actual"] = merged["actual_value"].notna().astype(int)

    if "is_eligible" in merged.columns:
        merged["is_eligible"] = pd.to_numeric(
            merged["is_eligible"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        merged["is_eligible"] = 0

    merged["is_scored_row"] = (
        (merged["has_prediction"] == 1)
        & (merged["has_actual"] == 1)
    ).astype(int)

    merged["is_bettable_row"] = (
        (merged["is_scored_row"] == 1)
        & (merged["is_eligible"] == 1)
    ).astype(int)

    merged["error"] = merged["actual_value"] - merged["pred_mean"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2

    merged["beat_projection"] = np.where(
        merged["is_scored_row"] == 1,
        (merged["actual_value"] >= merged["pred_mean"]).astype(float),
        np.nan,
    )

    merged["beat_baseline"] = np.where(
        merged["is_scored_row"] == 1,
        (merged["actual_value"] >= merged["baseline_mean"]).astype(float),
        np.nan,
    )

    return merged.reset_index(drop=True)


def make_scored_eval(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if "is_scored_row" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    return df[df["is_scored_row"] == 1].reset_index(drop=True)


def make_bettable_eval(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if "is_bettable_row" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    return df[df["is_bettable_row"] == 1].reset_index(drop=True)


def write_eval_outputs(
    eval_df: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = eval_df.copy() if eval_df is not None else pd.DataFrame()
    scored_df = make_scored_eval(raw_df)
    bettable_df = make_bettable_eval(raw_df)

    raw_df.to_csv(output_dir / "backtest_player_eval_raw.csv", index=False)
    scored_df.to_csv(output_dir / "backtest_player_eval.csv", index=False)
    bettable_df.to_csv(output_dir / "backtest_player_eval_bettable.csv", index=False)