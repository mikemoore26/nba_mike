# scripts/build_scored_projection_legs.py
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


# =========================
# UTF-8 SAFETY (WINDOWS)
# =========================
def _configure_utf8_output() -> None:
    os.environ["PYTHONUTF8"] = "1"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_utf8_output()


# =========================
# PATH HELPERS
# =========================
def _results_dir(run_date: str | None = None) -> Path:
    if run_date is None:
        run_date = datetime.today().strftime("%Y-%m-%d")
    return Path("results") / run_date


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Missing file: {path}")
    return pd.read_csv(path)


# =========================
# COLUMN CLEANING
# =========================
def _clean_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dup_cols = [c for c in df.columns if c.endswith("_y")]
    if dup_cols:
        df = df.drop(columns=dup_cols, errors="ignore")

    rename_map = {c: c[:-2] for c in df.columns if c.endswith("_x")}
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "minutes_proj": 0.0,
        "is_eligible": 1,
        "eligibility_reason": "",
        "role_score": 0.5,
        "stability_score": 0.5,
        "usage_score": 0.5,
        "fragility_score": 0.5,
        "projection_rank_score": 0.5,
        "confidence_score": 1.0,
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    numeric_cols = [
        "minutes_proj",
        "is_eligible",
        "role_score",
        "stability_score",
        "usage_score",
        "fragility_score",
        "projection_rank_score",
        "confidence_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col])

    return df


# =========================
# SANITY REPAIR
# =========================
def _repair_minutes_and_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bad_minutes_mask = df["minutes_proj"] <= 0
    star_mask = (
        (df["role_score"] >= 0.75) |
        (df["projection_rank_score"] >= 0.75)
    )

    repair_mask = bad_minutes_mask & star_mask

    if repair_mask.any():
        print(f"[REPAIR] Fixing {int(repair_mask.sum())} high-role players with 0 minutes")
        df.loc[repair_mask, "minutes_proj"] = 30.0
        df.loc[repair_mask, "is_eligible"] = 1
        df.loc[repair_mask, "eligibility_reason"] = "repaired_star_minutes"

    return df


def _debug_bad_rows(df: pd.DataFrame) -> None:
    bad = df[df["minutes_proj"] <= 0]

    if not bad.empty:
        print("\n[WARNING] Players with 0 minutes still present:")
        cols = [
            c for c in [
                "player",
                "team",
                "opp",
                "stat",
                "minutes_proj",
                "role_score",
                "projection_rank_score",
                "is_eligible",
                "eligibility_reason",
            ] if c in df.columns
        ]
        preview = bad[cols].drop_duplicates()
        print(preview.to_string(index=False))


# =========================
# SCORE BUILDING
# =========================
def _ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "p_hit": 0.0,
        "delta": 0.0,
        "safe_score": 0.0,
        "balanced_score": 0.0,
        "lotto_score": 0.0,
    }

    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val)

    need_safe = "safe_score" not in df.columns or df["safe_score"].abs().sum() == 0
    need_bal = "balanced_score" not in df.columns or df["balanced_score"].abs().sum() == 0
    need_lotto = "lotto_score" not in df.columns or df["lotto_score"].abs().sum() == 0

    if need_safe:
        df["safe_score"] = (
            0.50 * df["p_hit"]
            + 0.20 * df["role_score"]
            + 0.20 * df["stability_score"]
            + 0.10 * df["delta"].clip(lower=0, upper=1.5)
        )

    if need_bal:
        df["balanced_score"] = (
            0.35 * df["p_hit"]
            + 0.25 * df["delta"].clip(lower=0)
            + 0.15 * df["role_score"]
            + 0.10 * df["stability_score"]
            + 0.15 * df["projection_rank_score"]
        )

    if need_lotto:
        df["lotto_score"] = (
            0.45 * df["delta"].clip(lower=0)
            + 0.20 * df["usage_score"]
            + 0.15 * df["projection_rank_score"]
            + 0.10 * df["p_hit"]
            + 0.10 * df["role_score"]
        )

    return df


# =========================
# MAIN BUILD
# =========================
def build_scored_projection_legs(run_date: str | None = None) -> pd.DataFrame:
    results_dir = _results_dir(run_date)

    legs_path = results_dir / "projection_legs.csv"
    board_path = results_dir / "projection_board_ranked.csv"

    legs = _safe_read(legs_path)
    board = _safe_read(board_path)

    print(f"[INFO] legs: {len(legs)}")
    print(f"[INFO] board: {len(board)}")

    board_keep = [
        c for c in [
            "player",
            "team",
            "opp",
            "minutes_proj",
            "projection_rank_score",
            "confidence_score",
            "role_score",
            "stability_score",
            "usage_score",
            "fragility_score",
            "role_tier",
            "overall_rank",
        ] if c in board.columns
    ]
    board_small = board[board_keep].copy()

    key_cols = ["player", "team", "opp"]
    overlap_drop = [c for c in board_small.columns if c in legs.columns and c not in key_cols]
    if overlap_drop:
        legs = legs.drop(columns=overlap_drop, errors="ignore")

    df = legs.merge(
        board_small,
        on=key_cols,
        how="left",
        suffixes=("", "_board"),
    )

    df = _clean_merge(df)
    df = _ensure_columns(df)
    df = _repair_minutes_and_eligibility(df)
    df = _ensure_scores(df)

    _debug_bad_rows(df)

    out_path = results_dir / "projection_legs_scored.csv"
    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved scored legs to {out_path}")

    return df


def main():
    build_scored_projection_legs()


if __name__ == "__main__":
    main()