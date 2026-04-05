from __future__ import annotations

import os
import sys


def _configure_utf8_output() -> None:
    """
    Prevent Windows console crashes on non-ASCII player names.
    """
    os.environ["PYTHONUTF8"] = "1"

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue

        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_utf8_output()

# -------------------------------------------------------------------
# KEEP ALL YOUR EXISTING IMPORTS AND LOGIC BELOW THIS LINE
# -------------------------------------------------------------------

from datetime import datetime
from pathlib import Path
import pandas as pd
from functools import reduce


KEY_COLS = ["player", "team", "opp"]


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Missing file: {path}")
    return pd.read_csv(path)


def _compact_projection_frame(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    pred_col = f"pred_{stat}"
    keep_cols = [c for c in KEY_COLS if c in df.columns]

    if "minutes_proj" in df.columns:
        keep_cols.append("minutes_proj")

    if "projection_rank_score" in df.columns:
        keep_cols.append("projection_rank_score")

    if "confidence_score" in df.columns:
        keep_cols.append("confidence_score")

    if "pred_mean" in df.columns:
        df = df.copy()
        df[pred_col] = df["pred_mean"]
        keep_cols.append(pred_col)
    elif pred_col in df.columns:
        keep_cols.append(pred_col)
    else:
        raise ValueError(f"Could not find prediction column for stat={stat}")

    out = df[keep_cols].copy()

    # prevent duplicate cols during merge
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _outer_merge_projection_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=KEY_COLS)

    merged = frames[0].copy()

    for df in frames[1:]:
        merged = merged.merge(df, on=KEY_COLS, how="outer", suffixes=("", "_dup"))

        # coalesce duplicate helper cols
        for base_col in ["minutes_proj", "projection_rank_score", "confidence_score"]:
            dup_col = f"{base_col}_dup"
            if dup_col in merged.columns:
                if base_col not in merged.columns:
                    merged[base_col] = merged[dup_col]
                else:
                    merged[base_col] = merged[base_col].fillna(merged[dup_col])
                merged = merged.drop(columns=[dup_col])

    return merged


def build_projection_board(run_date: str | None = None) -> pd.DataFrame:
    if run_date is None:
        run_date = datetime.today().strftime("%Y-%m-%d")

    results_dir = Path("results") / run_date

    stat_file_map = {
        "pts": results_dir / "pred_pts.csv",
        "reb": results_dir / "pred_reb.csv",
        "ast": results_dir / "pred_ast.csv",
        "fg3": results_dir / "pred_fg3.csv",
    }

    frames: list[pd.DataFrame] = []

    for stat, path in stat_file_map.items():
        df = _safe_read(path)
        frames.append(_compact_projection_frame(df, stat))

    board = _outer_merge_projection_frames(frames)

    # stable sort
    sort_cols = [c for c in ["team", "opp", "player"] if c in board.columns]
    if sort_cols:
        board = board.sort_values(sort_cols).reset_index(drop=True)

    out_path = results_dir / "projection_board.csv"
    compact_path = results_dir / "projection_board_compact.csv"

    board.to_csv(out_path, index=False)

    compact_cols = [c for c in [
        "player", "team", "opp", "minutes_proj",
        "pred_pts", "pred_reb", "pred_ast", "pred_fg3"
    ] if c in board.columns]
    board[compact_cols].to_csv(compact_path, index=False)

    print(f"[SAVED] projection_board -> {out_path}")
    print(f"[SAVED] projection_board_compact -> {compact_path}")

    preview_cols = compact_cols
    print("\n[PREVIEW]")
    print(board[preview_cols].head(20).to_string(index=False))

    return board


def main() -> None:
    build_projection_board()


if __name__ == "__main__":
    main()