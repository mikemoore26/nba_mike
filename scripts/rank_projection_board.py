from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _configure_utf8_output() -> None:
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


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Missing file: {path}")
    return pd.read_csv(path)


def _results_dir(run_date: str | None = None) -> Path:
    if run_date is None:
        run_date = datetime.today().strftime("%Y-%m-%d")
    return Path("results") / run_date


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _rank_pct(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    return s.rank(pct=True, method="average").fillna(0.5)


def rank_projection_board(run_date: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = _results_dir(run_date)

    board_path = results_dir / "projection_board.csv"
    board = _safe_read(board_path)

    numeric_cols = [
        "minutes_proj",
        "pred_pts",
        "pred_reb",
        "pred_ast",
        "pred_fg3",
    ]
    board = _ensure_numeric(board, numeric_cols)

    # Create role/stability/usage/fragility proxies if they are not already present.
    if "minutes_strength" not in board.columns:
        board["minutes_strength"] = board["minutes_proj"].fillna(0) / 36.0
    board["minutes_strength"] = pd.to_numeric(board["minutes_strength"], errors="coerce").fillna(0).clip(0, 1)

    if "minutes_elite" not in board.columns:
        board["minutes_elite"] = (board["minutes_proj"].fillna(0) >= 32).astype(float)
    board["minutes_elite"] = pd.to_numeric(board["minutes_elite"], errors="coerce").fillna(0).clip(0, 1)

    if "stability_score" not in board.columns:
        board["stability_score"] = 0.65 * board["minutes_strength"] + 0.35 * board["minutes_elite"]
    board["stability_score"] = pd.to_numeric(board["stability_score"], errors="coerce").fillna(0.5).clip(0, 1)

    pts_pct = _rank_pct(board["pred_pts"])
    reb_pct = _rank_pct(board["pred_reb"])
    ast_pct = _rank_pct(board["pred_ast"])
    fg3_pct = _rank_pct(board["pred_fg3"])
    minutes_pct = _rank_pct(board["minutes_proj"])

    if "usage_score" not in board.columns:
        board["usage_score"] = (
            0.45 * pts_pct +
            0.20 * ast_pct +
            0.15 * fg3_pct +
            0.20 * minutes_pct
        )
    board["usage_score"] = pd.to_numeric(board["usage_score"], errors="coerce").fillna(0.5).clip(0, 1)

    if "role_score" not in board.columns:
        board["role_score"] = (
            0.40 * board["stability_score"] +
            0.35 * board["usage_score"] +
            0.25 * minutes_pct
        )
    board["role_score"] = pd.to_numeric(board["role_score"], errors="coerce").fillna(0.5).clip(0, 1)

    if "fragility_score" not in board.columns:
        board["fragility_score"] = (
            1.0 - (0.55 * board["stability_score"] + 0.45 * minutes_pct)
        )
    board["fragility_score"] = pd.to_numeric(board["fragility_score"], errors="coerce").fillna(0.5).clip(0, 1)

    def _role_tier(row: pd.Series) -> str:
        role = row["role_score"]
        frag = row["fragility_score"]
        if role >= 0.80 and frag <= 0.12:
            return "core"
        if role >= 0.62 and frag <= 0.22:
            return "solid"
        if frag >= 0.22:
            return "fragile"
        return "thin"

    board["role_tier"] = board.apply(_role_tier, axis=1)

    board["pred_pts_pct"] = pts_pct
    board["pred_reb_pct"] = reb_pct
    board["pred_ast_pct"] = ast_pct
    board["pred_fg3_pct"] = fg3_pct
    board["minutes_pct"] = minutes_pct
    board["confidence_pct"] = 1.0  # placeholder until explicit confidence layer is added
    board["role_pct"] = _rank_pct(board["role_score"])
    board["stability_pct"] = _rank_pct(board["stability_score"])
    board["usage_pct"] = _rank_pct(board["usage_score"])
    board["fragility_penalty"] = board["fragility_score"]

    board["all_around_pred_score"] = (
        0.35 * board["pred_pts_pct"] +
        0.20 * board["pred_reb_pct"] +
        0.20 * board["pred_ast_pct"] +
        0.10 * board["pred_fg3_pct"] +
        0.15 * board["minutes_pct"]
    )

    # If already present, keep existing projection rank. Otherwise create one.
    if "projection_rank_score" not in board.columns:
        board["projection_rank_score"] = (
            0.45 * board["all_around_pred_score"] +
            0.20 * board["role_score"] +
            0.20 * board["stability_score"] +
            0.15 * (1 - board["fragility_score"])
        )
    board["projection_rank_score"] = pd.to_numeric(board["projection_rank_score"], errors="coerce").fillna(0.5).clip(0, 1)
    board["projection_rank_score_board"] = board["projection_rank_score"]

    board = board.sort_values(
        ["projection_rank_score", "role_score", "minutes_proj"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    board["overall_rank"] = range(1, len(board) + 1)

    ranked_path = results_dir / "projection_board_ranked.csv"
    board.to_csv(ranked_path, index=False)

    top_all_around = board[[
        c for c in [
            "player",
            "team",
            "opp",
            "minutes_proj",
            "role_score",
            "stability_score",
            "usage_score",
            "fragility_score",
            "role_tier",
            "projection_rank_score",
            "pred_pts",
            "pred_reb",
            "pred_ast",
            "pred_fg3",
        ] if c in board.columns
    ]].head(20).copy()

    top_all_around_path = results_dir / "top_all_around.csv"
    top_all_around.to_csv(top_all_around_path, index=False)

    print(f"[SAVED] projection_board_ranked -> {ranked_path}")
    print(f"[SAVED] top_all_around -> {top_all_around_path}")

    print("\n[TOP ALL-AROUND PREVIEW]")
    preview_cols = [
        c for c in [
            "player",
            "team",
            "opp",
            "minutes_proj",
            "role_score",
            "stability_score",
            "usage_score",
            "fragility_score",
            "role_tier",
            "projection_rank_score",
            "pred_pts",
            "pred_reb",
            "pred_ast",
            "pred_fg3",
        ] if c in top_all_around.columns
    ]
    print(top_all_around[preview_cols].head(20).to_string(index=False))

    return board, top_all_around


def main() -> None:
    rank_projection_board()


if __name__ == "__main__":
    main()