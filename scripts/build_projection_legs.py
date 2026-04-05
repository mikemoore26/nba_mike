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
# LEG BUILDING
# =========================
def _infer_line(pred_mean: float, stat: str) -> float:
    """
    Temporary pseudo-line logic until market layer is live.
    """
    if pd.isna(pred_mean):
        return 0.0

    if stat in {"pts", "reb", "ast", "fg3"}:
        # Half-step books-style line
        line = max(0.5, float(int(pred_mean)) + 0.5)
        return line

    return max(0.5, round(float(pred_mean), 1))


def _infer_p_hit(pred_mean: float, line: float) -> float:
    """
    Lightweight placeholder hit-probability proxy.
    Keep bounded and monotonic with edge.
    """
    if pd.isna(pred_mean) or pd.isna(line):
        return 0.50

    edge = float(pred_mean) - float(line)
    p_hit = 0.50 + 0.12 * edge
    return float(min(0.95, max(0.05, p_hit)))


def build_projection_legs(run_date: str | None = None) -> pd.DataFrame:
    results_dir = _results_dir(run_date)

    board_path = results_dir / "projection_board_ranked.csv"
    board = _safe_read(board_path)

    stat_map = {
        "pts": "pred_pts",
        "reb": "pred_reb",
        "ast": "pred_ast",
        "fg3": "pred_fg3",
    }

    rows: list[dict] = []

    for _, row in board.iterrows():
        base = {
            "game_date": datetime.today().strftime("%Y-%m-%d"),
            "player": row.get("player", ""),
            "team": row.get("team", ""),
            "opp": row.get("opp", ""),
            "minutes_proj": row.get("minutes_proj", 0.0),
            "projection_rank_score": row.get("projection_rank_score", 0.5),
            "role_score": row.get("role_score", 0.5),
            "stability_score": row.get("stability_score", 0.5),
            "usage_score": row.get("usage_score", 0.5),
            "fragility_score": row.get("fragility_score", 0.5),
            "role_tier": row.get("role_tier", ""),
            "overall_rank": row.get("overall_rank", None),
            "confidence_score": row.get("confidence_score", 1.0),
        }

        for stat, pred_col in stat_map.items():
            pred_mean = row.get(pred_col, pd.NA)

            if pd.isna(pred_mean):
                continue

            pred_mean = float(pred_mean)
            line = _infer_line(pred_mean, stat)
            delta = pred_mean - line
            p_hit = _infer_p_hit(pred_mean, line)

            # crude eligibility until market / lineup layer is more mature
            minutes_proj = float(base["minutes_proj"]) if pd.notna(base["minutes_proj"]) else 0.0
            is_eligible = 1 if minutes_proj >= 10 and pred_mean >= 0.75 else 0
            eligibility_reason = "ok"
            if minutes_proj < 10:
                eligibility_reason = "minutes_lt_10"
            elif pred_mean < 0.75:
                eligibility_reason = "pred_mean_lt_0.75"

            rows.append({
                **base,
                "stat": stat,
                "pred_mean": pred_mean,
                "baseline_mean": pred_mean,
                "dist_name": "poisson" if stat in {"pts", "reb", "ast"} else "nbinom",
                "dispersion": 0.0 if stat in {"pts", "reb", "ast"} else 0.1616709598572226,
                "is_eligible": is_eligible,
                "eligibility_reason": eligibility_reason,
                "model_name": f"{stat}_composed" if stat == "fg3" else f"{stat}_hgbr",
                "model_version": "v1",
                "line": line,
                "side": "over",
                "p_hit": p_hit,
                "edge_raw": delta,
                "edge_abs": abs(delta),
                # placeholder scores, refined in scored legs script
                "score_safe": p_hit,
                "score_balanced": p_hit,
                "score_lotto": p_hit,
                "minutes_conf": 1.0,
                "stat_vol_penalty": 0.0,
                "extreme_line_penalty": 0.0,
                "rank_score": row.get("projection_rank_score", 0.5),
            })

    legs = pd.DataFrame(rows)

    out_path = results_dir / "projection_legs.csv"
    legs.to_csv(out_path, index=False)

    print(f"[SAVED] projection_legs -> {out_path}")
    print(f"[INFO] legs count: {len(legs)}")

    preview_cols = [
        c for c in [
            "player",
            "team",
            "opp",
            "stat",
            "pred_mean",
            "line",
            "p_hit",
            "minutes_proj",
            "is_eligible",
            "eligibility_reason",
        ] if c in legs.columns
    ]

    print("\n[PREVIEW]")
    if not legs.empty:
        print(legs[preview_cols].head(10).to_string(index=False))
    else:
        print("[INFO] No legs created.")

    return legs


def main() -> None:
    build_projection_legs()


if __name__ == "__main__":
    main()