from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from model_training.config import ASSISTS_MODEL_DIR


# -----------------------------
# IO + schema helpers
# -----------------------------
def _coalesce_column(df: pd.DataFrame, candidates: list[str], new_col: str) -> pd.DataFrame:
    out = df.copy()
    existing = [c for c in candidates if c in out.columns]
    if not existing:
        return out
    out[new_col] = out[existing].bfill(axis=1).iloc[:, 0]
    return out


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_name(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    # light normalization only; do NOT over-clean and risk false merges
    s = (
        s.str.replace(".", "", regex=False)
         .str.replace("'", "", regex=False)
         .str.replace("-", " ", regex=False)
         .str.replace(r"\s+", " ", regex=True)
         .str.lower()
         .str.strip()
    )
    return s


def _american_to_implied_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=odds.index, dtype=float)

    pos = odds > 0
    neg = odds < 0

    out.loc[pos] = 100.0 / (odds.loc[pos] + 100.0)
    out.loc[neg] = (-odds.loc[neg]) / ((-odds.loc[neg]) + 100.0)
    return out


def _load_ast_artifacts(model_dir: str | Path) -> dict:
    model_dir = Path(model_dir)
    with open(model_dir / "ast_artifacts.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pred_ast(pred_path: str | Path) -> pd.DataFrame:
    pred_path = Path(pred_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    df = pd.read_csv(pred_path)

    required = ["player", "team", "opp", "pred_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pred_ast file missing required columns: {missing}")

    out = df.copy()
    out["player_norm"] = _normalize_name(out["player"])
    out["pred_mean"] = _to_float(out["pred_mean"]).clip(lower=0)
    if "baseline_mean" in out.columns:
        out["baseline_mean"] = _to_float(out["baseline_mean"]).clip(lower=0)
    else:
        out["baseline_mean"] = np.nan

    if "minutes_proj" in out.columns:
        out["minutes_proj"] = _to_float(out["minutes_proj"])
    else:
        out["minutes_proj"] = np.nan

    if "is_eligible" in out.columns:
        out["is_eligible"] = _to_float(out["is_eligible"]).fillna(0).astype(int)
    else:
        out["is_eligible"] = 0

    if "eligibility_reason" not in out.columns:
        out["eligibility_reason"] = "unknown"

    return out


def _load_market_ast(market_path: str | Path) -> pd.DataFrame:
    market_path = Path(market_path)
    if not market_path.exists():
        raise FileNotFoundError(f"Market file not found: {market_path}")

    df = pd.read_csv(market_path)
    out = df.copy()

    # likely identity fields
    out = _coalesce_column(out, ["player", "player_name", "name", "Player"], "player")
    out = _coalesce_column(out, ["team", "team_abbr", "Team"], "team")
    out = _coalesce_column(out, ["opp", "opponent", "Opponent"], "opp")

    # market type / stat
    out = _coalesce_column(out, ["stat", "market_type", "market", "prop_type"], "stat")
    out = _coalesce_column(out, ["side", "selection", "bet_side", "outcome"], "side")

    # line / odds
    out = _coalesce_column(out, ["line", "points", "threshold", "value"], "line")
    out = _coalesce_column(out, ["odds", "price", "american_odds"], "odds")

    # optional source columns
    out = _coalesce_column(out, ["sportsbook", "book", "source"], "line_source")

    if "player" not in out.columns:
        raise ValueError("Market file missing player column.")
    if "line" not in out.columns:
        raise ValueError("Market file missing line column.")

    out["player_norm"] = _normalize_name(out["player"])
    out["line"] = _to_float(out["line"])
    out["odds"] = _to_float(out["odds"]) if "odds" in out.columns else np.nan

    if "stat" not in out.columns:
        out["stat"] = "ast"
    out["stat"] = out["stat"].astype(str).str.lower().str.strip()

    # keep AST only
    ast_aliases = {
        "ast", "assists", "assist", "player_assists", "player assists"
    }
    out = out[out["stat"].isin(ast_aliases)].copy()

    # normalize side
    if "side" in out.columns:
        side = out["side"].astype(str).str.lower().str.strip()
        side = side.replace(
            {
                "o": "over",
                "u": "under",
                "over ": "over",
                "under ": "under",
            }
        )
        out["side"] = side
    else:
        out["side"] = np.nan

    if "line_source" not in out.columns:
        out["line_source"] = "unknown"

    return out


# -----------------------------
# Probability math
# -----------------------------
def _prob_ge_k(mu: np.ndarray, k: int, alpha: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 0.0, None)

    if k <= 0:
        return np.ones_like(mu, dtype=float)

    if alpha <= 1e-12:
        return 1.0 - poisson.cdf(k - 1, mu)

    n = 1.0 / alpha
    p = n / (n + mu)
    return 1.0 - nbinom.cdf(k - 1, n, p)


def _prob_over_pushable_line(mu: np.ndarray, line: np.ndarray, alpha: float) -> np.ndarray:
    """
    For standard assist props:
      over 5.5 => P(X >= 6)
      over 5.0 => P(X >= 6)  [strict over]
    """
    k = np.floor(line).astype(int) + 1
    return _prob_ge_k(mu, k, alpha=alpha)


def _prob_under_pushable_line(mu: np.ndarray, line: np.ndarray, alpha: float) -> np.ndarray:
    """
    For standard assist props:
      under 5.5 => P(X <= 5) = 1 - P(X >= 6)
      under 5.0 => P(X <= 4) = 1 - P(X >= 5)
    """
    whole = np.isclose(line, np.floor(line))
    k = np.where(whole, np.floor(line).astype(int), np.floor(line).astype(int) + 1)
    return 1.0 - _prob_ge_k(mu, k, alpha=alpha)


# -----------------------------
# Core build
# -----------------------------
def build_ast_edge_table(
    *,
    pred_path: str | Path,
    market_path: str | Path,
    model_dir: str | Path = ASSISTS_MODEL_DIR,
    out_path: str | Path | None = None,
    min_prob_edge: float = 0.0,
    eligible_only: bool = False,
) -> pd.DataFrame:
    pred_df = _load_pred_ast(pred_path)
    market_df = _load_market_ast(market_path)
    artifacts = _load_ast_artifacts(model_dir)

    alpha = float(artifacts.get("dispersion_alpha_mom", 0.0))

    merged = market_df.merge(
        pred_df,
        on="player_norm",
        how="inner",
        suffixes=("_market", "_pred"),
    )

    if merged.empty:
        raise ValueError("No AST market rows matched prediction rows by normalized player name.")

    # prefer prediction-side identity fields when present
    merged["player"] = merged.get("player_pred", merged.get("player_market"))
    merged["team"] = merged.get("team_pred", merged.get("team_market"))
    merged["opp"] = merged.get("opp_pred", merged.get("opp_market"))

    merged["pred_mean"] = _to_float(merged["pred_mean"]).clip(lower=0)
    merged["baseline_mean"] = _to_float(merged["baseline_mean"]).clip(lower=0)
    merged["line"] = _to_float(merged["line"])
    merged["minutes_proj"] = _to_float(merged["minutes_proj"])
    merged["is_eligible"] = _to_float(merged["is_eligible"]).fillna(0).astype(int)

    # probabilities
    mu = merged["pred_mean"].to_numpy(dtype=float)
    line = merged["line"].to_numpy(dtype=float)

    merged["p_over"] = _prob_over_pushable_line(mu, line, alpha=alpha)
    merged["p_under"] = _prob_under_pushable_line(mu, line, alpha=alpha)

    # line-relative mean edges
    merged["mean_edge"] = merged["pred_mean"] - merged["line"]
    merged["baseline_edge"] = merged["baseline_mean"] - merged["line"]
    merged["delta_vs_baseline_edge"] = merged["mean_edge"] - merged["baseline_edge"]

    # side-aware model probability
    if "side" in merged.columns:
        merged["model_prob"] = np.where(
            merged["side"].eq("over"),
            merged["p_over"],
            np.where(merged["side"].eq("under"), merged["p_under"], np.nan),
        )
    else:
        merged["model_prob"] = np.nan

    # implied probability from price if present
    merged["implied_prob"] = _american_to_implied_prob(merged["odds"]) if "odds" in merged.columns else np.nan

    # probability edge
    merged["prob_edge_over"] = merged["p_over"] - merged["implied_prob"]
    merged["prob_edge_under"] = merged["p_under"] - merged["implied_prob"]
    merged["prob_edge"] = np.where(
        merged["side"].eq("over"),
        merged["prob_edge_over"],
        np.where(merged["side"].eq("under"), merged["prob_edge_under"], np.nan),
    )

    # expected value style edge in price terms if market odds exist
    # EV on 1-unit risk for American odds
    payout = np.where(
        merged["odds"] > 0,
        merged["odds"] / 100.0,
        np.where(merged["odds"] < 0, 100.0 / (-merged["odds"]), np.nan),
    )
    merged["ev_over_1u"] = merged["p_over"] * payout - (1.0 - merged["p_over"])
    merged["ev_under_1u"] = merged["p_under"] * payout - (1.0 - merged["p_under"])
    merged["ev_1u"] = np.where(
        merged["side"].eq("over"),
        merged["ev_over_1u"],
        np.where(merged["side"].eq("under"), merged["ev_under_1u"], np.nan),
    )

    # hidden edge: when mean edge is strong but probability edge looks modest
    merged["abs_mean_edge"] = merged["mean_edge"].abs()
    merged["abs_prob_edge"] = merged["prob_edge"].abs()
    merged["hidden_mean_signal"] = np.where(
        merged["abs_mean_edge"] >= 1.0,
        merged["abs_mean_edge"] - merged["abs_prob_edge"].fillna(0.0),
        0.0,
    )

    # ranking score
    merged["edge_score"] = (
        0.45 * merged["prob_edge"].fillna(0.0)
        + 0.30 * merged["ev_1u"].fillna(0.0)
        + 0.20 * merged["mean_edge"] * np.where(merged["side"].eq("under"), -1.0, 1.0)
        + 0.05 * merged["hidden_mean_signal"].fillna(0.0)
    )

    if eligible_only:
        merged = merged[merged["is_eligible"] == 1].copy()

    if min_prob_edge > 0:
        merged = merged[merged["prob_edge"].abs().fillna(0.0) >= min_prob_edge].copy()

    keep_cols = [
        "player",
        "team",
        "opp",
        "line_source",
        "stat",
        "side",
        "line",
        "odds",
        "implied_prob",
        "pred_mean",
        "baseline_mean",
        "mean_edge",
        "baseline_edge",
        "delta_vs_baseline_edge",
        "p_over",
        "p_under",
        "model_prob",
        "prob_edge_over",
        "prob_edge_under",
        "prob_edge",
        "ev_over_1u",
        "ev_under_1u",
        "ev_1u",
        "hidden_mean_signal",
        "minutes_proj",
        "is_eligible",
        "eligibility_reason",
        "edge_score",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]

    out_df = merged[keep_cols].copy()

    # sort strongest first
    sort_cols = ["edge_score", "prob_edge", "ev_1u", "hidden_mean_signal"]
    sort_cols = [c for c in sort_cols if c in out_df.columns]
    out_df = out_df.sort_values(sort_cols, ascending=False, kind="mergesort").reset_index(drop=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AST edge table from predictions + market lines.")
    parser.add_argument("--pred-path", required=True, help="Path to pred_ast.csv")
    parser.add_argument("--market-path", required=True, help="Path to AST market CSV")
    parser.add_argument("--model-dir", default=str(ASSISTS_MODEL_DIR), help="Model dir containing ast_artifacts.json")
    parser.add_argument("--out-path", default=None, help="Where to save ranked AST edge CSV")
    parser.add_argument("--min-prob-edge", type=float, default=0.0, help="Filter to abs(prob_edge) >= this value")
    parser.add_argument("--eligible-only", action="store_true", help="Keep only model-eligible players")

    args = parser.parse_args()

    out_df = build_ast_edge_table(
        pred_path=args.pred_path,
        market_path=args.market_path,
        model_dir=args.model_dir,
        out_path=args.out_path,
        min_prob_edge=args.min_prob_edge,
        eligible_only=args.eligible_only,
    )

    print(f"[AST EDGE] Rows: {len(out_df)}")
    if len(out_df):
        print(out_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()