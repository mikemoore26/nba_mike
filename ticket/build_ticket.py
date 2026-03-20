from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PRED_FILES = [
    "pred_ast.csv",
    "pred_reb.csv",
    "pred_fg3.csv",
    "pred_pts.csv",
]


@dataclass(frozen=True)
class TicketConfig:
    ticket_name: str
    n_legs: int
    score_col: str
    max_per_team: int = 2
    max_per_stat: int = 2
    max_per_game: int = 3
    min_minutes: float = 0.0


SAFE_CONFIG = TicketConfig(
    ticket_name="safe",
    n_legs=4,
    score_col="score_safe",
    max_per_team=1,
    max_per_stat=1,
    max_per_game=2,
    min_minutes=22.0,
)

BALANCED_CONFIG = TicketConfig(
    ticket_name="balanced",
    n_legs=6,
    score_col="score_balanced",
    max_per_team=2,
    max_per_stat=2,
    max_per_game=3,
    min_minutes=16.0,
)

LOTTO_CONFIG = TicketConfig(
    ticket_name="lotto",
    n_legs=8,
    score_col="score_lotto",
    max_per_team=2,
    max_per_stat=3,
    max_per_game=3,
    min_minutes=12.0,
)


def _load_projection_files(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)

    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for fname in PRED_FILES:
        path = results_dir / fname
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            missing.append(fname)

    if not frames:
        raise FileNotFoundError(
            f"No projection files found in {results_dir}. "
            f"Expected one or more of: {PRED_FILES}"
        )

    if missing:
        print(f"[WARN] Missing projection files: {missing}")

    out = pd.concat(frames, ignore_index=True)
    return out


def _safe_num(series: pd.Series, fill: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill)


def _ensure_game_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    team = out["team"].astype(str)
    opp = out["opp"].astype(str)
    out["game_key"] = team.where(team < opp, opp) + "_vs_" + opp.where(team < opp, team)
    return out


def _projection_strength(df: pd.DataFrame) -> pd.Series:
    """
    Cross-stat normalized raw projection strength.
    We normalize within stat because 25 points and 4 threes are different units.
    """
    out = pd.Series(index=df.index, dtype=float)

    for stat, idx in df.groupby("stat").groups.items():
        s = _safe_num(df.loc[idx, "pred_mean"])
        mean = s.mean()
        std = s.std(ddof=0)

        if pd.isna(std) or std < 1e-9:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (s - mean) / std

    return out.fillna(0.0)


def _delta_strength(df: pd.DataFrame) -> pd.Series:
    """
    How much the model is above its own baseline.
    Also normalized within stat.
    """
    out = pd.Series(index=df.index, dtype=float)

    for stat, idx in df.groupby("stat").groups.items():
        s = _safe_num(df.loc[idx, "delta_mean"])
        mean = s.mean()
        std = s.std(ddof=0)

        if pd.isna(std) or std < 1e-9:
            out.loc[idx] = 0.0
        else:
            out.loc[idx] = (s - mean) / std

    return out.fillna(0.0)


def _minutes_confidence(df: pd.DataFrame) -> pd.Series:
    mins = _safe_num(df["minutes_proj"])
    return (mins / 36.0).clip(lower=0.0, upper=1.0)


def _stat_volatility_penalty(df: pd.DataFrame) -> pd.Series:
    """
    Slight penalty so safe ticket doesn't become all fragile categories.
    """
    stat = df["stat"].astype(str).str.lower()
    penalty = pd.Series(0.0, index=df.index, dtype=float)

    penalty.loc[stat.eq("fg3m")] = 0.10
    penalty.loc[stat.eq("pts")] = 0.06
    penalty.loc[stat.eq("ast")] = 0.04
    penalty.loc[stat.eq("reb")] = 0.03

    return penalty


def score_projection_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores one-row-per-player-per-stat projections.
    """
    required = [
        "game_date",
        "player",
        "team",
        "opp",
        "stat",
        "pred_mean",
        "baseline_mean",
        "delta_mean",
        "minutes_proj",
        "is_eligible",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Projection pool missing required columns: {missing}")

    out = df.copy()
    out = _ensure_game_key(out)

    out["pred_mean"] = _safe_num(out["pred_mean"])
    out["baseline_mean"] = _safe_num(out["baseline_mean"])
    out["delta_mean"] = _safe_num(out["delta_mean"])
    out["minutes_proj"] = _safe_num(out["minutes_proj"])
    out["is_eligible"] = _safe_num(out["is_eligible"]).astype(int)

    out["proj_strength"] = _projection_strength(out)
    out["delta_strength"] = _delta_strength(out)
    out["minutes_conf"] = _minutes_confidence(out)
    out["stat_vol_penalty"] = _stat_volatility_penalty(out)

    # SAFE: stability first
    out["score_safe"] = (
        0.95 * out["minutes_conf"]
        + 0.65 * out["proj_strength"]
        + 0.35 * out["delta_strength"]
        - 0.40 * out["stat_vol_penalty"]
    )

    # BALANCED: blend projection and delta
    out["score_balanced"] = (
        0.55 * out["minutes_conf"]
        + 0.70 * out["proj_strength"]
        + 0.75 * out["delta_strength"]
        - 0.20 * out["stat_vol_penalty"]
    )

    # LOTTO: ceiling + delta
    out["score_lotto"] = (
        0.25 * out["minutes_conf"]
        + 0.90 * out["proj_strength"]
        + 1.05 * out["delta_strength"]
        - 0.10 * out["stat_vol_penalty"]
    )

    return out


def _candidate_pool(df: pd.DataFrame, cfg: TicketConfig) -> pd.DataFrame:
    out = df.copy()
    out = out[out["is_eligible"] == 1].copy()
    out = out[out["minutes_proj"] >= cfg.min_minutes].copy()

    # one row per player/stat
    out = out.sort_values(cfg.score_col, ascending=False, kind="mergesort")
    out = out.drop_duplicates(subset=["game_date", "player", "stat"], keep="first").reset_index(drop=True)

    return out


def _can_add_projection(
    row: pd.Series,
    chosen: list[pd.Series],
    *,
    cfg: TicketConfig,
) -> bool:
    if not chosen:
        return True

    chosen_df = pd.DataFrame(chosen)

    # max 1 pick per player
    if (chosen_df["player"] == row["player"]).any():
        return False

    # max per team
    if (chosen_df["team"] == row["team"]).sum() >= cfg.max_per_team:
        return False

    # max per stat
    if (chosen_df["stat"] == row["stat"]).sum() >= cfg.max_per_stat:
        return False

    # max per game
    if (chosen_df["game_key"] == row["game_key"]).sum() >= cfg.max_per_game:
        return False

    return True


def build_ticket_from_projections(
    scored_df: pd.DataFrame,
    *,
    cfg: TicketConfig,
) -> pd.DataFrame:
    pool = _candidate_pool(scored_df, cfg)

    chosen: list[pd.Series] = []

    for _, row in pool.iterrows():
        if _can_add_projection(row, chosen, cfg=cfg):
            chosen.append(row)

        if len(chosen) >= cfg.n_legs:
            break

    if not chosen:
        return pd.DataFrame(columns=scored_df.columns)

    out = pd.DataFrame(chosen).reset_index(drop=True)
    out["ticket_name"] = cfg.ticket_name
    out = out.sort_values(cfg.score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    return out


def _ticket_summary(ticket_df: pd.DataFrame, cfg: TicketConfig) -> dict:
    if ticket_df.empty:
        return {
            "ticket_name": cfg.ticket_name,
            "n_legs": 0,
            "avg_pred_mean": 0.0,
            "avg_delta_mean": 0.0,
            "avg_minutes_proj": 0.0,
            "sum_score": 0.0,
        }

    return {
        "ticket_name": cfg.ticket_name,
        "n_legs": int(len(ticket_df)),
        "avg_pred_mean": float(_safe_num(ticket_df["pred_mean"]).mean()),
        "avg_delta_mean": float(_safe_num(ticket_df["delta_mean"]).mean()),
        "avg_minutes_proj": float(_safe_num(ticket_df["minutes_proj"]).mean()),
        "sum_score": float(_safe_num(ticket_df[cfg.score_col]).sum()),
    }


def build_all_tickets(
    *,
    results_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_df = _load_projection_files(results_dir)
    scored_df = score_projection_pool(pred_df)

    safe_ticket = build_ticket_from_projections(scored_df, cfg=SAFE_CONFIG)
    balanced_ticket = build_ticket_from_projections(scored_df, cfg=BALANCED_CONFIG)
    lotto_ticket = build_ticket_from_projections(scored_df, cfg=LOTTO_CONFIG)

    scored_path = results_dir / "scored_projections.csv"
    safe_path = results_dir / "ticket_safe.csv"
    balanced_path = results_dir / "ticket_balanced.csv"
    lotto_path = results_dir / "ticket_lotto.csv"
    summary_path = results_dir / "ticket_summary.csv"

    scored_df.to_csv(scored_path, index=False)
    safe_ticket.to_csv(safe_path, index=False)
    balanced_ticket.to_csv(balanced_path, index=False)
    lotto_ticket.to_csv(lotto_path, index=False)

    summary_df = pd.DataFrame(
        [
            _ticket_summary(safe_ticket, SAFE_CONFIG),
            _ticket_summary(balanced_ticket, BALANCED_CONFIG),
            _ticket_summary(lotto_ticket, LOTTO_CONFIG),
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    return {
        "scored_projections": scored_df,
        "ticket_safe": safe_ticket,
        "ticket_balanced": balanced_ticket,
        "ticket_lotto": lotto_ticket,
        "ticket_summary": summary_df,
    }