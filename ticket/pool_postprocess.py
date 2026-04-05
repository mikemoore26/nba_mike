# ticket/pool_postprocess.py
from __future__ import annotations

import pandas as pd


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def select_best_line_per_player_stat(
    df: pd.DataFrame,
    *,
    score_col: str = "score_balanced",
) -> pd.DataFrame:
    """
    Collapse multiple candidate lines/sides for the same player+stat
    down to the single best betting expression.

    Why:
    - avoids duplicate exposure (same player/stat appearing 2-3 times)
    - forces the system to choose its strongest line
    - improves diversity in balanced/lotto tickets
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    required = ["player", "stat"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"select_best_line_per_player_stat missing required columns: {missing}"
        )

    if score_col not in out.columns:
        raise ValueError(
            f"select_best_line_per_player_stat expected score column '{score_col}'"
        )

    if "p_hit" not in out.columns:
        out["p_hit"] = 0.0
    if "edge_raw" not in out.columns:
        out["edge_raw"] = 0.0
    if "minutes_proj" not in out.columns:
        out["minutes_proj"] = 0.0

    out["player"] = out["player"].astype(str).str.strip()
    out["stat"] = out["stat"].astype(str).str.strip().str.lower()

    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(-999.0)
    out["p_hit"] = _safe_num(out, "p_hit", default=0.0)
    out["edge_raw"] = _safe_num(out, "edge_raw", default=0.0)
    out["minutes_proj"] = _safe_num(out, "minutes_proj", default=0.0)

    # strongest candidate first within each player/stat
    out = out.sort_values(
        ["player", "stat", score_col, "p_hit", "edge_raw", "minutes_proj"],
        ascending=[True, True, False, False, False, False],
    )

    out = (
        out.groupby(["player", "stat"], as_index=False, dropna=False)
        .head(1)
        .reset_index(drop=True)
    )

    return out


def filter_scored_legs(
    df: pd.DataFrame,
    *,
    min_p_hit: float = 0.55,
    min_edge_raw: float = 0.50,
    require_any_ticket_flag: bool = True,
) -> pd.DataFrame:
    """
    Remove weak legs before ranking/ticket construction.

    Default thresholds are intentionally moderate:
    - p_hit >= 0.55
    - edge_raw >= 0.50

    These are strong enough to remove obvious noise while preserving
    real candidate volume for safe/balanced/lotto.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    if "p_hit" not in out.columns:
        raise ValueError("filter_scored_legs expected column 'p_hit'")
    if "edge_raw" not in out.columns:
        raise ValueError("filter_scored_legs expected column 'edge_raw'")

    out["p_hit"] = pd.to_numeric(out["p_hit"], errors="coerce")
    out["edge_raw"] = pd.to_numeric(out["edge_raw"], errors="coerce")

    keep = (
        out["p_hit"].ge(min_p_hit)
        & out["edge_raw"].ge(min_edge_raw)
    )

    if require_any_ticket_flag:
        ticket_flag_cols = [c for c in ["can_safe", "can_balanced", "can_lotto"] if c in out.columns]
        if ticket_flag_cols:
            any_flag = out[ticket_flag_cols].apply(
                lambda r: any(pd.to_numeric(r, errors="coerce").fillna(0).astype(int) == 1),
                axis=1,
            )
            keep = keep & any_flag

    out = out.loc[keep].copy()

    return out.reset_index(drop=True)


def build_curated_scored_pool(
    df: pd.DataFrame,
    *,
    score_col: str = "score_balanced",
    min_p_hit: float = 0.55,
    min_edge_raw: float = 0.50,
) -> pd.DataFrame:
    """
    Recommended sequence before ranking/ticket building:

    scored legs
      -> hard filter
      -> best line per player/stat
    """
    out = filter_scored_legs(
        df,
        min_p_hit=min_p_hit,
        min_edge_raw=min_edge_raw,
        require_any_ticket_flag=True,
    )

    if out.empty:
        return out

    out = select_best_line_per_player_stat(
        out,
        score_col=score_col,
    )

    return out.reset_index(drop=True)