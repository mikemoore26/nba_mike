# model_training/rebounds/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


REB_FEATURES = [
    "min_rolling_5",
    "min_rolling_10",
    "min_delta_5",
    "reb_rolling_5",
    "reb_rolling_10",
    "reb_delta_5",
    "reb_var_10",
    "player_reb_season_avg",
    "player_min_season_avg",
    "home_game",
    "days_rest",
    "back_to_back",
]

OPTIONAL_REB_FEATURES = [
    "team_pace_to_date",
    "opp_pace_to_date",
    "team_fg_pct_to_date",
    "opp_fg_pct_allowed_to_date",
    "opp_reb_allowed_to_date",
]


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


def _grp_roll_mean(series: pd.Series, window: int, minp: int):
    return (
        series.groupby(level=0, sort=False)
        .rolling(window, min_periods=minp)
        .mean()
        .droplevel(0)
    )


def _grp_roll_var(series: pd.Series, window: int, minp: int):
    return (
        series.groupby(level=0, sort=False)
        .rolling(window, min_periods=minp)
        .var()
        .droplevel(0)
    )


def _parse_minutes_to_float(s: pd.Series) -> pd.Series:
    """
    Handles common log formats:
      - numeric strings: "34.5"
      - clock strings: "34:12" -> 34.2
      - DNP-ish strings -> NaN
    """
    x = s.astype("string").str.strip()

    # DNP / inactive / not played variants -> NaN
    dnp_like = x.str.contains(
        r"(did not play|dnp|inactive|didn't play|not with team|suspended|injured|coach's decision)",
        case=False,
        na=False,
        regex=True,
    )
    x = x.mask(dnp_like, other=pd.NA)

    # MM:SS -> minutes float
    has_colon = x.str.contains(":", na=False)
    mmss = x.where(has_colon)

    mins = pd.to_numeric(x.where(~has_colon), errors="coerce")

    if has_colon.any():
        parts = mmss.str.split(":", n=1, expand=True)
        mm = pd.to_numeric(parts[0], errors="coerce")
        ss = pd.to_numeric(parts[1], errors="coerce")
        mmss_mins = mm + (ss / 60.0)
        mins = mins.where(~has_colon, mmss_mins)

    return mins.astype(float)


def build_features_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- enforce date ---
    if "date" not in out.columns and "game_date" in out.columns:
        out["date"] = out["game_date"]

    _require_cols(out, ["player", "date"], "BASE")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "player"]).copy()
    out = out.sort_values(["player", "date"]).reset_index(drop=True)

    # --- canonical mapping ---
    # Minutes -> min
    if "min" not in out.columns:
        if "mp_minutes" in out.columns:
            out["min"] = out["mp_minutes"]
        elif "minutes" in out.columns:
            out["min"] = out["minutes"]
        elif "mp" in out.columns:
            out["min"] = out["mp"]

    # Rebounds -> reb
    if "reb" not in out.columns:
        if "trb" in out.columns:
            out["reb"] = out["trb"]
        elif "rebounds" in out.columns:
            out["reb"] = out["rebounds"]
        elif "total_rebounds" in out.columns:
            out["reb"] = out["total_rebounds"]

    _require_cols(out, ["min", "reb"], "REB_TARGETS")

    # --- FORCE numeric with robust minutes parsing ---
    out["min"] = _parse_minutes_to_float(out["min"])
    out["reb"] = pd.to_numeric(out["reb"], errors="coerce")

    # --- group rolling (no boundary bleed) ---
    out["_pidx"] = out["player"].astype("string")
    out = out.set_index(["_pidx", out.index])

    min_l1 = out.groupby(level=0)["min"].shift(1)
    reb_l1 = out.groupby(level=0)["reb"].shift(1)

    out["min_rolling_5"] = _grp_roll_mean(min_l1, 5, 1)
    out["min_rolling_10"] = _grp_roll_mean(min_l1, 10, 1)

    out["reb_rolling_5"] = _grp_roll_mean(reb_l1, 5, 1)
    out["reb_rolling_10"] = _grp_roll_mean(reb_l1, 10, 1)

    out["min_delta_5"] = out["min_rolling_5"] - out["min_rolling_10"]
    out["reb_delta_5"] = out["reb_rolling_5"] - out["reb_rolling_10"]

    out["reb_var_10"] = _grp_roll_var(reb_l1, 10, 3)

    out = out.reset_index(level=0, drop=True)

    # --- home/rest flags ---
    if "home_game" not in out.columns and "is_home" in out.columns:
        out["home_game"] = out["is_home"].astype(int)

    if "days_rest" not in out.columns:
        prev_date = out.groupby("player")["date"].shift(1)
        out["days_rest"] = (out["date"] - prev_date).dt.days.clip(lower=0)

    if "back_to_back" not in out.columns:
        out["back_to_back"] = (out["days_rest"] <= 1).astype(int)

    out = add_player_baselines(out)
    return out


def add_player_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = ["player", "season"] if "season" in out.columns else ["player"]
    out = out.sort_values(keys + ["date"]).reset_index(drop=True)

    g = out.groupby(keys, sort=False)

    out["player_reb_season_avg"] = (
        g["reb"]
        .apply(lambda s: s.shift(1).expanding(min_periods=5).mean())
        .reset_index(level=list(range(len(keys))), drop=True)
    )

    out["player_min_season_avg"] = (
        g["min"]
        .apply(lambda s: s.shift(1).expanding(min_periods=5).mean())
        .reset_index(level=list(range(len(keys))), drop=True)
    )

    return out
