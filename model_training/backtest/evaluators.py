#model_training/backtest/evaluators.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _rmse(x: pd.Series) -> float:
    if len(x) == 0:
        return np.nan
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(x))))


def make_minutes_bucket(minutes: pd.Series) -> pd.Series:
    bins = [-np.inf, 16, 24, 30, 36, np.inf]
    labels = ["<=16", "17-24", "25-30", "31-36", "37+"]
    return pd.cut(minutes, bins=bins, labels=labels)


def make_pred_bucket(pred: pd.Series, stat: str) -> pd.Series:
    stat = str(stat).lower()

    if stat == "pts":
        bins = [-np.inf, 10, 15, 20, 25, 30, np.inf]
        labels = ["<=10", "11-15", "16-20", "21-25", "26-30", "31+"]
    elif stat == "reb":
        bins = [-np.inf, 4, 6, 8, 10, 12, np.inf]
        labels = ["<=4", "5-6", "7-8", "9-10", "11-12", "13+"]
    elif stat == "ast":
        bins = [-np.inf, 2, 4, 6, 8, 10, np.inf]
        labels = ["<=2", "3-4", "5-6", "7-8", "9-10", "11+"]
    else:
        bins = [-np.inf, 1, 2, 3, 4, 5, np.inf]
        labels = ["<=1", "2", "3", "4", "5", "6+"]

    return pd.cut(pred, bins=bins, labels=labels)


def enrich_player_eval(player_eval: pd.DataFrame) -> pd.DataFrame:
    df = player_eval.copy()

    if "minutes_proj" in df.columns:
        df["minutes_proj"] = pd.to_numeric(df["minutes_proj"], errors="coerce")
    else:
        df["minutes_proj"] = np.nan

    df["pred_mean"] = pd.to_numeric(df["pred_mean"], errors="coerce")
    df["minutes_bucket"] = make_minutes_bucket(df["minutes_proj"])

    df["pred_bucket"] = df.apply(
        lambda r: make_pred_bucket(pd.Series([r["pred_mean"]]), r["stat"]).iloc[0],
        axis=1,
    )

    return df


def summarize_player_eval(player_eval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = enrich_player_eval(player_eval)
    df = df.dropna(subset=["pred_mean", "actual_value"]).copy()

    by_stat = (
        df.groupby("stat", dropna=False)
        .agg(
            n=("actual_value", "size"),
            mean_pred=("pred_mean", "mean"),
            mean_actual=("actual_value", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            rmse=("error", _rmse),
            beat_projection_rate=("beat_projection", "mean"),
            beat_baseline_rate=("beat_baseline", "mean"),
        )
        .reset_index()
        .sort_values("stat")
    )

    by_bucket = (
        df.groupby(["stat", "minutes_bucket", "pred_bucket"], dropna=False, observed=False)
        .agg(
            n=("actual_value", "size"),
            mean_pred=("pred_mean", "mean"),
            mean_actual=("actual_value", "mean"),
            mae=("abs_error", "mean"),
            bias=("error", "mean"),
            rmse=("error", _rmse),
            beat_projection_rate=("beat_projection", "mean"),
            beat_baseline_rate=("beat_baseline", "mean"),
        )
        .reset_index()
        .sort_values(["stat", "minutes_bucket", "pred_bucket"])
    )

    return by_stat, by_bucket


def _normalize_ticket_legs(ticket_legs: pd.DataFrame) -> pd.DataFrame:
    df = ticket_legs.copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["player"] = df["player"].astype(str).str.strip()

    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.strip().str.upper()
    else:
        df["team"] = ""

    df["stat"] = df["stat"].astype(str).str.strip().str.lower()

    if "side" in df.columns:
        df["side"] = df["side"].astype(str).str.strip().str.lower()

    if "line" in df.columns:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")

    if "ticket_type" not in df.columns:
        df["ticket_type"] = "unknown"

    if "ticket_id" not in df.columns:
        # stable enough for backtest grouping
        df["ticket_id"] = (
            df["ticket_type"].astype(str)
            + "_"
            + df.groupby(["game_date", "ticket_type"]).cumcount().add(1).astype(str)
        )

    return df


def _normalize_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    act = actuals.copy()

    act["game_date"] = pd.to_datetime(act["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    act["player"] = act["player"].astype(str).str.strip()

    if "team" in act.columns:
        act["team"] = act["team"].astype(str).str.strip().str.upper()
    else:
        act["team"] = ""

    return act


def _coalesce_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), vals)
    return out


def _map_actual_value(merged: pd.DataFrame) -> pd.Series:
    pts = _coalesce_series(merged, ["pts"])
    reb = _coalesce_series(merged, ["reb", "trb"])
    ast = _coalesce_series(merged, ["ast"])
    fg3 = _coalesce_series(merged, ["fg3m", "fg3"])

    actual_value = pd.Series(np.nan, index=merged.index, dtype="float64")

    stat = merged["stat"].astype(str).str.lower()
    actual_value = actual_value.where(stat != "pts", pts)
    actual_value = actual_value.where(stat != "reb", reb)
    actual_value = actual_value.where(stat != "ast", ast)
    actual_value = actual_value.where(stat != "fg3", fg3)

    return actual_value


def _grade_leg_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["has_actual"] = out["actual_value"].notna()
    out["has_line"] = out["line"].notna()
    out["has_side"] = out["side"].isin(["over", "under"])
    out["is_evaluable_leg"] = out["has_actual"] & out["has_line"] & out["has_side"]

    out["leg_result"] = pd.Series(pd.NA, index=out.index, dtype="object")

    over_mask = out["is_evaluable_leg"] & (out["side"] == "over")
    under_mask = out["is_evaluable_leg"] & (out["side"] == "under")
    push_mask = out["is_evaluable_leg"] & (out["actual_value"] == out["line"])

    out.loc[over_mask, "leg_result"] = np.where(
        out.loc[over_mask, "actual_value"] > out.loc[over_mask, "line"],
        "win",
        "loss",
    )

    out.loc[under_mask, "leg_result"] = np.where(
        out.loc[under_mask, "actual_value"] < out.loc[under_mask, "line"],
        "win",
        "loss",
    )

    out.loc[push_mask, "leg_result"] = "push"

    out["leg_win"] = out["leg_result"].eq("win")
    out["leg_loss"] = out["leg_result"].eq("loss")
    out["leg_push"] = out["leg_result"].eq("push")

    return out


def evaluate_ticket_frames(
    ticket_legs: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate ticket legs against realized box score actuals.

    Expected ticket_legs columns:
        game_date, player, team, stat, line, side, ticket_type
    Optional:
        ticket_id, pred_mean, p_hit, baseline_mean, minutes_proj, etc.
    """
    if ticket_legs is None or ticket_legs.empty:
        return pd.DataFrame()

    if actuals is None or actuals.empty:
        return pd.DataFrame()

    df = _normalize_ticket_legs(ticket_legs)
    act = _normalize_actuals(actuals)

    required_cols = ["game_date", "player", "team", "stat", "line", "side", "ticket_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return pd.DataFrame()

    merged = df.merge(
        act,
        on=["game_date", "player", "team"],
        how="left",
        suffixes=("", "_actual"),
    )

    merged["actual_value"] = _map_actual_value(merged)
    merged = _grade_leg_results(merged)

    # drop tickets with zero evaluable legs
    leg_eval = merged[merged["is_evaluable_leg"]].copy()
    if leg_eval.empty:
        return pd.DataFrame()

    if "pred_mean" in leg_eval.columns:
        leg_eval["pred_mean"] = pd.to_numeric(leg_eval["pred_mean"], errors="coerce")
    else:
        leg_eval["pred_mean"] = np.nan

    leg_eval["line_minus_pred"] = leg_eval["line"] - leg_eval["pred_mean"]
    leg_eval["pred_minus_line"] = leg_eval["pred_mean"] - leg_eval["line"]
    leg_eval["edge_vs_line"] = np.where(
        leg_eval["side"] == "over",
        leg_eval["pred_mean"] - leg_eval["line"],
        leg_eval["line"] - leg_eval["pred_mean"],
    )

    ticket_summary = (
        leg_eval.groupby(["game_date", "ticket_type", "ticket_id"], dropna=False)
        .agg(
            legs=("player", "size"),
            win_legs=("leg_win", "sum"),
            loss_legs=("leg_loss", "sum"),
            push_legs=("leg_push", "sum"),
            avg_line=("line", "mean"),
            avg_actual=("actual_value", "mean"),
            avg_pred=("pred_mean", "mean"),
            avg_edge_vs_line=("edge_vs_line", "mean"),
        )
        .reset_index()
    )

    ticket_summary["all_legs_win"] = ticket_summary["loss_legs"].eq(0) & ticket_summary["win_legs"].eq(ticket_summary["legs"])
    ticket_summary["has_any_loss"] = ticket_summary["loss_legs"] > 0
    ticket_summary["all_push"] = ticket_summary["push_legs"].eq(ticket_summary["legs"])

    ticket_summary["ticket_result"] = np.select(
        [
            ticket_summary["all_legs_win"],
            ticket_summary["all_push"],
            ticket_summary["has_any_loss"],
        ],
        [
            "win",
            "push",
            "loss",
        ],
        default="partial",
    )

    ticket_summary["ticket_win"] = ticket_summary["ticket_result"].eq("win")
    ticket_summary["ticket_loss"] = ticket_summary["ticket_result"].eq("loss")
    ticket_summary["ticket_push"] = ticket_summary["ticket_result"].eq("push")

    return ticket_summary.sort_values(["game_date", "ticket_type", "ticket_id"]).reset_index(drop=True)


def summarize_ticket_eval(ticket_eval: pd.DataFrame) -> pd.DataFrame:
    if ticket_eval.empty:
        return pd.DataFrame()

    return (
        ticket_eval.groupby("ticket_type", dropna=False)
        .agg(
            n_tickets=("ticket_id", "size"),
            avg_legs=("legs", "mean"),
            ticket_win_rate=("ticket_win", "mean"),
            ticket_loss_rate=("ticket_loss", "mean"),
            ticket_push_rate=("ticket_push", "mean"),
            avg_win_legs=("win_legs", "mean"),
            avg_loss_legs=("loss_legs", "mean"),
            avg_push_legs=("push_legs", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_actual=("avg_actual", "mean"),
            avg_edge_vs_line=("avg_edge_vs_line", "mean"),
        )
        .reset_index()
        .sort_values("ticket_type")
    )