from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# Helpers
# =========================================================

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_numeric(df: pd.DataFrame, col: str, default=np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _ensure_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")
    return df[col].fillna(default).astype(bool)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}")


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def _rmse(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(vals))))


# =========================================================
# Loaders
# =========================================================

def load_player_eval(results_root: Path) -> pd.DataFrame:
    candidates = [
        results_root / "backtest_player_eval.csv",
        results_root / "backtest_player_eval_raw.csv",
        results_root / "backtest_player_eval_bettable.csv",
    ]

    for p in candidates:
        df = _safe_read_csv(p)
        if not df.empty:
            return df

    # fallback: aggregate per-date files
    dfs = []
    for p in sorted(results_root.glob("*/player_eval_scored.csv")):
        df = _safe_read_csv(p)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_ticket_eval(results_root: Path) -> pd.DataFrame:
    p = results_root / "backtest_ticket_eval.csv"
    df = _safe_read_csv(p)
    if not df.empty:
        return df

    dfs = []
    for p in sorted(results_root.glob("*/backtest_ticket_eval.csv")):
        df = _safe_read_csv(p)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# Player analysis
# =========================================================

def _normalize_player_eval(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str).str.strip().str.lower()

    out["pred_mean"] = _safe_numeric(out, "pred_mean")
    out["baseline_mean"] = _safe_numeric(out, "baseline_mean")
    out["actual_value"] = _safe_numeric(out, "actual_value")
    out["minutes_proj"] = _safe_numeric(out, "minutes_proj")

    if "error" not in out.columns:
        out["error"] = out["actual_value"] - out["pred_mean"]
    else:
        out["error"] = _safe_numeric(out, "error")

    if "abs_error" not in out.columns:
        out["abs_error"] = out["error"].abs()
    else:
        out["abs_error"] = _safe_numeric(out, "abs_error")

    if "beat_projection" not in out.columns:
        out["beat_projection"] = out["actual_value"] > out["pred_mean"]
    else:
        out["beat_projection"] = _ensure_bool(out, "beat_projection")

    if "beat_baseline" not in out.columns:
        out["beat_baseline"] = out["actual_value"] > out["baseline_mean"]
    else:
        out["beat_baseline"] = _ensure_bool(out, "beat_baseline")

    return out


def build_player_overview(player_eval: pd.DataFrame) -> pd.DataFrame:
    if player_eval.empty:
        return pd.DataFrame()

    df = _normalize_player_eval(player_eval)
    df = df.dropna(subset=["stat", "pred_mean", "actual_value"]).copy()

    if df.empty:
        return pd.DataFrame()

    out = (
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

    return out


def _pred_bucket(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    if g["pred_mean"].notna().sum() == 0:
        g["pred_bucket"] = pd.NA
        return g

    try:
        g["pred_bucket"] = pd.qcut(
            g["pred_mean"],
            q=min(5, g["pred_mean"].nunique()),
            duplicates="drop",
        )
    except Exception:
        g["pred_bucket"] = pd.NA

    return g


def build_player_bucket_overview(player_eval: pd.DataFrame) -> pd.DataFrame:
    if player_eval.empty:
        return pd.DataFrame()

    df = _normalize_player_eval(player_eval)
    df = df.dropna(subset=["stat", "pred_mean", "actual_value"]).copy()
    if df.empty:
        return pd.DataFrame()

    # avoids the deprecated groupby.apply-on-grouping-columns behavior
    parts = []
    for stat, g in df.groupby("stat", dropna=False):
        gg = _pred_bucket(g)
        gg["stat"] = stat
        parts.append(gg)

    if not parts:
        return pd.DataFrame()

    bucketed = pd.concat(parts, ignore_index=True)
    bucketed = bucketed.dropna(subset=["pred_bucket"]).copy()

    if bucketed.empty:
        return pd.DataFrame()

    out = (
        bucketed.groupby(["stat", "pred_bucket"], dropna=False, observed=False)
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
        .sort_values(["stat", "pred_bucket"])
    )

    out["pred_bucket"] = out["pred_bucket"].astype(str)
    return out


def build_top_pick_overview(player_eval: pd.DataFrame) -> pd.DataFrame:
    if player_eval.empty:
        return pd.DataFrame()

    df = _normalize_player_eval(player_eval)
    df = df.dropna(subset=["stat", "pred_mean", "actual_value"]).copy()
    if df.empty:
        return pd.DataFrame()

    # choose ranking field from best available option
    rank_field = None
    for c in ["rank_score", "score", "p_hit", "pred_mean"]:
        if c in df.columns:
            rank_field = c
            break

    if rank_field is None:
        return pd.DataFrame()

    df[rank_field] = _safe_numeric(df, rank_field)
    df = df.dropna(subset=[rank_field]).copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for stat, g in df.groupby("stat", dropna=False):
        if len(g) < 10:
            continue

        g = g.copy()
        g["rank_pct"] = g[rank_field].rank(pct=True, method="average")

        for thresh in [0.90, 0.95]:
            sub = g.loc[g["rank_pct"] >= thresh].copy()
            if sub.empty:
                continue

            rows.append(
                {
                    "stat": stat,
                    "top_score_threshold": thresh,
                    "n": int(len(sub)),
                    "beat_projection_rate": float(sub["beat_projection"].mean()),
                    "avg_overperformance": float((sub["actual_value"] - sub["pred_mean"]).mean()),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["stat", "top_score_threshold"])


def build_stat_calibration_suggestions(player_overview: pd.DataFrame) -> pd.DataFrame:
    if player_overview.empty:
        return pd.DataFrame()

    df = player_overview.copy()
    if not {"stat", "mean_pred", "mean_actual"}.issubset(df.columns):
        return pd.DataFrame()

    df["mean_pred"] = pd.to_numeric(df["mean_pred"], errors="coerce")
    df["mean_actual"] = pd.to_numeric(df["mean_actual"], errors="coerce")

    df["additive_adj"] = df["mean_actual"] - df["mean_pred"]
    df["multiplier"] = df["mean_actual"] / df["mean_pred"].replace(0, np.nan)

    return df[["stat", "additive_adj", "multiplier"]].copy()


def build_top_bias_players(player_eval: pd.DataFrame, *, top_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if player_eval.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _normalize_player_eval(player_eval)
    df = df.dropna(subset=["player", "stat", "error"]).copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    g = (
        df.groupby(["player", "stat"], dropna=False)
        .agg(
            n=("error", "size"),
            mean_error=("error", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .reset_index()
    )

    g = g.loc[g["n"] >= 3].copy()
    if g.empty:
        return pd.DataFrame(), pd.DataFrame()

    over = g.sort_values("mean_error", ascending=False).head(top_n).reset_index(drop=True)
    under = g.sort_values("mean_error", ascending=True).head(top_n).reset_index(drop=True)
    return over, under


# =========================================================
# Ticket analysis
# =========================================================

def build_ticket_overview(ticket_eval: pd.DataFrame) -> pd.DataFrame:
    if ticket_eval is None or ticket_eval.empty:
        return pd.DataFrame()

    df = ticket_eval.copy()

    for col in ["ticket_win", "ticket_loss", "ticket_push"]:
        if col not in df.columns:
            df[col] = False

    defaults = {
        "legs": 0,
        "win_legs": 0,
        "loss_legs": 0,
        "push_legs": 0,
        "avg_line": np.nan,
        "avg_actual": np.nan,
        "avg_pred": np.nan,
        "avg_edge_vs_line": np.nan,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    for col in defaults:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    out = (
        df.groupby("ticket_type", dropna=False)
        .agg(
            n_tickets=("ticket_id", "size"),
            avg_legs=("legs", "mean"),
            ticket_win_rate=("ticket_win", "mean"),
            ticket_loss_rate=("ticket_loss", "mean"),
            ticket_push_rate=("ticket_push", "mean"),
            avg_win_legs=("win_legs", "mean"),
            avg_loss_legs=("loss_legs", "mean"),
            avg_push_legs=("push_legs", "mean"),
            avg_line=("avg_line", "mean"),
            avg_actual=("avg_actual", "mean"),
            avg_pred=("avg_pred", "mean"),
            avg_edge_vs_line=("avg_edge_vs_line", "mean"),
        )
        .reset_index()
        .sort_values("ticket_type")
    )

    out["avg_leg_win_rate"] = out["avg_win_legs"] / out["avg_legs"].replace(0, np.nan)
    out["avg_leg_loss_rate"] = out["avg_loss_legs"] / out["avg_legs"].replace(0, np.nan)
    out["avg_leg_push_rate"] = out["avg_push_legs"] / out["avg_legs"].replace(0, np.nan)

    return out


# =========================================================
# Charts
# =========================================================

def _maybe_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        print(f"[WARN] Skipping chart {path.name}: missing data")
        return

    plot_df = df[[x, y]].dropna()
    if plot_df.empty:
        print(f"[WARN] Skipping chart {path.name}: missing data")
        return

    plt.figure(figsize=(8, 4.5))
    plt.bar(plot_df[x].astype(str), plot_df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    _save_fig(path)


def _maybe_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plot_df = df[[x, y]].dropna() if not df.empty and x in df.columns and y in df.columns else pd.DataFrame()
    if plot_df.empty:
        print(f"[WARN] Skipping chart {path.name}: missing data")
        return

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(plot_df[x], plot_df[y], alpha=0.4)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    _save_fig(path)


def make_player_charts(player_overview: pd.DataFrame, player_eval: pd.DataFrame, analysis_dir: Path) -> None:
    _maybe_bar_plot(
        player_overview,
        "stat",
        "mae",
        "Player MAE by Stat",
        analysis_dir / "player_mae_by_stat.png",
    )
    _maybe_bar_plot(
        player_overview,
        "stat",
        "bias",
        "Player Bias by Stat",
        analysis_dir / "player_bias_by_stat.png",
    )
    _maybe_bar_plot(
        player_overview,
        "stat",
        "beat_projection_rate",
        "Player Beat Projection Rate by Stat",
        analysis_dir / "player_beat_projection_rate_by_stat.png",
    )

    if not player_overview.empty and {"stat", "mean_pred", "mean_actual"}.issubset(player_overview.columns):
        plot_df = player_overview[["stat", "mean_pred", "mean_actual"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(8, 4.5))
            x = np.arange(len(plot_df))
            width = 0.35
            plt.bar(x - width / 2, plot_df["mean_pred"], width=width, label="mean_pred")
            plt.bar(x + width / 2, plot_df["mean_actual"], width=width, label="mean_actual")
            plt.xticks(x, plot_df["stat"].astype(str))
            plt.title("Player Mean Pred vs Actual by Stat")
            plt.xlabel("stat")
            plt.ylabel("value")
            plt.legend()
            _save_fig(analysis_dir / "player_mean_pred_vs_actual_by_stat.png")
        else:
            print("[WARN] Skipping chart player_mean_pred_vs_actual_by_stat.png: missing data")

    # per-stat scatter
    if not player_eval.empty:
        df = _normalize_player_eval(player_eval)
        for stat in ["ast", "fg3", "pts", "reb"]:
            sub = df.loc[df["stat"] == stat].copy()
            _maybe_scatter(
                sub,
                "pred_mean",
                "actual_value",
                f"Pred vs Actual - {stat}",
                analysis_dir / f"player_scatter_pred_vs_actual_{stat}.png",
            )

        # bucket curves
        for stat in ["ast", "fg3", "pts", "reb"]:
            sub = df.loc[df["stat"] == stat].copy()
            if sub.empty:
                print(f"[WARN] Skipping chart player_beat_projection_by_pred_bucket_{stat}.png: missing data")
                continue

            try:
                sub["pred_bucket"] = pd.qcut(
                    sub["pred_mean"],
                    q=min(5, max(1, sub["pred_mean"].nunique())),
                    duplicates="drop",
                )
            except Exception:
                print(f"[WARN] Skipping chart player_beat_projection_by_pred_bucket_{stat}.png: missing data")
                continue

            curve = (
                sub.groupby("pred_bucket", observed=False)
                .agg(beat_projection_rate=("beat_projection", "mean"))
                .reset_index()
            )

            if curve.empty:
                print(f"[WARN] Skipping chart player_beat_projection_by_pred_bucket_{stat}.png: missing data")
                continue

            plt.figure(figsize=(7, 4.5))
            plt.plot(curve["pred_bucket"].astype(str), curve["beat_projection_rate"], marker="o")
            plt.title(f"Beat Projection by Pred Bucket - {stat}")
            plt.xlabel("pred_bucket")
            plt.ylabel("beat_projection_rate")
            _save_fig(analysis_dir / f"player_beat_projection_by_pred_bucket_{stat}.png")

        # top-pick accuracy curve using available rank field
        rank_field = None
        for c in ["rank_score", "score", "p_hit", "pred_mean"]:
            if c in df.columns:
                rank_field = c
                break

        if rank_field is not None:
            for stat in ["ast", "fg3", "pts", "reb"]:
                sub = df.loc[df["stat"] == stat].copy()
                if len(sub) < 10:
                    print(f"[WARN] Skipping chart top_pick_accuracy_curve_{stat}.png: missing data")
                    continue

                sub[rank_field] = _safe_numeric(sub, rank_field)
                sub = sub.dropna(subset=[rank_field]).copy()
                if sub.empty:
                    print(f"[WARN] Skipping chart top_pick_accuracy_curve_{stat}.png: missing data")
                    continue

                sub["rank_pct"] = sub[rank_field].rank(pct=True, method="average")

                rows = []
                for thresh in np.arange(0.60, 1.00, 0.05):
                    s = sub.loc[sub["rank_pct"] >= thresh].copy()
                    if len(s) == 0:
                        continue
                    rows.append(
                        {
                            "threshold": thresh,
                            "beat_projection_rate": s["beat_projection"].mean(),
                        }
                    )

                curve = pd.DataFrame(rows)
                if curve.empty:
                    print(f"[WARN] Skipping chart top_pick_accuracy_curve_{stat}.png: missing data")
                    continue

                plt.figure(figsize=(7, 4.5))
                plt.plot(curve["threshold"], curve["beat_projection_rate"], marker="o")
                plt.title(f"Top Pick Accuracy Curve - {stat}")
                plt.xlabel("top score threshold")
                plt.ylabel("beat_projection_rate")
                _save_fig(analysis_dir / f"top_pick_accuracy_curve_{stat}.png")


def make_ticket_charts(ticket_overview: pd.DataFrame, analysis_dir: Path) -> None:
    _maybe_bar_plot(
        ticket_overview,
        "ticket_type",
        "ticket_win_rate",
        "Ticket Win Rate by Type",
        analysis_dir / "ticket_win_rate_by_type.png",
    )
    _maybe_bar_plot(
        ticket_overview,
        "ticket_type",
        "avg_leg_win_rate",
        "Ticket Avg Leg Win Rate by Type",
        analysis_dir / "ticket_avg_leg_win_rate_by_type.png",
    )
    _maybe_bar_plot(
        ticket_overview,
        "ticket_type",
        "avg_edge_vs_line",
        "Ticket Avg Edge vs Line by Type",
        analysis_dir / "ticket_avg_edge_vs_line_by_type.png",
    )
    _maybe_bar_plot(
        ticket_overview,
        "ticket_type",
        "avg_pred",
        "Ticket Avg Pred by Type",
        analysis_dir / "ticket_avg_pred_by_type.png",
    )
    _maybe_bar_plot(
        ticket_overview,
        "ticket_type",
        "avg_actual",
        "Ticket Avg Actual by Type",
        analysis_dir / "ticket_avg_actual_by_type.png",
    )

    if not ticket_overview.empty and {"ticket_type", "avg_pred", "avg_actual"}.issubset(ticket_overview.columns):
        plot_df = ticket_overview[["ticket_type", "avg_pred", "avg_actual"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(8, 4.5))
            x = np.arange(len(plot_df))
            width = 0.35
            plt.bar(x - width / 2, plot_df["avg_pred"], width=width, label="avg_pred")
            plt.bar(x + width / 2, plot_df["avg_actual"], width=width, label="avg_actual")
            plt.xticks(x, plot_df["ticket_type"].astype(str))
            plt.title("Ticket Avg Pred vs Actual by Type")
            plt.xlabel("ticket_type")
            plt.ylabel("value")
            plt.legend()
            _save_fig(analysis_dir / "ticket_avg_pred_vs_actual_by_type.png")
        else:
            print("[WARN] Skipping chart ticket_avg_pred_vs_actual_by_type.png: missing data")


# =========================================================
# Console findings
# =========================================================

def print_key_findings(
    player_overview: pd.DataFrame,
    calibration_df: pd.DataFrame,
    top_pick_overview: pd.DataFrame,
    ticket_overview: pd.DataFrame,
) -> None:
    print("\n=== KEY FINDINGS ===")

    if player_overview is not None and not player_overview.empty:
        if {"stat", "mae", "rmse"}.issubset(player_overview.columns):
            best_mae_row = player_overview.sort_values("mae", ascending=True).iloc[0]
            print(
                f"[PLAYER] Best stat by MAE: {best_mae_row['stat']} "
                f"(MAE={best_mae_row['mae']:.3f}, RMSE={best_mae_row['rmse']:.3f})"
            )

        if {"stat", "bias"}.issubset(player_overview.columns):
            tmp = player_overview.copy()
            tmp["abs_bias"] = pd.to_numeric(tmp["bias"], errors="coerce").abs()
            biggest_bias_row = tmp.sort_values("abs_bias", ascending=False).iloc[0]
            print(
                f"[PLAYER] Biggest absolute bias: {biggest_bias_row['stat']} "
                f"(bias={biggest_bias_row['bias']:.3f})"
            )

    if calibration_df is not None and not calibration_df.empty:
        print("\n=== CALIBRATION SUGGESTIONS ===")
        for _, row in calibration_df.iterrows():
            stat = row.get("stat", "unknown")
            additive_adj = row.get("additive_adj", np.nan)
            multiplier = row.get("multiplier", np.nan)
            print(
                f"[CAL] {stat}: additive_adj={additive_adj:.3f}, "
                f"multiplier={multiplier:.3f}"
            )

    if top_pick_overview is not None and not top_pick_overview.empty:
        print("\n=== TOP-PICK ACCURACY ===")
        tmp = top_pick_overview.sort_values(["stat", "top_score_threshold"], ascending=[True, True])
        for _, row in tmp.iterrows():
            print(
                f"[TOP] {row['stat']} top>={row['top_score_threshold']:.2f}: "
                f"n={int(row['n'])}, "
                f"beat_projection_rate={row['beat_projection_rate']:.3f}, "
                f"avg_overperformance={row['avg_overperformance']:.3f}"
            )

    if ticket_overview is not None and not ticket_overview.empty:
        print("\n=== TICKET FINDINGS ===")

        if {"ticket_type", "ticket_win_rate"}.issubset(ticket_overview.columns):
            best_ticket_row = ticket_overview.sort_values("ticket_win_rate", ascending=False).iloc[0]
            print(
                f"[TICKET] Best ticket type by win rate: {best_ticket_row['ticket_type']} "
                f"(win_rate={best_ticket_row['ticket_win_rate']:.3f})"
            )

        if {"ticket_type", "avg_leg_win_rate"}.issubset(ticket_overview.columns):
            best_leg_row = ticket_overview.sort_values("avg_leg_win_rate", ascending=False).iloc[0]
            print(
                f"[TICKET] Best avg leg win rate: {best_leg_row['ticket_type']} "
                f"(avg_leg_win_rate={best_leg_row['avg_leg_win_rate']:.3f})"
            )

        if {"ticket_type", "avg_edge_vs_line"}.issubset(ticket_overview.columns):
            best_edge_row = ticket_overview.sort_values("avg_edge_vs_line", ascending=False).iloc[0]
            print(
                f"[TICKET] Best avg edge vs line: {best_edge_row['ticket_type']} "
                f"(avg_edge_vs_line={best_edge_row['avg_edge_vs_line']:.3f})"
            )


# =========================================================
# Main
# =========================================================

def main(*, results_root: str = "results_backtest") -> None:
    results_root_p = Path(results_root)
    analysis_dir = results_root_p / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    player_eval = load_player_eval(results_root_p)
    ticket_eval = load_ticket_eval(results_root_p)

    player_overview = build_player_overview(player_eval)
    player_bucket_overview = build_player_bucket_overview(player_eval)
    top_pick_overview = build_top_pick_overview(player_eval)
    calibration_df = build_stat_calibration_suggestions(player_overview)
    top_overpredicted_players, top_underpredicted_players = build_top_bias_players(player_eval)
    ticket_overview = build_ticket_overview(ticket_eval)

    _save_csv(player_overview, analysis_dir / "player_overview.csv")
    _save_csv(player_bucket_overview, analysis_dir / "player_bucket_overview.csv")
    _save_csv(top_pick_overview, analysis_dir / "top_pick_overview.csv")
    _save_csv(calibration_df, analysis_dir / "stat_calibration_suggestions.csv")
    _save_csv(top_overpredicted_players, analysis_dir / "top_overpredicted_players.csv")
    _save_csv(top_underpredicted_players, analysis_dir / "top_underpredicted_players.csv")
    _save_csv(ticket_overview, analysis_dir / "ticket_overview.csv")

    make_player_charts(player_overview, player_eval, analysis_dir)
    make_ticket_charts(ticket_overview, analysis_dir)

    print_key_findings(
        player_overview=player_overview,
        calibration_df=calibration_df,
        top_pick_overview=top_pick_overview,
        ticket_overview=ticket_overview,
    )

    print(f"\n[DONE] Analysis complete -> {analysis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="results_backtest")
    args = parser.parse_args()

    main(results_root=args.results_root)