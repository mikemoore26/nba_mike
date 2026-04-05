from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def _safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()

    for col in ["player", "team", "opp", "stat"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()

    if "team" in out.columns:
        out["team"] = out["team"].str.upper()
    if "opp" in out.columns:
        out["opp"] = out["opp"].str.upper()
    if "stat" in out.columns:
        out["stat"] = out["stat"].str.lower()

    numeric_cols = [
        "p_hit",
        "delta",
        "pred_mean",
        "line",
        "minutes_proj",
        "role_score",
        "stability_score",
        "usage_score",
        "fragility_score",
        "projection_rank_score",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _summary_by_stat(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty or "stat" not in df.columns:
        return pd.DataFrame()

    g = (
        df.groupby("stat", dropna=False)
        .agg(
            n=("stat", "size"),
            avg_p_hit=("p_hit", "mean") if "p_hit" in df.columns else ("stat", "size"),
            avg_delta=("delta", "mean") if "delta" in df.columns else ("stat", "size"),
            avg_pred=("pred_mean", "mean") if "pred_mean" in df.columns else ("stat", "size"),
            avg_line=("line", "mean") if "line" in df.columns else ("stat", "size"),
            avg_minutes=("minutes_proj", "mean") if "minutes_proj" in df.columns else ("stat", "size"),
            avg_role_score=("role_score", "mean") if "role_score" in df.columns else ("stat", "size"),
            avg_stability=("stability_score", "mean") if "stability_score" in df.columns else ("stat", "size"),
            avg_usage=("usage_score", "mean") if "usage_score" in df.columns else ("stat", "size"),
            avg_fragility=("fragility_score", "mean") if "fragility_score" in df.columns else ("stat", "size"),
            avg_rank_score=("projection_rank_score", "mean") if "projection_rank_score" in df.columns else ("stat", "size"),
        )
        .reset_index()
    )

    total = g["n"].sum()
    g["share"] = g["n"] / total if total > 0 else 0.0
    g["stage"] = label

    cols = [
        "stage",
        "stat",
        "n",
        "share",
        "avg_p_hit",
        "avg_delta",
        "avg_pred",
        "avg_line",
        "avg_minutes",
        "avg_role_score",
        "avg_stability",
        "avg_usage",
        "avg_fragility",
        "avg_rank_score",
    ]
    cols = [c for c in cols if c in g.columns]
    return g[cols].sort_values("stat").reset_index(drop=True)


def _top_fg3(df: pd.DataFrame, label: str, top_n: int = 20) -> pd.DataFrame:
    if df.empty or "stat" not in df.columns:
        return pd.DataFrame()

    sub = df.loc[df["stat"].eq("fg3")].copy()
    if sub.empty:
        return pd.DataFrame()

    sort_cols = [c for c in ["p_hit", "delta", "role_score", "projection_rank_score"] if c in sub.columns]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=False)

    keep_cols = [
        "player",
        "team",
        "opp",
        "stat",
        "pred_mean",
        "line",
        "p_hit",
        "delta",
        "minutes_proj",
        "role_score",
        "stability_score",
        "usage_score",
        "fragility_score",
        "projection_rank_score",
    ]
    keep_cols = [c for c in keep_cols if c in sub.columns]

    sub = sub[keep_cols].head(top_n).copy()
    sub.insert(0, "stage", label)
    return sub


def analyze_stat_representation(run_date: str | None = None) -> dict[str, pd.DataFrame]:
    if run_date is None:
        run_date = datetime.today().strftime("%Y-%m-%d")

    results_dir = Path("results") / run_date
    tickets_dir = results_dir / "tickets"

    raw_legs = _normalize(_safe_read(results_dir / "projection_legs.csv"))
    ranked_board = _normalize(_safe_read(results_dir / "projection_board_ranked.csv"))
    safe_ticket = _normalize(_safe_read(tickets_dir / "ticket_safe.csv"))
    balanced_ticket = _normalize(_safe_read(tickets_dir / "ticket_balanced.csv"))
    lotto_ticket = _normalize(_safe_read(tickets_dir / "ticket_lotto.csv"))

    # rebuild the "filtered candidate" set the way ticket construction sees it
    filtered = raw_legs.copy()
    if not filtered.empty and not ranked_board.empty:
        role_cols = [
            "player", "team", "opp",
            "role_score", "stability_score", "usage_score",
            "fragility_score", "projection_rank_score",
            "minutes_proj",
        ]
        role_cols = [c for c in role_cols if c in ranked_board.columns]
        role_df = ranked_board[role_cols].drop_duplicates(subset=["player", "team", "opp"], keep="first").copy()

        filtered = filtered.merge(
            role_df,
            on=["player", "team", "opp"],
            how="left",
            suffixes=("", "_board"),
        )

    if not filtered.empty:
        filtered = filtered[
            (_safe_num(filtered, "p_hit") >= 0.58) &
            (_safe_num(filtered, "minutes_proj") >= 26) &
            (_safe_num(filtered, "delta") >= 0.75) &
            (_safe_num(filtered, "role_score") >= 0.58) &
            (_safe_num(filtered, "fragility_score") <= 0.35) &
            (~filtered["player"].astype(str).str.contains(r"\(TW\)", na=False))
        ].copy()

    summaries = pd.concat(
        [
            _summary_by_stat(raw_legs, "raw_projection_legs"),
            _summary_by_stat(filtered, "filtered_candidate_pool"),
            _summary_by_stat(safe_ticket, "safe_ticket"),
            _summary_by_stat(balanced_ticket, "balanced_ticket"),
            _summary_by_stat(lotto_ticket, "lotto_ticket"),
        ],
        ignore_index=True,
    )

    fg3_detail = pd.concat(
        [
            _top_fg3(raw_legs, "raw_projection_legs", top_n=20),
            _top_fg3(filtered, "filtered_candidate_pool", top_n=20),
            _top_fg3(safe_ticket, "safe_ticket", top_n=20),
            _top_fg3(balanced_ticket, "balanced_ticket", top_n=20),
            _top_fg3(lotto_ticket, "lotto_ticket", top_n=20),
        ],
        ignore_index=True,
    )

    summary_path = results_dir / "stat_representation_summary.csv"
    summaries.to_csv(summary_path, index=False)
    print(f"[SAVED] stat_representation_summary -> {summary_path}")

    fg3_path = results_dir / "fg3_representation_detail.csv"
    fg3_detail.to_csv(fg3_path, index=False)
    print(f"[SAVED] fg3_representation_detail -> {fg3_path}")

    return {
        "summary": summaries,
        "fg3_detail": fg3_detail,
    }


def main() -> None:
    run_date = datetime.today().strftime("%Y-%m-%d")
    out = analyze_stat_representation(run_date=run_date)

    summary = out["summary"]
    fg3_detail = out["fg3_detail"]

    if not summary.empty:
        print("\n[STAT REPRESENTATION SUMMARY]")
        print(summary.to_string(index=False))

    if not fg3_detail.empty:
        print("\n[TOP FG3 DETAIL]")
        print(fg3_detail.head(20).to_string(index=False))
    else:
        print("\n[TOP FG3 DETAIL]")
        print("[WARN] No FG3 rows found.")


if __name__ == "__main__":
    main()