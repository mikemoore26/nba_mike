from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from model_training.common.history_prep import prepare_history_df
from model_training.config import PATH_GAMLOGS_COMBINED
from model_training.utils.team_codes import norm_team

from scripts.predict_pts import predict_pts_for_date
from scripts.predict_reb import predict_reb_for_date
from scripts.predict_ast import predict_ast_for_date
from scripts.predict_fg3 import predict_fg3_for_date

from ticket.pseudo_legs import expand_to_pseudo_legs
from ticket.score_legs import score_legs
from ticket.build_ticket import build_all_tickets
from ticket.pool_postprocess import build_curated_scored_pool

try:
    from ticket.rank_legs import rank_projection_pool
except ImportError:
    def rank_projection_pool(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df.copy()

from model_training.backtest.evaluators import evaluate_ticket_frames


RESULTS_ROOT = Path("results_backtest")


def _date_range(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def _safe_concat(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _load_history_df() -> pd.DataFrame:
    history_df = pd.read_csv(PATH_GAMLOGS_COMBINED, low_memory=False)
    history_df = prepare_history_df(history_df, norm_team_fn=norm_team)

    if "game_date" not in history_df.columns:
        raise KeyError("prepare_history_df must produce a 'game_date' column")

    history_df["game_date"] = pd.to_datetime(
        history_df["game_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    if "player" in history_df.columns:
        history_df["player"] = history_df["player"].astype(str).str.strip()

    if "team" in history_df.columns:
        history_df["team"] = history_df["team"].astype(str).str.strip().str.upper()

    return history_df


def _load_actuals_for_date(history_df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    return history_df.loc[history_df["game_date"] == run_date].copy()


def _attach_ticket_ids(
    ticket_frames_dict: dict[str, pd.DataFrame],
    *,
    date_str: str,
) -> list[pd.DataFrame]:
    """
    Critical fix:
    Every leg in the same built ticket must share the same ticket_id,
    otherwise evaluation treats each leg as its own ticket.
    """
    out: list[pd.DataFrame] = []

    for key in ["safe", "balanced", "lotto"]:
        df_t = ticket_frames_dict.get(key)
        if df_t is None or not isinstance(df_t, pd.DataFrame) or df_t.empty:
            continue

        df_t = df_t.copy()
        df_t["ticket_type"] = key

        # One ticket per type per date in the current builder design
        df_t["ticket_id"] = f"{date_str}_{key}_1"

        # Stable leg ordering for debugging / readability
        if "score" in df_t.columns:
            df_t = df_t.sort_values("score", ascending=False).reset_index(drop=True)
        elif "score_balanced" in df_t.columns:
            df_t = df_t.sort_values("score_balanced", ascending=False).reset_index(drop=True)
        elif "p_hit" in df_t.columns:
            df_t = df_t.sort_values("p_hit", ascending=False).reset_index(drop=True)
        else:
            df_t = df_t.reset_index(drop=True)

        df_t["leg_order"] = range(1, len(df_t) + 1)
        out.append(df_t)

    return out


def run_backtest(
    *,
    start_date: str,
    end_date: str,
) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[BACKTEST] Running from {start_date} -> {end_date}")

    history_df = _load_history_df()
    all_ticket_evals: list[pd.DataFrame] = []

    for date_str in _date_range(start_date, end_date):
        print(f"\n[DATE] {date_str}")

        try:
            results_dir = RESULTS_ROOT / date_str
            results_dir.mkdir(parents=True, exist_ok=True)

            # -----------------------------
            # STEP 1: Predict player means
            # -----------------------------
            print("[STEP] Predicting stats...")

            df_pts = predict_pts_for_date(
                game_date=date_str,
                write_output=True,
                print_debug=False,
            )
            df_reb = predict_reb_for_date(
                game_date=date_str,
                write_output=True,
                print_debug=False,
            )
            df_ast = predict_ast_for_date(
                game_date=date_str,
                write_output=True,
                print_debug=False,
            )
            df_fg3 = predict_fg3_for_date(
                game_date=date_str,
                write_output=True,
                print_debug=False,
            )

            pred_all = _safe_concat([df_pts, df_reb, df_ast, df_fg3])

            if pred_all.empty:
                print("[WARN] No predictions, skipping.")
                continue

            pred_all_path = results_dir / "pred_all.csv"
            pred_all.to_csv(pred_all_path, index=False)
            print(f"[SAVED] {pred_all_path}")

            # -----------------------------
            # STEP 2: Expand pseudo bet legs
            # -----------------------------
            print("[STEP] Expanding pseudo legs...")
            pseudo_legs = expand_to_pseudo_legs(
                pred_all,
                line_offsets=(-1.0, 0.0, 1.0),
                min_prob=0.50,
                keep_both_sides=True,
            )

            pseudo_path = results_dir / "pseudo_legs.csv"
            pseudo_legs.to_csv(pseudo_path, index=False)
            print(f"[SAVED] {pseudo_path}")

            if pseudo_legs.empty:
                print("[WARN] pseudo_legs empty, skipping.")
                continue

            # -----------------------------
            # STEP 3: Score legs
            # -----------------------------
            print("[STEP] Scoring legs...")
            scored = score_legs(pseudo_legs)

            scored_raw_path = results_dir / "scored_legs_raw.csv"
            scored.to_csv(scored_raw_path, index=False)
            print(f"[SAVED] {scored_raw_path}")

            if scored.empty:
                print("[WARN] scored_legs empty, skipping.")
                continue

            # -----------------------------
            # STEP 4: Curate scored pool
            # -----------------------------
            print("[STEP] Curating scored pool...")
            scored_curated = build_curated_scored_pool(
                scored,
                score_col="score_balanced",
                min_p_hit=0.55,
                min_edge_raw=0.50,
            )

            scored_curated_path = results_dir / "scored_legs_curated.csv"
            scored_curated.to_csv(scored_curated_path, index=False)
            print(f"[SAVED] {scored_curated_path}")

            if scored_curated.empty:
                print("[WARN] scored_curated empty, skipping.")
                continue

            # -----------------------------
            # STEP 5: Ranked pool
            # -----------------------------
            print("[STEP] Ranking pool...")
            ranked_pool = rank_projection_pool(scored_curated)

            ranked_path = results_dir / "ranked_pool.csv"
            ranked_pool.to_csv(ranked_path, index=False)
            print(f"[SAVED] {ranked_path}")

            # -----------------------------
            # STEP 6: Build tickets
            # -----------------------------
            print("[STEP] Building tickets...")
            tickets = build_all_tickets(scored_curated)

            tickets_dir = results_dir / "tickets"
            tickets_dir.mkdir(parents=True, exist_ok=True)

            if isinstance(tickets, dict):
                builder_ranked_pool = tickets.get("ranked_pool")
                if isinstance(builder_ranked_pool, pd.DataFrame) and not builder_ranked_pool.empty:
                    builder_ranked_path = results_dir / "ranked_pool_from_builder.csv"
                    builder_ranked_pool.to_csv(builder_ranked_path, index=False)
                    print(f"[SAVED] {builder_ranked_path}")

            ticket_frames: list[pd.DataFrame] = []

            if isinstance(tickets, dict):
                ticket_frames = _attach_ticket_ids(
                    tickets,
                    date_str=date_str,
                )

                for df_t in ticket_frames:
                    ticket_type = str(df_t["ticket_type"].iloc[0])
                    out_path = tickets_dir / f"ticket_{ticket_type}.csv"
                    df_t.to_csv(out_path, index=False)
                    print(f"[SAVED] {out_path}")

                summary_df = tickets.get("summary")
                if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                    summary_path = tickets_dir / "ticket_summary.csv"
                    summary_df.to_csv(summary_path, index=False)
                    print(f"[SAVED] {summary_path}")

            elif isinstance(tickets, pd.DataFrame):
                df_t = tickets.copy()
                df_t["ticket_type"] = "unknown"
                df_t["ticket_id"] = f"{date_str}_unknown_1"
                df_t["leg_order"] = range(1, len(df_t) + 1)

                out_path = tickets_dir / "tickets.csv"
                df_t.to_csv(out_path, index=False)
                print(f"[SAVED] {out_path}")

                ticket_frames = [df_t]

            if not ticket_frames:
                print("[WARN] No ticket frames generated.")
                continue

            ticket_legs = pd.concat(ticket_frames, ignore_index=True)

            ticket_legs_path = results_dir / "ticket_legs.csv"
            ticket_legs.to_csv(ticket_legs_path, index=False)
            print(f"[SAVED] {ticket_legs_path}")

            # -----------------------------
            # STEP 7: Load actuals
            # -----------------------------
            print("[STEP] Loading actuals...")
            actuals = _load_actuals_for_date(history_df, date_str)

            if actuals.empty:
                print("[WARN] No actuals found, skipping.")
                continue

            # -----------------------------
            # STEP 8: Evaluate tickets
            # -----------------------------
            print("[STEP] Evaluating tickets...")
            ticket_eval = evaluate_ticket_frames(
                ticket_legs=ticket_legs,
                actuals=actuals,
            )

            if ticket_eval.empty:
                print("[WARN] ticket_eval empty")
                continue

            ticket_eval_path = results_dir / "backtest_ticket_eval.csv"
            ticket_eval.to_csv(ticket_eval_path, index=False)
            print(f"[SAVED] {ticket_eval_path}")

            all_ticket_evals.append(ticket_eval)

        except Exception as e:
            print(f"[ERROR] Failed on {date_str}: {e}")
            continue

    if all_ticket_evals:
        final_df = pd.concat(all_ticket_evals, ignore_index=True)
        final_path = RESULTS_ROOT / "backtest_ticket_eval.csv"
        final_df.to_csv(final_path, index=False)
        print(f"\n[SAVED] FINAL -> {final_path}")
    else:
        print("\n[WARN] No ticket evaluations generated.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="2025-01-01")
    parser.add_argument("--end-date", type=str, default="2025-01-10")
    args = parser.parse_args()

    run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()