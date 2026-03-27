from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from model_training.assists.predict import predict_ast_player_means
from model_training.common.history_prep import prepare_history_df
from model_training.common.predict_slate import print_slate_debug, resolve_matchups
from model_training.common.today_row import build_today_rows_v2
from model_training.config import ASSISTS_MODEL_DIR, PATH_GAMLOGS_COMBINED
from model_training.utils.team_codes import norm_team


def _auto_relax_active_within_days(
    *,
    history_df: pd.DataFrame,
    run_date: str,
    active_within_days: int | None,
) -> int | None:
    if active_within_days is None:
        return None

    hist_max_date = pd.to_datetime(history_df["game_date"], errors="coerce").max()
    if pd.isna(hist_max_date):
        return active_within_days

    gap_days = (pd.to_datetime(run_date) - hist_max_date).days
    if gap_days <= active_within_days:
        return active_within_days

    relaxed = gap_days + 7
    print(
        f"[AST] Relaxing active_within_days from {active_within_days} to {relaxed} "
        f"because history is stale relative to run_date "
        f"(hist_max_date={hist_max_date.date()}, run_date={run_date})."
    )
    return relaxed


def _load_history_df(
    *,
    rebuild_history: bool = False,
) -> pd.DataFrame:
    combined_path = Path(PATH_GAMLOGS_COMBINED)

    if rebuild_history:
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    history_df = pd.read_csv(combined_path, low_memory=False)
    history_df = prepare_history_df(history_df, norm_team_fn=norm_team)

    if "game_date" not in history_df.columns:
        raise KeyError("prepare_history_df must produce a 'game_date' column")

    history_df["game_date"] = pd.to_datetime(
        history_df["game_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    return history_df


def _build_historical_slate_from_gamelogs(
    *,
    history_df: pd.DataFrame,
    run_date: str,
) -> pd.DataFrame:
    """
    Build a historical slate directly from the gamelog history for one date.

    Required columns in prepared history:
        game_date, team, opp
    Optional:
        is_home or home_game
    """
    df_day = history_df.loc[history_df["game_date"] == run_date].copy()
    if df_day.empty:
        raise ValueError(f"No games found in history for run_date={run_date}")

    required = ["team", "opp"]
    missing = [c for c in required if c not in df_day.columns]
    if missing:
        raise KeyError(
            f"Historical slate build requires columns {missing}. "
            f"Available columns: {sorted(df_day.columns.tolist())}"
        )

    keep_cols = ["game_date", "team", "opp"]

    if "is_home" in df_day.columns:
        keep_cols.append("is_home")
    elif "home_game" in df_day.columns:
        df_day["is_home"] = pd.to_numeric(df_day["home_game"], errors="coerce").fillna(0).astype(int)
        keep_cols.append("is_home")

    slate = df_day[keep_cols].drop_duplicates().copy()

    rev = slate.copy()
    rev["team"], rev["opp"] = slate["opp"], slate["team"]

    if "is_home" in slate.columns:
        rev["is_home"] = 1 - pd.to_numeric(slate["is_home"], errors="coerce").fillna(0).astype(int)

    slate = pd.concat([slate, rev], ignore_index=True).drop_duplicates(
        subset=["game_date", "team", "opp"]
    )

    if "is_home" not in slate.columns:
        slate["is_home"] = 0

    slate["game_date"] = pd.to_datetime(slate["game_date"], errors="coerce")
    slate["team"] = slate["team"].astype(str).str.strip().str.upper()
    slate["opp"] = slate["opp"].astype(str).str.strip().str.upper()
    slate["is_home"] = pd.to_numeric(slate["is_home"], errors="coerce").fillna(0).astype(int)

    return slate.reset_index(drop=True)


def predict_ast_for_date(
    *,
    game_date: str | None = None,
    use_tomorrow: bool = False,
    rebuild_history: bool = False,
    min_games_required: int = 3,
    active_within_days: int | None = 21,
    min_minutes_threshold: float = 8.0,
    max_players_per_team: int = 12,
    away_team: str | None = None,
    home_team: str | None = None,
    write_output: bool = True,
    print_debug: bool = True,
) -> pd.DataFrame:
    """
    Predict AST means for a specific slate date.

    Historical / backtest mode:
        if game_date is provided, force run_date=game_date and derive slate
        directly from historical gamelogs.

    Live mode:
        if game_date is None, use resolve_matchups().
    """
    history_df = _load_history_df(rebuild_history=rebuild_history)

    # =========================================================
    # Historical mode (backtests)
    # =========================================================
    if game_date is not None:
        run_date = pd.to_datetime(game_date).strftime("%Y-%m-%d")
        history_cutoff_date = run_date
        results_dir = Path("results_backtest") / run_date

        slate_df = _build_historical_slate_from_gamelogs(
            history_df=history_df,
            run_date=run_date,
        )

    # =========================================================
    # Live mode
    # =========================================================
    else:
        schedule_dt = datetime.today() + (
            timedelta(days=1) if use_tomorrow else timedelta(days=0)
        )

        slate_df, run_date, history_cutoff_date, results_dir = resolve_matchups(
            schedule_dt=schedule_dt,
            history_df=history_df,
            away_team=away_team,
            home_team=home_team,
            feature_date=None,
        )

    if print_debug:
        print_slate_debug(
            prefix="AST",
            slate_df=slate_df,
            run_date=run_date,
            history_cutoff_date=history_cutoff_date,
        )

    history_df_pre = history_df.loc[
        pd.to_datetime(history_df["game_date"], errors="coerce")
        < pd.to_datetime(run_date)
    ].copy()

    if history_df_pre.empty:
        raise ValueError(f"No historical rows before run_date={run_date}")

    effective_active_within_days = _auto_relax_active_within_days(
        history_df=history_df_pre,
        run_date=run_date,
        active_within_days=active_within_days,
    )

    slate_for_today_rows = slate_df.drop(
        columns=[
            c
            for c in ["_run_date", "_history_cutoff_date", "_schedule_source"]
            if c in slate_df.columns
        ],
        errors="ignore",
    )

    today_df = build_today_rows_v2(
        df_hist=history_df_pre,
        slate_df=slate_for_today_rows,
        min_games_required=min_games_required,
        active_within_days=effective_active_within_days,
        min_minutes_threshold=min_minutes_threshold,
        max_players_per_team=max_players_per_team,
        error_on_empty=True,
    )

    pred_df = predict_ast_player_means(
        history_df=history_df_pre,
        today_df=today_df,
        model_dir=ASSISTS_MODEL_DIR,
    ).copy()

    pred_df["game_date"] = run_date
    if "stat" not in pred_df.columns:
        pred_df["stat"] = "ast"

    if write_output:
        results_dir.mkdir(parents=True, exist_ok=True)

        out_path = results_dir / "pred_ast.csv"
        pred_df.to_csv(out_path, index=False)

        (results_dir / "_meta_ast.txt").write_text(
            "\n".join(
                [
                    f"run_date={run_date}",
                    f"history_cutoff_date={history_cutoff_date}",
                    f"active_within_days={effective_active_within_days}",
                    f"min_games_required={min_games_required}",
                    f"min_minutes_threshold={min_minutes_threshold}",
                    f"max_players_per_team={max_players_per_team}",
                    f"rows={len(pred_df)}",
                ]
            )
            + "\n"
        )

        print(f"[AST] Saved -> {out_path}")

    if print_debug and not pred_df.empty:
        print(
            pred_df.sort_values("pred_mean", ascending=False)
            .head(20)
            .to_string(index=False)
        )

    return pred_df


def main(
    *,
    use_tomorrow: bool = False,
    rebuild_history: bool = False,
    min_games_required: int = 3,
    active_within_days: int | None = 21,
    min_minutes_threshold: float = 8.0,
    max_players_per_team: int = 12,
    away_team: str | None = None,
    home_team: str | None = None,
    game_date: str | None = None,
) -> None:
    predict_ast_for_date(
        game_date=game_date,
        use_tomorrow=use_tomorrow,
        rebuild_history=rebuild_history,
        min_games_required=min_games_required,
        active_within_days=active_within_days,
        min_minutes_threshold=min_minutes_threshold,
        max_players_per_team=max_players_per_team,
        away_team=away_team,
        home_team=home_team,
        write_output=True,
        print_debug=True,
    )


if __name__ == "__main__":
    main()