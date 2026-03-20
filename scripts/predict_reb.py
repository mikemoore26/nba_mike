from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from model_training.common.history_prep import prepare_history_df
from model_training.common.predict_slate import print_slate_debug, resolve_matchups
from model_training.common.today_row import build_today_rows_v2
from model_training.config import PATH_GAMLOGS_COMBINED, REBOUNDS_MODEL_DIR
from model_training.rebounds.predict import predict_reb_player_means
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



def main(
    *,
    use_tomorrow: bool = False,
    rebuild_history: bool = False,
    min_games_required: int = 3,
    active_within_days: int | None = 21,
    min_minutes_threshold: float = 10.0,
    max_players_per_team: int = 12,
    away_team: str | None = None,
    home_team: str | None = None,
    game_date: str | None = None,
) -> None:
    schedule_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    schedule_date = schedule_dt.strftime("%Y-%m-%d")

    combined_path = Path(PATH_GAMLOGS_COMBINED)
    if rebuild_history:
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    history_df = pd.read_csv(combined_path, low_memory=False)
    history_df = prepare_history_df(history_df, norm_team_fn=norm_team)

    slate_df, run_date, history_cutoff_date, results_dir = resolve_matchups(
        schedule_dt=schedule_dt,
        history_df=history_df,
        away_team=away_team,
        home_team=home_team,
        feature_date=game_date,
    )

    print_slate_debug(prefix="REB", 
                      slate_df=slate_df, 
                      run_date=run_date, 
                      history_cutoff_date=history_cutoff_date)

    effective_active_within_days = _auto_relax_active_within_days(
        history_df=history_df,
        run_date=run_date,
        active_within_days=active_within_days,
    )

    today_df = build_today_rows_v2(
        df_hist=history_df,
        slate_df=slate_df.drop(columns=[c for c in ["_run_date", "_history_cutoff_date", "_schedule_source"] if c in slate_df.columns]),
        min_games_required=min_games_required,
        active_within_days=active_within_days,
        min_minutes_threshold=min_minutes_threshold,
        max_players_per_team=max_players_per_team,
        error_on_empty=True,
    )

    pred_df = predict_reb_player_means(
        history_df=history_df,
        today_df=today_df,
        model_dir=REBOUNDS_MODEL_DIR,
    )

    out_path = results_dir / "pred_reb.csv"
    pred_df.to_csv(out_path, index=False)

    (results_dir / "_meta_reb.txt").write_text(
        f"run_date={run_date}\nhistory_cutoff_date={history_cutoff_date}\nrows={len(pred_df)}\n"
    )

    print(f"[REB] Saved -> {out_path}")
    print(pred_df.sort_values("pred_mean", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()