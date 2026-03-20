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
    schedule_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    schedule_date = schedule_dt.strftime("%Y-%m-%d")

    combined_path = Path(PATH_GAMLOGS_COMBINED)
    if rebuild_history:
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    history_df = pd.read_csv(combined_path, low_memory=False)
    history_df = prepare_history_df(history_df, norm_team_fn=norm_team)

    slate_df, slate_date, results_dir = resolve_matchups(
        schedule_dt=schedule_dt,
        schedule_date=schedule_date,
        history_df=history_df,
        away_team=away_team,
        home_team=home_team,
        game_date=game_date,
    )

    print_slate_debug(prefix="AST", slate_df=slate_df, slate_date=slate_date)

    today_df = build_today_rows_v2(
        df_hist=history_df,
        slate_df=slate_df,
        min_games_required=min_games_required,
        active_within_days=active_within_days,
        min_minutes_threshold=min_minutes_threshold,
        max_players_per_team=max_players_per_team,
        error_on_empty=True,
    )

    pred_df = predict_ast_player_means(
        history_df=history_df,
        today_df=today_df,
        model_dir=ASSISTS_MODEL_DIR,
    )

    out_path = results_dir / "pred_ast.csv"
    pred_df.to_csv(out_path, index=False)

    (results_dir / "_meta_ast.txt").write_text(
        f"slate_date={slate_date}\nrows={len(pred_df)}\n"
    )

    print(f"[AST] Saved -> {out_path}")
    print(pred_df.sort_values("pred_mean", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()