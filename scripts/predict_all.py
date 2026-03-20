from __future__ import annotations

from scripts.predict_ast import main as predict_ast
from scripts.predict_reb import main as predict_reb
from scripts.predict_fg3 import main as predict_fg3
from scripts.predict_pts import main as predict_pts


def main() -> None:
    print("Predicting all player projections...")

    # TEMP:
    # history is stale relative to today's slate, so we relax recency gating
    relaxed_active_within_days = 35

    print("Predicting AST...")
    predict_ast(
        use_tomorrow=False,
        rebuild_history=False,
        min_games_required=3,
        active_within_days=relaxed_active_within_days,
        min_minutes_threshold=8.0,
        max_players_per_team=12,
    )

    print("Predicting REB...")
    predict_reb(
        use_tomorrow=False,
        rebuild_history=False,
        min_games_required=3,
        active_within_days=relaxed_active_within_days,
        min_minutes_threshold=10.0,
        max_players_per_team=12,
    )

    print("Predicting FG3...")
    predict_fg3(
        use_tomorrow=False,
        rebuild_history=False,
        min_games_required=3,
        active_within_days=relaxed_active_within_days,
        min_minutes_threshold=8.0,
        max_players_per_team=12,
    )

    print("Predicting PTS...")
    predict_pts(
        use_tomorrow=False,
        rebuild_history=False,
        min_games_required=3,
        active_within_days=relaxed_active_within_days,
        min_minutes_threshold=10.0,
        max_players_per_team=12,
    )

    print("All projection files complete.")


if __name__ == "__main__":
    main()