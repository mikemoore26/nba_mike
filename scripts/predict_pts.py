# scripts/predict_fg3.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from model_training.threes.predict import predict_game_fg3
from model_training.threes.probability import add_prob_ge_k
from model_training.config import PATH_GAMLOGS_COMBINED, THREES_MODEL_DIR
from model_training.utils.team_codes import norm_team


def _safe_get_games(schedule_dt: datetime, schedule_date: str, out_dir: Path) -> pd.DataFrame:
    """
    Tries to pull schedule from nba_api (stats.nba.com).
    If stats.nba.com times out, returns empty df so caller can fallback.
    """
    try:
        from nba_scraper.schedule import get_todays_games_cached

        df_games = get_todays_games_cached(
            cache_dir=Path("./data/cache"),
            game_date=schedule_dt.date(),
        )
        return df_games

    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, TimeoutError) as e:
        msg = (
            f"[WARN] Schedule fetch failed for schedule_date={schedule_date}.\n"
            f"       Reason: {type(e).__name__}: {e}\n"
            f"       Falling back to manual matchup config if provided.\n"
        )
        print(msg)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + msg)
        return pd.DataFrame()

    except Exception as e:
        msg = (
            f"[WARN] Schedule fetch failed (unexpected) for schedule_date={schedule_date}.\n"
            f"       Reason: {type(e).__name__}: {e}\n"
            f"       Falling back to manual matchup config if provided.\n"
        )
        print(msg)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + msg)
        return pd.DataFrame()


def main(
    *,
    use_tomorrow: bool = False,
    rebuild_history: bool = False,
    # ----------------------------
    # MANUAL MATCHUP FALLBACK
    # ----------------------------
    away_team: str | None = None,
    home_team: str | None = None,
    game_date: str | None = None,  # if None -> uses date_use
) -> None:
    # --- schedule date (controls folder + schedule) ------------------------
    schedule_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    schedule_date = schedule_dt.strftime("%Y-%m-%d")

    # --- output dir ALWAYS uses schedule_date ------------------------------
    out_dir = Path("results") / schedule_date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.txt").write_text(f"[INFO] schedule_date={schedule_date}\n")

    # date_use controls predict_game_fg3 game_date
    date_use = schedule_date if game_date is None else game_date

    # --- history (READ ONLY by default) -----------------------------------
    combined_path = Path(PATH_GAMLOGS_COMBINED) if not isinstance(PATH_GAMLOGS_COMBINED, Path) else PATH_GAMLOGS_COMBINED
    if rebuild_history:
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    if not combined_path.exists():
        raise FileNotFoundError(
            f"Missing combined gamelogs CSV: {combined_path}\n"
            "Run: python -m scripts.train_models_3p (or rebuild_history=True once)."
        )

    history_df = pd.read_csv(combined_path, low_memory=False)
    if "date" not in history_df.columns:
        raise ValueError(f"'date' column missing from {combined_path}")
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df = history_df.dropna(subset=["date"]).copy()
    if history_df.empty:
        raise ValueError(f"{combined_path} contains 0 valid rows after parsing 'date'")

    # --- model paths -------------------------------------------------------
    model_dir = Path(THREES_MODEL_DIR) if not isinstance(THREES_MODEL_DIR, Path) else THREES_MODEL_DIR
    fg3a_model_path = model_dir / "fg3a_model.joblib"
    fg3_rate_model_path = model_dir / "fg3_rate_model.joblib"
    features_path = model_dir / "features.joblib"

    for p in [fg3a_model_path, fg3_rate_model_path, features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model artifact: {p}")

    # --- load schedule (safe) ---------------------------------------------
    df_games = _safe_get_games(schedule_dt, schedule_date, out_dir)

    matchups: list[tuple[str, str]] = []

    if not df_games.empty and {"away_abbrev", "home_abbrev"}.issubset(df_games.columns):
        print(df_games[["away_abbrev", "home_abbrev", "status_text", "game_id"]])
        for g in df_games.itertuples(index=False):
            away = norm_team(g.away_abbrev)
            home = norm_team(g.home_abbrev)
            if away and home and away != "NAN" and home != "NAN":
                matchups.append((away, home))

    # fallback to manual matchup
    if not matchups:
        if not away_team or not home_team:
            raise ValueError(
                "No schedule games available (stats.nba.com timeout or empty schedule) AND no manual matchup provided.\n"
                "Pass away_team/home_team (and optionally game_date) to run a single matchup."
            )
        matchups = [(norm_team(away_team), norm_team(home_team))]
        msg = f"[INFO] Using manual matchup fallback: {matchups[0][0]}@{matchups[0][1]} | game_date={date_use}\n"
        print(msg)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + msg)

    # --- run predictions ---------------------------------------------------
    all_res: list[pd.DataFrame] = []

    for away, home in matchups:
        out = predict_game_fg3(
            history_df=history_df,
            away_team=away,
            home_team=home,
            game_date=date_use,
            fg3a_model_path=str(fg3a_model_path),
            fg3_rate_model_path=str(fg3_rate_model_path),
            threes_features_path=str(features_path),
        )

        out.sort_values("pred_fg3", ascending=False, inplace=True)
        out.to_csv(out_dir / f"{away}_at_{home}_fg3_predictions.csv", index=False)

        out2 = add_prob_ge_k(out, k=2)
        out2 = add_prob_ge_k(out2, k=3)
        out2.to_csv(out_dir / f"{away}_at_{home}_p2_p3.csv", index=False)

        all_res.append(out2)

    if all_res:
        all_df = pd.concat(all_res, ignore_index=True)
        all_df.to_csv(out_dir / "all_matchups_fg3_predictions.csv", index=False)
        print(f"[INFO] Wrote combined results -> {out_dir / 'all_matchups_fg3_predictions.csv'}")
    else:
        print("[WARN] No matchup outputs were produced.")


if __name__ == "__main__":
    main(
        use_tomorrow=False,
        rebuild_history=False,
        # manual fallback (optional)
        # away_team="LAL",
        # home_team="BOS",
        # game_date="2026-02-20",
    )