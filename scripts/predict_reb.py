# scripts/predict_reb.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from model_training.rebounds.predict import predict_game_reb
from model_training.config import PATH_GAMLOGS_COMBINED, REBOUNDS_MODEL_DIR
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
    over_baseline_delta: float = 2.0,
    reb_blend: float = 0.25,
    # ----------------------------
    # MANUAL MATCHUP FALLBACK
    # ----------------------------
    away_team: str | None = None,
    home_team: str | None = None,
    game_date: str | None = None,  # if None -> uses model_date
) -> None:
    # --- schedule date (THIS controls folder + schedule) -------------------
    schedule_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    schedule_date = schedule_dt.strftime("%Y-%m-%d")

    # --- output dir ALWAYS uses schedule_date ------------------------------
    out_dir = Path("results") / schedule_date
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_meta.txt").write_text(f"[INFO] schedule_date={schedule_date}\n")

    # --- history (READ ONLY by default) -----------------------------------
    combined_path = Path(PATH_GAMLOGS_COMBINED) if not isinstance(PATH_GAMLOGS_COMBINED, Path) else PATH_GAMLOGS_COMBINED
    if rebuild_history:
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    if not combined_path.exists():
        raise FileNotFoundError(
            f"Missing combined gamelogs CSV: {combined_path}\n"
            "Rebuild once (rebuild_history=True) or ensure scraper wrote the file."
        )

    history_df = pd.read_csv(combined_path, low_memory=False)
    if "date" not in history_df.columns and "game_date" in history_df.columns:
        history_df["date"] = history_df["game_date"]
    if "date" not in history_df.columns:
        raise ValueError(f"'date' column missing from {combined_path}")

    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df = history_df.dropna(subset=["date"]).copy()
    if history_df.empty:
        raise ValueError(f"{combined_path} contains 0 valid rows after parsing 'date'")

    # --- decide model_date (THIS controls predict_game_reb game_date) ------
    max_hist_date = history_df["date"].max()
    if pd.isna(max_hist_date):
        raise ValueError("History has no valid dates after parsing.")

    schedule_dt_pd = pd.to_datetime(schedule_date)
    if schedule_dt_pd > (max_hist_date + pd.Timedelta(days=1)):
        model_date = max_hist_date.strftime("%Y-%m-%d")
        note = (
            f"[INFO] schedule_date={schedule_date} beyond history max date={max_hist_date.date()}.\n"
            f"       Using model_date={model_date} for features/predictions (likely ASB / no new logs).\n"
        )
        print(note)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + note)
    else:
        model_date = schedule_date
        note = f"[INFO] Using model_date={model_date} (history covers schedule_date).\n"
        print(note)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + note)

    # if manual game_date not provided, use model_date
    if game_date is None:
        game_date = model_date

    # --- model paths -------------------------------------------------------
    reb_dir = Path(REBOUNDS_MODEL_DIR) if not isinstance(REBOUNDS_MODEL_DIR, Path) else REBOUNDS_MODEL_DIR
    reb_model_path = reb_dir / "reb_model.joblib"
    reb_features_path = reb_dir / "features.joblib"

    for p in [reb_model_path, reb_features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model artifact: {p}")

    # --- load schedule (safe) ---------------------------------------------
    df_games = _safe_get_games(schedule_dt, schedule_date, out_dir)

    # Decide matchups
    matchups: list[tuple[str, str]] = []

    if not df_games.empty and {"away_abbrev", "home_abbrev"}.issubset(df_games.columns):
        print(df_games[["away_abbrev", "home_abbrev", "status_text", "game_id"]])
        for g in df_games.itertuples(index=False):
            away = norm_team(g.away_abbrev)
            home = norm_team(g.home_abbrev)
            if away and home and away != "NAN" and home != "NAN":
                matchups.append((away, home))

    # If schedule failed or empty, require manual teams
    if not matchups:
        if not away_team or not home_team:
            raise ValueError(
                "No schedule games available (stats.nba.com timeout or empty schedule) AND no manual matchup provided.\n"
                "Pass away_team/home_team (and optionally game_date) to run a single matchup."
            )
        matchups = [(norm_team(away_team), norm_team(home_team))]

        msg = f"[INFO] Using manual matchup fallback: {matchups[0][0]}@{matchups[0][1]} | game_date={game_date}\n"
        print(msg)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + msg)

    # --- run predictions ---------------------------------------------------
    all_raw: list[pd.DataFrame] = []

    for away, home in matchups:
        out = predict_game_reb(
            history_df=history_df,
            away_team=away,
            home_team=home,
            game_date=game_date,  # IMPORTANT: use model_date-safe date
            reb_model_path=str(reb_model_path),
            features_path=str(reb_features_path),
            min_games_required=10,
            recent_n=5,
            reb_blend=reb_blend,
            use_v2=True,
            over_baseline_delta=over_baseline_delta,
        )

        out.sort_values("pred_reb", ascending=False, inplace=True)
        out.to_csv(out_dir / f"{away}_at_{home}_reb_predictions.csv", index=False)
        all_raw.append(out)

    # --- write all-matchups outputs ---------------------------------------
    if all_raw:
        all_df_raw = pd.concat(all_raw, ignore_index=True)
        all_df_raw.to_csv(out_dir / "all_matchups_reb_predictions.csv", index=False)
        print(f"[INFO] Wrote combined raw results -> {out_dir / 'all_matchups_reb_predictions.csv'}")
    else:
        print("[WARN] No raw matchup outputs were produced.")


if __name__ == "__main__":
    # If schedule fetch dies, this still runs your manual config.
    main(
        use_tomorrow=False,
        rebuild_history=False,
        over_baseline_delta=2.0,
        reb_blend=0.25,
        away_team="LAL",
        home_team="BOS",
        game_date="2026-02-20",
    )