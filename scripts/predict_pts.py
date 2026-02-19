# scripts/predict_pts.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from nba_scraper.schedule import get_todays_games_cached
from model_training.points.predict import predict_game_points
from model_training.points.probability import add_prob_pts_ge_line
from model_training.config import PATH_GAMLOGS_COMBINED, THREES_MODEL_DIR, POINTS_MODEL_DIR
from model_training.utils.team_codes import norm_team

import time

def main(
    *,
    use_tomorrow: bool = False,
    rebuild_history: bool = False,
    points_line: float | None = 22.5,
    prob_method: str = "normal",   # "normal" or "discrete"
) -> None:
    # --- schedule date (THIS controls folder + schedule) -------------------
    schedule_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    schedule_date = schedule_dt.strftime("%Y-%m-%d")

    # --- output dir ALWAYS uses schedule_date ------------------------------
    out_dir = Path("results") / schedule_date
    out_dir.mkdir(parents=True, exist_ok=True)

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
    if "date" not in history_df.columns:
        raise ValueError(f"'date' column missing from {combined_path}")
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df = history_df.dropna(subset=["date"]).copy()
    if history_df.empty:
        raise ValueError(f"{combined_path} contains 0 valid rows after parsing 'date'")

    # --- decide model_date (THIS controls predict_game_points game_date) ---
    max_hist_date = history_df["date"].max()
    if pd.isna(max_hist_date):
        raise ValueError("History has no valid dates after parsing.")

    schedule_dt_pd = pd.to_datetime(schedule_date)
    # If schedule date is beyond history, use history max date for modeling
    if schedule_dt_pd > (max_hist_date + pd.Timedelta(days=1)):
        model_date = max_hist_date.strftime("%Y-%m-%d")
        note = (
            f"[INFO] schedule_date={schedule_date} beyond history max date={max_hist_date.date()}.\n"
            f"       Using model_date={model_date} for features/predictions (likely ASB / no new logs).\n"
        )
        print(note)
        (out_dir / "_meta.txt").write_text(note)
    else:
        model_date = schedule_date
        note = f"[INFO] Using model_date={model_date} (history covers schedule_date).\n"
        print(note)
        (out_dir / "_meta.txt").write_text(note)

    # --- load schedule using schedule_date (not model_date) ----------------
    df_games = get_todays_games_cached(cache_dir=Path("./data/cache"), game_date=schedule_dt.date())
    if df_games.empty:
        msg = f"[INFO] No games found for schedule_date={schedule_date}. Wrote {out_dir / '_meta.txt'}.\n"
        print(msg)
        (out_dir / "_meta.txt").write_text((out_dir / "_meta.txt").read_text() + msg)
        return

    print(df_games[["away_abbrev", "home_abbrev", "status_text", "game_id"]])

    # --- model paths -------------------------------------------------------
    threes_dir = Path(THREES_MODEL_DIR) if not isinstance(THREES_MODEL_DIR, Path) else THREES_MODEL_DIR
    points_dir = Path(POINTS_MODEL_DIR) if not isinstance(POINTS_MODEL_DIR, Path) else POINTS_MODEL_DIR

    fg3a_model_path = threes_dir / "fg3a_model.joblib"
    fg3_rate_model_path = threes_dir / "fg3_rate_model.joblib"
    threes_features_path = threes_dir / "features.joblib"

    fg2a_model_path = points_dir / "fg2a.joblib"
    fg2_rate_model_path = points_dir / "fg2_rate.joblib"
    fta_model_path = points_dir / "fta.joblib"
    ft_rate_model_path = points_dir / "ft_rate.joblib"
    points_features_path = points_dir / "points_features.joblib"

    for p in [
        fg3a_model_path, fg3_rate_model_path, threes_features_path,
        fg2a_model_path, fg2_rate_model_path, fta_model_path, ft_rate_model_path, points_features_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model artifact: {p}")

    # --- run predictions ---------------------------------------------------
    all_raw: list[pd.DataFrame] = []
    all_prob: list[pd.DataFrame] = []

    for g in df_games.itertuples(index=False):
        away = norm_team(g.away_abbrev)
        home = norm_team(g.home_abbrev)

        if not away or not home or away == "NAN" or home == "NAN":
            continue

        out = predict_game_points(
            history_df=history_df,
            away_team=away,
            home_team=home,
            game_date=model_date,   # IMPORTANT: use model_date (safe)
            # points models
            fg2a_model_path=str(fg2a_model_path),
            fg2_rate_model_path=str(fg2_rate_model_path),
            fta_model_path=str(fta_model_path),
            ft_rate_model_path=str(ft_rate_model_path),
            points_features_path=str(points_features_path),
            # 3P models
            fg3a_model_path=str(fg3a_model_path),
            fg3_rate_model_path=str(fg3_rate_model_path),
            threes_features_path=str(threes_features_path),
            min_games_required=10,
            recent_n=5,
            over_line_delta=3.0,
            use_v2=True,
        )

        out.sort_values("pred_pts", ascending=False, inplace=True)
        out.to_csv(out_dir / f"{away}_at_{home}_pts_predictions.csv", index=False)
        all_raw.append(out)

        if points_line is not None:
            out2 = add_prob_pts_ge_line(
                out,
                line_value=float(points_line),
                method=prob_method,
                out_col=f"p_over_{str(points_line).replace('.', '_')}",
                max_n_cap=40 if prob_method == "discrete" else None,
            )
            out2.to_csv(out_dir / f"{away}_at_{home}_pts_p_over_{str(points_line).replace('.', '_')}.csv", index=False)
            all_prob.append(out2)

    # --- write all-matchups outputs ---------------------------------------
    if all_raw:
        all_df_raw = pd.concat(all_raw, ignore_index=True)
        all_df_raw.to_csv(out_dir / "all_matchups_pts_predictions.csv", index=False)
        print(f"[INFO] Wrote combined raw results -> {out_dir / 'all_matchups_pts_predictions.csv'}")
    else:
        print("[WARN] No raw matchup outputs were produced.")

    if points_line is not None and all_prob:
        all_df_prob = pd.concat(all_prob, ignore_index=True)
        out_name = f"all_matchups_pts_p_over_{str(points_line).replace('.', '_')}.csv"
        all_df_prob.to_csv(out_dir / out_name, index=False)
        print(f"[INFO] Wrote combined prob results -> {out_dir / out_name}")
    elif points_line is not None:
        print("[WARN] No probability matchup outputs were produced.")

   

    # def file_stamp(p: Path) -> str:
    #     if not p.exists():
    #         return f"{p}  (MISSING)"
    #     return f"{p}  | modified={time.ctime(p.stat().st_mtime)}  | size={p.stat().st_size}"

    # print("\n=== MODEL FILES LOADED (POINTS) ===")
    # print(file_stamp(Path(fg2a_model_path)))
    # print(file_stamp(Path(fg2_rate_model_path)))
    # print(file_stamp(Path(fta_model_path)))
    # print(file_stamp(Path(ft_rate_model_path)))
    # print(file_stamp(Path(points_features_path)))

    


if __name__ == "__main__":
    main(use_tomorrow=False, rebuild_history=False, points_line=22.5, prob_method="normal")
