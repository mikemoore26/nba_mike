# scripts/predict_fg3.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from nba_scraper.schedule import get_todays_games_cached
from model_training.threes.predict import predict_game_fg3
from model_training.threes.probability import add_prob_ge_k
from model_training.config import PATH_GAMLOGS_COMBINED, PATH_TO_MODEL_dir


def main(*, use_tomorrow: bool = False, rebuild_history: bool = False) -> None:
    # --- choose date -------------------------------------------------------
    base_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    date_use = base_dt.strftime("%Y-%m-%d")

    # --- load schedule inside main ----------------------------------------
    df_games = get_todays_games_cached(cache_dir=Path("./data/cache"), game_date=base_dt.date())
    if df_games.empty:
        print(f"[INFO] No games found for {date_use}.")
        return

    print(df_games[["away_abbrev", "home_abbrev", "status_text", "game_id"]])

    # --- history (READ ONLY by default) -----------------------------------
    combined_path = Path(PATH_GAMLOGS_COMBINED) if not isinstance(PATH_GAMLOGS_COMBINED, Path) else PATH_GAMLOGS_COMBINED
    if rebuild_history:
        # optional manual rebuild; do NOT do this by default
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    if not combined_path.exists():
        raise FileNotFoundError(
            f"Missing combined gamelogs CSV: {combined_path}\n"
            "Run: python -m scripts.train_models_3p (or rebuild_history=True once)."
        )

    history_df = pd.read_csv(combined_path)
    if "date" not in history_df.columns:
        raise ValueError(f"'date' column missing from {combined_path}")
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    history_df = history_df.dropna(subset=["date"]).copy()
    if history_df.empty:
        raise ValueError(f"{combined_path} contains 0 valid rows after parsing 'date'")

    # --- output dir --------------------------------------------------------
    out_dir = Path("results") / date_use
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model paths -------------------------------------------------------
    model_dir = Path(PATH_TO_MODEL_dir) if not isinstance(PATH_TO_MODEL_dir, Path) else PATH_TO_MODEL_dir
    fg3a_model_path = model_dir / "fg3a_model.joblib"
    fg3_rate_model_path = model_dir / "fg3_rate_model.joblib"
    features_path = model_dir / "features.joblib"

    for p in [fg3a_model_path, fg3_rate_model_path, features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model artifact: {p}")

    # --- run predictions ---------------------------------------------------
    all_res: list[pd.DataFrame] = []

    for g in df_games.itertuples(index=False):
        away = str(g.away_abbrev).upper()
        home = str(g.home_abbrev).upper()

        if not away or not home or away == "NAN" or home == "NAN":
            continue

        out = predict_game_fg3(
            history_df=history_df,
            away_team=away,
            home_team=home,
            game_date=date_use,
            fg3a_model_path=str(fg3a_model_path),
            fg3_rate_model_path=str(fg3_rate_model_path),
            features_path=str(features_path),
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
    main(use_tomorrow=False, rebuild_history=False)
