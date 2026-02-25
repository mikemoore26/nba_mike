# scripts/predict_fg3_auto.py
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from model_training.config import PATH_GAMLOGS_COMBINED, THREES_MODEL_DIR
from model_training.utils.team_codes import norm_team
from model_training.threes.predict import predict_game_fg3


def _safe_get_games(schedule_dt: datetime) -> pd.DataFrame:
    """
    Pull schedule from your nba_scraper (stats.nba.com).
    If it times out, return empty df so we can fallback.
    """
    try:
        from nba_scraper.schedule import get_todays_games_cached

        return get_todays_games_cached(
            cache_dir=Path("./data/cache"),
            game_date=schedule_dt.date(),
        )
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, TimeoutError) as e:
        print(f"[WARN] Schedule fetch failed: {type(e).__name__}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[WARN] Schedule fetch failed (unexpected): {type(e).__name__}: {e}")
        return pd.DataFrame()


def _matchups_from_games_df(df_games: pd.DataFrame) -> list[tuple[str, str]]:
    if df_games.empty:
        return []
    if not {"away_abbrev", "home_abbrev"}.issubset(df_games.columns):
        return []

    matchups: list[tuple[str, str]] = []
    for g in df_games.itertuples(index=False):
        away = norm_team(getattr(g, "away_abbrev", None))
        home = norm_team(getattr(g, "home_abbrev", None))
        if away and home and away != "NAN" and home != "NAN":
            matchups.append((away, home))
    return matchups


def _load_local_slate_csv(path: Path) -> list[tuple[str, str]]:
    """
    Local fallback file: data/cache/todays_games.csv
    Must have columns: away_abbrev, home_abbrev
    """
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if not {"away_abbrev", "home_abbrev"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: away_abbrev, home_abbrev")
    pairs: list[tuple[str, str]] = []
    for r in df.itertuples(index=False):
        away = norm_team(getattr(r, "away_abbrev"))
        home = norm_team(getattr(r, "home_abbrev"))
        if away and home and away != "NAN" and home != "NAN":
            pairs.append((away, home))
    return pairs


def main(
    *,
    use_tomorrow: bool = False,
    rebuild_history: bool = False,
    # manual fallback if schedule + local slate both fail:
    away_team: str | None = None,
    home_team: str | None = None,
    game_date: str | None = None,
) -> None:
    # ----------------------------
    # Dates + output dir
    # ----------------------------
    schedule_dt = datetime.today() + (timedelta(days=1) if use_tomorrow else timedelta(days=0))
    schedule_date = schedule_dt.strftime("%Y-%m-%d")
    date_use = schedule_date if game_date is None else game_date

    out_dir = Path("results") / schedule_date / "fg3"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "_meta.txt"
    meta_path.write_text(f"[INFO] schedule_date={schedule_date}\n[INFO] game_date_used={date_use}\n")

    # ----------------------------
    # History load (once)
    # ----------------------------
    combined_path = Path(PATH_GAMLOGS_COMBINED) if not isinstance(PATH_GAMLOGS_COMBINED, Path) else PATH_GAMLOGS_COMBINED
    if rebuild_history:
        from model_training.data_loading import build_all_gamelogs_combined
        build_all_gamelogs_combined(write_combined_csv=True)

    if not combined_path.exists():
        raise FileNotFoundError(f"Missing combined gamelogs CSV: {combined_path}")

    history_df = pd.read_csv(combined_path, low_memory=False)

    # ----------------------------
    # Model artifacts
    # ----------------------------
    model_dir = Path(THREES_MODEL_DIR) if not isinstance(THREES_MODEL_DIR, Path) else THREES_MODEL_DIR
    fg3a_model_path = model_dir / "fg3a_model.joblib"
    fg3_rate_model_path = model_dir / "fg3_rate_model.joblib"
    features_path = model_dir / "features.joblib"  # compat

    for p in [fg3a_model_path, fg3_rate_model_path, features_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing model artifact: {p}")

    # ----------------------------
    # Matchups: stats -> local csv -> manual
    # ----------------------------
    df_games = _safe_get_games(schedule_dt)
    matchups = _matchups_from_games_df(df_games)

    if matchups:
        msg = f"[INFO] Using stats.nba.com schedule ({len(matchups)} games)\n"
        print(msg.strip())
        meta_path.write_text(meta_path.read_text() + msg)

    if not matchups:
        slate_csv = Path("./data/cache/todays_games.csv")
        matchups = _load_local_slate_csv(slate_csv)
        if matchups:
            msg = f"[INFO] Using local slate fallback: {slate_csv} ({len(matchups)} games)\n"
            print(msg.strip())
            meta_path.write_text(meta_path.read_text() + msg)

    if not matchups:
        if not away_team or not home_team:
            raise ValueError(
                "No schedule games available AND no local slate csv AND no manual matchup provided.\n"
                "Fix options:\n"
                "  1) Create data/cache/todays_games.csv with columns away_abbrev,home_abbrev\n"
                "  2) Run with manual: main(away_team='LAL', home_team='BOS')\n"
            )
        matchups = [(norm_team(away_team), norm_team(home_team))]
        msg = f"[INFO] Using manual matchup fallback: {matchups[0][0]}@{matchups[0][1]}\n"
        print(msg.strip())
        meta_path.write_text(meta_path.read_text() + msg)

    # ----------------------------
    # Run slate
    # ----------------------------
    all_res: list[pd.DataFrame] = []

    for away, home in matchups:
        print(f"[INFO] Predicting FG3 for {away}@{home} | game_date={date_use}")

        out = predict_game_fg3(
            history_df=history_df,
            away_team=away,
            home_team=home,
            game_date=date_use,
            fg3a_model_path=str(fg3a_model_path),
            fg3_rate_model_path=str(fg3_rate_model_path),
            threes_features_path=str(features_path),
        )

    sort_col = "pred_mean" if "pred_mean" in out.columns else "pred_fg3"

    out = out.sort_values([sort_col], ascending=False).reset_index(drop=True)
    out.to_csv(out_dir / f"{away}_at_{home}_fg3.csv", index=False)
    all_res.append(out)

    slate = pd.concat(all_res, ignore_index=True)
    slate.to_csv(out_dir / "all_matchups_fg3.csv", index=False)
    print(f"[INFO] Wrote slate -> {out_dir / 'all_matchups_fg3.csv'}")


if __name__ == "__main__":
    main(
        use_tomorrow=False,
        rebuild_history=False,
        # manual fallback:
        # away_team="LAL",
        # home_team="BOS",
    )