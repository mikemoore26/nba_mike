from features import build_features_no_leak, add_player_baselines
from rate import compute_final_rate_bayes
from attempts import expected_fg3a_ceiling
import pandas as pd
import joblib
import numpy as np


def predict_game_fg3(
    history_df: pd.DataFrame,
    away_team: str,
    home_team: str,
    game_date,
    fg3a_model_path,
    features_path,
    min_games_required: int = 10,
    recent_n: int = 5,
    fg3a_blend: float = 0.25,   # 25% model FG3A, 75% expected FG3A
):
    fg3a_pipe = joblib.load(fg3a_model_path)
    FEATURES = joblib.load(features_path)

    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["player", "date"])

    # players on either team (based on latest team)
    latest_team = (
        history.sort_values("date")
              .groupby("player")
              .tail(1)[["player", "team", "season"]]
    )
    players = latest_team[latest_team["team"].isin([away_team, home_team])]["player"].tolist()

    # require enough games
    game_counts = history.groupby("player").size()
    players = [p for p in players if game_counts.get(p, 0) >= min_games_required]

    rows = []
    for p in players:
        ph = history[history["player"] == p].sort_values("date")
        prev = ph.iloc[-1]

        team = prev["team"]
        is_home = 1 if team == home_team else 0
        opp = away_team if is_home else home_team

        rows.append({
            "player": p,
            "season": prev["season"],
            "date": pd.Timestamp(game_date),
            "team": team,
            "opp": opp,

            "mp_minutes": float(ph["mp_minutes"].tail(recent_n).mean()),
            "fga": float(ph["fga"].tail(recent_n).mean()),
            "fg3a": expected_fg3a_ceiling(ph, recent_n=recent_n),
            "pts": float(ph["pts"].tail(recent_n).mean()),
            "usage": float(ph["usage"].tail(recent_n).mean()),

            "is_home": int(is_home),
            "starter_flag": int(prev.get("starter_flag", 1)),

            "fg3": np.nan,
        })

    today_df = pd.DataFrame(rows)
    if today_df.empty:
        raise ValueError("No eligible players found for this matchup.")

    combined = pd.concat([history, today_df], ignore_index=True)

    # Must exist in your environment
    combined = build_features_no_leak(combined)
    combined = add_player_baselines(combined)

    X_today = combined.tail(len(today_df))[FEATURES].copy().reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # require only minutes rolling (avoid over-filtering)
    mask = X_today[["min_rolling_5"]].notna().all(axis=1).to_numpy()

    X_ok = X_today.loc[mask].copy()
    out = today_df.loc[mask, ["player", "team", "opp", "is_home", "fg3a"]].copy()

    # --- FG3A: blend expected + model ---
    model_fg3a = np.clip(fg3a_pipe.predict(X_ok), 0, None)
    expected_fg3a = out["fg3a"].to_numpy()
    final_fg3a = (1.0 - fg3a_blend) * expected_fg3a + fg3a_blend * model_fg3a
    final_fg3a = np.clip(final_fg3a, 0, None)

    # --- Rate: Bayesian computed ---
    league_fg3_pct = history["fg3"].sum() / max(history["fg3a"].sum(), 1)
    final_rate = np.array([compute_final_rate_bayes(X_ok.iloc[i], league_fg3_pct) for i in range(len(X_ok))])

    out["pred_fg3a"] = final_fg3a
    out["pred_rate"] = final_rate
    out["pred_fg3"] = out["pred_fg3a"] * out["pred_rate"]

    out = out.drop(columns=["fg3a"])
    return out.sort_values("pred_fg3", ascending=False).reset_index(drop=True)

