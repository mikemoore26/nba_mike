import pandas as pd
import joblib

from model_training.threes.features import build_features_no_leak, add_player_baselines

import numpy as np

PATH_GAMLOGS_COMBINED = './data/all_gamelogs_combined.csv'
PATH_TO_MODEL_dir = './models/threes/'

# ----------------------------
# 1) CEILING-AWARE EXPECTED FG3A
# ----------------------------
def expected_fg3a_ceiling(ph: pd.DataFrame, recent_n: int = 5) -> float:
    """
    Better than mean(last N):
    - base = trailing mean (stable)
    - ceiling = recent 80th percentile (captures spike behavior)
    - blend them so stars can pop
    - cap at recent 95th percentile
    """
    tail = ph["fg3a"].tail(max(10, recent_n * 2)).dropna()
    if tail.empty:
        return 0.0

    base = float(tail.tail(recent_n).mean())
    ceiling = float(tail.quantile(0.80))
    exp = 0.65 * base + 0.35 * ceiling

    cap = float(tail.quantile(0.95))
    return float(np.clip(exp, 0, cap))


# ----------------------------
# 2) COMPUTED RATE (NO RATE MODEL) + ELITE BOOST
# ----------------------------
def compute_final_rate(X_row: pd.Series, league_fg3_pct: float) -> float:
    """
    Stabilized rate:
    - Use bayesian-shrunk baseline so tiny samples don't dominate
    - Blend baseline + recent form + league
    """

    # Inputs from features
    player_pct = X_row.get("player_fg3_pct_season", np.nan)
    recent_form = X_row.get("fg3_pct_rolling_10", np.nan)

    # Bayesian shrinkage prior
    prior_pct = league_fg3_pct
    prior_att = 80.0  # bigger = more conservative early season

    # If player_pct missing, start at league
    if np.isnan(player_pct):
        player_pct = prior_pct

    # Stabilize recent_form too
    if np.isnan(recent_form):
        recent_form = player_pct

    # Clamp both to sane shooting range before blending
    player_pct = float(np.clip(player_pct, 0.20, 0.50))
    recent_form = float(np.clip(recent_form, 0.15, 0.60))

    # Blend (baseline dominates; recent is a smaller modifier)
    rate = (
        0.70 * player_pct +
        0.20 * recent_form +
        0.10 * prior_pct
    )

    # Optional elite bump ONLY if baseline is truly elite
    if player_pct >= 0.40:
        rate *= 1.03

    return float(np.clip(rate, 0.18, 0.45))


# ----------------------------
# 3) PREDICT A GAME: FG3A MODEL + COMPUTED RATE
# ----------------------------
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
    # Load FG3A model + features
    fg3a_pipe = joblib.load(fg3a_model_path)
    FEATURES = joblib.load(features_path)

    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values(["player", "date"])

    # latest team per player
    latest_team = (
        history.sort_values("date")
              .groupby("player")
              .tail(1)[["player", "team", "season"]]
    )
    players = latest_team[latest_team["team"].isin([away_team, home_team])]["player"].tolist()

    # require enough games
    game_counts = history.groupby("player").size()
    players = [p for p in players if game_counts.get(p, 0) >= min_games_required]

    # build today rows using recent averages (no fixed numbers)
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
            # upgraded attempts expectation
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

    # Append + rebuild features
    combined = pd.concat([history, today_df], ignore_index=True)

    # NOTE: These must exist in your environment
    combined = build_features_no_leak(combined)
    combined = add_player_baselines(combined)

    # Today feature rows
    X_today = combined.tail(len(today_df))[FEATURES].copy().reset_index(drop=True)
    today_df = today_df.reset_index(drop=True)

    # Loosen mask: only require minutes rolling exists
    min_required = ["min_rolling_5"]
    mask = X_today[min_required].notna().all(axis=1).to_numpy()

    X_ok = X_today.loc[mask].copy()
    out = today_df.loc[mask, ["player", "team", "opp", "is_home", "fg3a"]].copy()

    # -------------------------
    # Attempts: blend expected FG3A + model FG3A
    # -------------------------
    model_fg3a = np.clip(fg3a_pipe.predict(X_ok), 0, None)
    expected_fg3a = out["fg3a"].to_numpy()

    final_fg3a = (1.0 - fg3a_blend) * expected_fg3a + fg3a_blend * model_fg3a
    final_fg3a = np.clip(final_fg3a, 0, None)

    # -------------------------
    # Rate: computed, no rate model
    # -------------------------
    league_fg3_pct = history["fg3"].sum() / max(history["fg3a"].sum(), 1)

    final_rate = np.array([compute_final_rate(X_ok.iloc[i], league_fg3_pct) for i in range(len(X_ok))])

    # -------------------------
    # Final FG3
    # -------------------------
    out["pred_fg3a"] = final_fg3a
    out["pred_rate"] = final_rate
    out["pred_fg3"] = out["pred_fg3a"] * out["pred_rate"]

    out = out.drop(columns=["fg3a"])
    out = out.sort_values("pred_fg3", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    history = pd.read_csv(PATH_GAMLOGS_COMBINED, parse_dates=["date"])
    out = []
    date = "2025-01-26"
    matchups = [("ind", "atl"), ("phi", "cho"), ("orl", "cle"),
                     ("por", "bos"), ("lal", "chi"), ("mem", "hou"), ("gsw", "min")]
    

    for matchup in matchups:
        out = predict_game_fg3(
            history_df=history,
            away_team=matchup[0].upper(),
            home_team=matchup[1].upper(),
            game_date=date,
            fg3a_model_path=PATH_TO_MODEL_dir + "model_fg3a.joblib",
            features_path=PATH_TO_MODEL_dir + "features.joblib",
        )

        out.to_csv(f"results/predicted_fg3_{date}_{matchup[0]}_vs_{matchup[1]}.csv", index=False)