# model_training/threes/rate.py
import numpy as np
import pandas as pd

def compute_final_rate_bayes(X_row: pd.Series, league_fg3_pct: float) -> float:
    """
    Uses true bayesian shrinkage:
      base = (made + prior_made) / (att + prior_att)
    Then blends in recent form in a volume-aware way.
    Requires these features exist in X_row:
      - player_fg3_made_sum
      - player_fg3_att_sum
      - fg3_pct_rolling_10
    """
    made = X_row.get("player_fg3_made_sum", np.nan)
    att = X_row.get("player_fg3_att_sum", np.nan)
    recent_form = X_row.get("fg3_pct_rolling_10", np.nan)

    # prior
    prior_att = 80.0
    prior_made = prior_att * league_fg3_pct

    # shrunken baseline
    if np.isnan(att) or att <= 0 or np.isnan(made):
        base = league_fg3_pct
        att_val = 0.0
    else:
        base = (made + prior_made) / (att + prior_att)
        att_val = float(att)

    # recent fallback
    if np.isnan(recent_form):
        recent_form = base

    # clamp inputs
    base = float(np.clip(base, 0.20, 0.50))
    recent_form = float(np.clip(recent_form, 0.15, 0.60))

    # volume-aware blending: more attempts -> trust recent more
    vol_weight = float(np.clip(att_val / 200.0, 0.0, 1.0))  # att=200 => full
    w_recent = 0.10 + 0.20 * vol_weight                     # 0.10..0.30
    w_base = 0.85 - 0.20 * vol_weight                       # 0.85..0.65
    w_league = 1.0 - (w_base + w_recent)                    # remaining

    rate = (w_base * base) + (w_recent * recent_form) + (w_league * league_fg3_pct)

    # small elite bump if truly elite baseline
    if base >= 0.40:
        rate *= 1.02

    return float(np.clip(rate, 0.18, 0.45))

