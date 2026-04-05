from pathlib import Path
import pandas as pd


PRED_PATH = Path("results/2026-04-03/pred_reb.csv")  # change date
HIST_PATH = Path("data/all_gamelogs_combined.csv")


def canon_date(df):
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    else:
        df["game_date"] = pd.to_datetime(df["date"])
    return df


pred = pd.read_csv(PRED_PATH, low_memory=False)
hist = pd.read_csv(HIST_PATH, low_memory=False)

pred = canon_date(pred)
hist = canon_date(hist)

pred["player"] = pred["player"].str.strip()
hist["player"] = hist["player"].str.strip()

df = pred.merge(
    hist[["game_date", "player", "reb"]],
    on=["game_date", "player"],
    how="inner",
)

print(f"\nMerged rows: {len(df)}")

if len(df) == 0:
    raise ValueError("Still empty → wrong date or using today's slate")

df["reb_ge_10_actual"] = (df["reb"] >= 10).astype(int)
df["reb_ge_12_actual"] = (df["reb"] >= 12).astype(int)

def summarize(pred_col, actual_col):
    return {
        "mean_pred": df[pred_col].mean(),
        "actual_rate": df[actual_col].mean(),
        "diff": df[pred_col].mean() - df[actual_col].mean(),
    }

print("\n=== FINAL MODEL (AFTER BLENDING) ===")

print("\nREB >= 10")
print(summarize("p_reb_ge_10", "reb_ge_10_actual"))

print("\nREB >= 12")
print(summarize("p_reb_ge_12", "reb_ge_12_actual"))