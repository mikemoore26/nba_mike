import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from model_training.threes.features import build_features_no_leak, add_player_baselines


# ----------------------------
# Model builders
# ----------------------------
def make_hgbr():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=600,
            random_state=42
        ))
    ])


# ----------------------------
# Time split (no leakage)
# ----------------------------
def time_split(df: pd.DataFrame, split_date: str = "2025-01-01"):
    split_date = pd.Timestamp(split_date)
    train = df[df["date"] < split_date].copy()
    valid = df[df["date"] >= split_date].copy()
    return train, valid


# ----------------------------
# Train both models
# ----------------------------
def train_models(
    csv_path,
    fg3a_model_path,
    fg3_model_path,
    features_path,
    split_date="2025-01-01",
):
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # You MUST have these in df already or create them before features:
    # df["mp_minutes"], df["starter_flag"], df["is_home"], etc.
    df = build_features_no_leak(df)
    df = add_player_baselines(df)

    features = joblib.load(features_path)

    # Keep only rows where labels exist
    df = df.dropna(subset=features + ["fg3a", "fg3"]).copy()
    df = df[df["fg3a"] >= 0].copy()

    train_df, valid_df = time_split(df, split_date=split_date)

    # ----------------------------
    # Model A: FG3A
    # ----------------------------
    X_tr_a, y_tr_a = train_df[features], train_df["fg3a"]
    X_va_a, y_va_a = valid_df[features], valid_df["fg3a"]

    fg3a_pipe = make_hgbr()
    fg3a_pipe.fit(X_tr_a, y_tr_a)

    pred_a = np.clip(fg3a_pipe.predict(X_va_a), 0, None)
    print("FG3A MAE:", mean_absolute_error(y_va_a, pred_a))

    # ----------------------------
    # Model B: FG3 (made threes)
    # ----------------------------
    X_tr_b, y_tr_b = train_df[features], train_df["fg3"]
    X_va_b, y_va_b = valid_df[features], valid_df["fg3"]

    fg3_pipe = make_hgbr()
    fg3_pipe.fit(X_tr_b, y_tr_b)

    pred_b = np.clip(fg3_pipe.predict(X_va_b), 0, None)
    print("FG3  MAE:", mean_absolute_error(y_va_b, pred_b))

    # Save
    joblib.dump(fg3a_pipe, fg3a_model_path)
    joblib.dump(fg3_pipe, fg3_model_path)
    joblib.dump(features, features_path)

    print("\nSaved:")
    print(" -", fg3a_model_path)
    print(" -", fg3_model_path)
    print(" -", features_path)

    return fg3a_pipe, fg3_pipe


if __name__ == "__main__":

    PATH_GAMLOGS_COMBINED = './data/all_gamelogs_combined.csv'
    PATH_TO_MODEL_dir = './models/threes/'
    
    train_models(
        csv_path=PATH_GAMLOGS_COMBINED,
        fg3a_model_path=PATH_TO_MODEL_dir + "fg3a_model.joblib",
        fg3_model_path=PATH_TO_MODEL_dir + "fg3_model.joblib",
        features_path=PATH_TO_MODEL_dir + "features.joblib",
        split_date="2025-01-01",
    )