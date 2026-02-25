from pathlib import Path
import pandas as pd

from model_training.config import PATH_GAMLOGS_COMBINED, THREES_MODEL_DIR
from model_training.threes.predict import predict_game_fg3


def main():
    history_df = pd.read_csv(PATH_GAMLOGS_COMBINED, low_memory=False)

    out = predict_game_fg3(
        history_df=history_df,
        away_team="LAL",
        home_team="BOS",
        game_date="2026-02-20",
        fg3a_model_path=str(Path(THREES_MODEL_DIR) / "fg3a_model.joblib"),
        fg3_rate_model_path=str(Path(THREES_MODEL_DIR) / "fg3_rate_model.joblib"),
        threes_features_path=str(Path(THREES_MODEL_DIR) / "features.joblib"),
    )

    print(out.head(20))
    out.to_csv("fg3_test_output.csv", index=False)
    print(out.columns.tolist())


if __name__ == "__main__":
    main()