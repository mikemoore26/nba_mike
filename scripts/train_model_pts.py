from model_training.points.tune import tune_and_train_points_models
from model_training.config import PATH_GAMLOGS_COMBINED, POINTS_MODEL_DIR

if __name__ == "__main__":
    tune_and_train_points_models(
        csv_path=str(PATH_GAMLOGS_COMBINED),
        model_dir=str(POINTS_MODEL_DIR),
        split_date="2025-01-01",
    )
