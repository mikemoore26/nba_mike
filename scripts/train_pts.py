#scripts/train_model_pts.py
from model_training.points.tune import tune_and_train_points_models
from model_training.config import PATH_GAMLOGS_COMBINED, POINTS_MODEL_DIR
from model_training.data_loading import build_all_gamelogs_combined

if __name__ == "__main__":
    build_all_gamelogs_combined(
        seasons=None,               # or [2024, 2025, 2026]
        use_parquet=True,
        write_combined_csv=True,
    )
    tune_and_train_points_models(
        csv_path=str(PATH_GAMLOGS_COMBINED),
        model_dir=str(POINTS_MODEL_DIR),
        split_date="2025-01-01",
    )
