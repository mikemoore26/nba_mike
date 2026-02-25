# scripts/train_fg3.py
from __future__ import annotations

from pathlib import Path

from model_training.config import (
    PATH_GAMLOGS_COMBINED,
    THREES_MODEL_DIR
    ,   # if you already have this as a base models dir
)

# from model_training.data_loading import UPDATE_ALL_GAMLOGS

from model_training.threes.tune import tune_and_train_threes_models
from model_training.data_loading import build_all_gamelogs_combined


def main():
    # 1) Make sure gamelogs are current
    build_all_gamelogs_combined(
        seasons=None,               # or [2024, 2025, 2026]
        use_parquet=True,
        write_combined_csv=True,
    )
    # 2) Decide model output directory
    # You can either use your config base dir or hardcode ./models/threes
    # This keeps it consistent with your predict script paths.
    model_dir = Path(THREES_MODEL_DIR) if not isinstance(THREES_MODEL_DIR, Path) else THREES_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # 3) Train + tune via grid search (Poisson FG3A + Logit Rate)
    tune_and_train_threes_models(
        csv_path=str(PATH_GAMLOGS_COMBINED),
        model_dir=str(model_dir),
        split_date="2025-01-01",
        save_features_artifact=True,
    )


if __name__ == "__main__":
    main()