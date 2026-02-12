# scripts/train_models_3p.py
from __future__ import annotations

from pathlib import Path

from model_training.config import PATH_GAMLOGS_COMBINED, PATH_TO_MODEL_dir
from model_training.data_loading import build_all_gamelogs_combined
from model_training.threes.train_model import train_models


def main() -> None:
    # 1) Build combined dataset used by training (writes CSV)
    build_all_gamelogs_combined(
        seasons=None,               # or [2024, 2025, 2026]
        use_parquet=True,
        write_combined_csv=True,
    )

    # 2) Train models
    model_dir = Path(PATH_TO_MODEL_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_models(
        csv_path=str(PATH_GAMLOGS_COMBINED),
        fg3a_model_path=str(model_dir / "fg3a_model.joblib"),
        fg3_rate_model_path=str(model_dir / "fg3_rate_model.joblib"),
        features_path=str(model_dir / "features.joblib"),
        split_date="2025-01-01",
    )


if __name__ == "__main__":
    main()
