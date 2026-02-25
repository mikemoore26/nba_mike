# scripts/train_model_reb.py
from __future__ import annotations

from pathlib import Path

from model_training.config import PATH_GAMLOGS_COMBINED, REBOUNDS_MODEL_DIR
from model_training.data_loading import build_all_gamelogs_combined
from model_training.rebounds.train_model import train_models


def main() -> None:
    # 1) Build combined dataset used by training (writes CSV)
    build_all_gamelogs_combined(
        seasons=None,               # or [2024, 2025, 2026]
        use_parquet=True,
        write_combined_csv=True,
    )

    # 2) Train model(s)
    model_dir = Path(REBOUNDS_MODEL_DIR) if not isinstance(REBOUNDS_MODEL_DIR, Path) else REBOUNDS_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    train_models(
        csv_path=str(PATH_GAMLOGS_COMBINED),
        reb_model_path=str(model_dir / "reb_model.joblib"),
        features_path=str(model_dir / "features.joblib"),
        split_date="2025-01-01",
    )


if __name__ == "__main__":
    main()
