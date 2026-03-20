from __future__ import annotations

from model_training.config import PATH_GAMLOGS_COMBINED, REBOUNDS_MODEL_DIR
from model_training.rebounds.train_model import main


if __name__ == "__main__":
    main(
        csv_path=PATH_GAMLOGS_COMBINED,
        model_dir=REBOUNDS_MODEL_DIR,
        split_date="2025-01-01",
    )