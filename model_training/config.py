# model_training/config.py
from __future__ import annotations

from pathlib import Path

PATH_GAMLOGS_COMBINED = Path("./data/all_gamelogs_combined.csv")
GAMELOG_PARQUET_ROOT = Path("./data/gamelogs/gamelogs_parquet/")

# model_training/config.py
from pathlib import Path

PATH_GAMLOGS_COMBINED = Path("./data/all_gamelogs_combined.csv")
GAMELOG_PARQUET_ROOT = Path("./data/gamelogs/gamelogs_parquet/")

# 👇 ADD THIS BACK
PATH_TO_MODEL_dir = Path("./models/threes/")

