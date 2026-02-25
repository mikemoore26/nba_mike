# model_training/config.py
from __future__ import annotations
from pathlib import Path

# -----------------------------
# Data
# -----------------------------
PATH_GAMLOGS_COMBINED = Path("./data/all_gamelogs_combined.csv")
GAMELOG_PARQUET_ROOT = Path("./data/gamelogs/gamelogs_parquet/")

# -----------------------------
# Model Directories
# -----------------------------
MODELS_ROOT = Path("./models")

THREES_MODEL_DIR = MODELS_ROOT / "threes"
POINTS_MODEL_DIR = MODELS_ROOT / "points"
REBOUNDS_MODEL_DIR = MODELS_ROOT / "rebounds"
