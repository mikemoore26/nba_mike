# pts/rebounds/models.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from .features import REB_FEATURES, OPTIONAL_REB_FEATURES



def make_reb_hgbr(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.05,
            max_iter=600,
            random_state=seed
        )),
    ])



def _feature_cols(df: pd.DataFrame) -> list[str]:
    cols = list(REB_FEATURES)
    cols += [c for c in OPTIONAL_REB_FEATURES if c in df.columns]
    return cols


def model_reb(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    feat = _feature_cols(train_df)

    missing = sorted(set(feat) - set(train_df.columns))
    if missing:
        raise ValueError(f"[REB] Missing feature columns: {missing}")

    pipe = make_hgbr()
    pipe.fit(train_df[feat], train_df["reb"])

    pred = np.clip(pipe.predict(valid_df[feat]), 0, None)
    print("REB MAE:", mean_absolute_error(valid_df["reb"], pred))

    return pipe, feat
