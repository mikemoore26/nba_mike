# model_training/threes/rate_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer


@dataclass
class BinomialGLMRateModel:
    """
    Statsmodels GLM Binomial model that predicts p3 (make probability).
    Fit: y = fg3/fg3a with var_weights = fg3a.
    """
    imputer: SimpleImputer
    result_: Any
    feature_names_: List[str]

    @classmethod
    def fit_from_df(cls, df: pd.DataFrame, feature_cols: List[str]) -> "BinomialGLMRateModel":
        tr = df[df["fg3a"] > 0].copy()

        X = tr[feature_cols]
        y = (tr["fg3"] / tr["fg3a"]).astype(float)
        n = tr["fg3a"].astype(float)

        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        X_imp = sm.add_constant(X_imp, has_constant="add")

        glm = sm.GLM(y, X_imp, family=sm.families.Binomial(), var_weights=n)
        res = glm.fit()

        return cls(imputer=imp, result_=res, feature_names_=feature_cols)

    def predict_p(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names_]
        X_imp = self.imputer.transform(X)
        X_imp = sm.add_constant(X_imp, has_constant="add")
        p = self.result_.predict(X_imp)
        return np.clip(p, 0.0, 1.0)
