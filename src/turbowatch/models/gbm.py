"""Gradient-boosting RUL baseline with a quantile uncertainty band (P10–P90)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _hgb(seed: int, **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed, **kw)


@dataclass
class GBMRul:
    seed: int = 0
    quantiles: tuple[float, float] = (0.1, 0.9)
    features_: list[str] = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> GBMRul:
        self.features_ = list(X.columns)
        self.point_ = _hgb(self.seed).fit(X, y)
        self.lo_ = _hgb(self.seed, loss="quantile", quantile=self.quantiles[0]).fit(X, y)
        self.hi_ = _hgb(self.seed, loss="quantile", quantile=self.quantiles[1]).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.point_.predict(X[self.features_]), 0, None)

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        lo = np.clip(self.lo_.predict(X[self.features_]), 0, None)
        hi = np.clip(self.hi_.predict(X[self.features_]), 0, None)
        return np.minimum(lo, hi), np.maximum(lo, hi)
