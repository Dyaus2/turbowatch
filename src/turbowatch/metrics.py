"""RUL metrics: RMSE and the PHM08 asymmetric score (late predictions penalised more)."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def phm08(y_true, y_pred) -> float:
    """Saxena et al. 2008: d = pred - true; exp(-d/13)-1 if early, exp(d/10)-1 if late."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def interval_coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))
