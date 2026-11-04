"""Engine-grouped splits: an engine is in exactly one of fit / calibration / holdout."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EngineSplit:
    fit: tuple[int, ...]
    calib: tuple[int, ...]
    holdout: tuple[int, ...]

    def frame(self, df: pd.DataFrame, part: str) -> pd.DataFrame:
        units = set(getattr(self, part))
        return df[df["unit"].isin(units)].reset_index(drop=True)


def split_engines(units, fit: float = 0.7, calib: float = 0.15, seed: int = 0) -> EngineSplit:
    u = np.array(sorted(set(int(x) for x in units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(u)
    n_fit = int(round(fit * len(u)))
    n_cal = int(round(calib * len(u)))
    return EngineSplit(tuple(sorted(u[:n_fit])), tuple(sorted(u[n_fit:n_fit + n_cal])), tuple(sorted(u[n_fit + n_cal:])))
