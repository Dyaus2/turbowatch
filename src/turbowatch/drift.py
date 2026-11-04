"""Population Stability Index per sensor (bins = reference quantiles) for monitoring the incoming stream."""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WATCH, PSI_ALERT = 0.1, 0.25  # common rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 significant


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref) + eps
    q = np.histogram(cur, edges)[0] / max(len(cur), 1) + eps
    return float(np.sum((q - p) * np.log(q / p)))


class DriftMonitor:
    """Reference = per-regime-normalised sensors of the training engines; current = a recent window of the stream."""

    def __init__(self, reference: pd.DataFrame, sensors: list[str]):
        self.sensors = sensors
        self.ref = {s: reference[s].to_numpy(float) for s in sensors}

    def report(self, current: pd.DataFrame) -> list[dict]:
        out = []
        for s in self.sensors:
            v = psi(self.ref[s], current[s].to_numpy(float))
            out.append({"sensor": s, "psi": round(v, 4), "level": "alert" if v >= PSI_ALERT else "watch" if v >= PSI_WATCH else "ok"})
        return sorted(out, key=lambda d: -d["psi"])
