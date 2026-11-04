"""k-of-n persistence alarm policy tuned on a cost matrix, plus held-out alarm metrics.

Event definitions (per engine, true RUL known for every cycle):
- danger zone: true RUL <= horizon (default 30 cycles)  -> an alarm here is useful;
- false alarm: an alarm raised while true RUL > early_ok (default 60) -> premature, costs a
  needless inspection;
- an engine whose observed window reaches the danger zone and never alarms there or earlier
  inside the acceptable window -> missed failure.
Lead time = true RUL at the first useful alarm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    threshold: float
    k: int
    n: int

    def alarms(self, signal: np.ndarray, units: np.ndarray) -> np.ndarray:
        """Row is in alarm if >= k of the last n rows of the same engine exceed the threshold."""
        over = pd.Series((np.asarray(signal) >= self.threshold).astype(int))
        cnt = over.groupby(np.asarray(units)).transform(lambda x: x.rolling(self.n, min_periods=1).sum())
        return (cnt.to_numpy() >= self.k)


@dataclass(frozen=True)
class Costs:
    missed_failure: float = 100.0
    false_alarm: float = 10.0


def engine_outcomes(units: np.ndarray, true_rul: np.ndarray, alarm: np.ndarray,
                    horizon: int = 30, early_ok: int = 60) -> pd.DataFrame:
    df = pd.DataFrame({"unit": units, "rul": true_rul, "alarm": alarm})
    rows = []
    for u, g in df.groupby("unit"):
        reaches_danger = bool((g["rul"] <= horizon).any())
        false = bool((g["alarm"] & (g["rul"] > early_ok)).any())
        useful = g[g["alarm"] & (g["rul"] <= early_ok)]
        detected = reaches_danger and len(useful) > 0
        lead = float(useful["rul"].max()) if detected else np.nan
        rows.append({"unit": u, "positive": reaches_danger, "false_alarm": false, "detected": detected, "lead": lead})
    return pd.DataFrame(rows)


def summarize(out: pd.DataFrame) -> dict:
    pos = out[out["positive"]]
    return {
        "engines": int(len(out)),
        "positive_engines": int(len(pos)),
        "detection_rate": float(pos["detected"].mean()) if len(pos) else float("nan"),
        "false_alarm_rate": float(out["false_alarm"].mean()),
        "median_lead_cycles": float(pos.loc[pos["detected"], "lead"].median()) if pos["detected"].any() else float("nan"),
        "missed": int((pos["detected"] == 0).sum()),
    }


def cost(out: pd.DataFrame, costs: Costs) -> float:
    pos = out[out["positive"]]
    return float((~pos["detected"]).sum() * costs.missed_failure + out["false_alarm"].sum() * costs.false_alarm)


def tune(signal: np.ndarray, units: np.ndarray, true_rul: np.ndarray, costs: Costs = Costs(),
         thresholds: np.ndarray | None = None, kn: tuple[tuple[int, int], ...] = ((1, 1), (2, 3), (3, 5), (5, 8)),
         horizon: int = 30, early_ok: int = 60, max_false_alarm_rate: float | None = None) -> tuple[Policy, float]:
    """Grid search on CALIBRATION engines. Optionally constrain the false-alarm rate."""
    if thresholds is None:
        thresholds = np.unique(np.quantile(signal, np.linspace(0.5, 0.995, 60)))
    best, best_c = None, np.inf
    for k, n in kn:
        for t in thresholds:
            p = Policy(float(t), k, n)
            out = engine_outcomes(units, true_rul, p.alarms(signal, units), horizon, early_ok)
            if max_false_alarm_rate is not None and out["false_alarm"].mean() > max_false_alarm_rate:
                continue
            c = cost(out, costs)
            if c < best_c:
                best, best_c = p, c
    if best is None:  # constraint impossible -> most conservative
        best = Policy(float(np.max(thresholds)), kn[-1][0], kn[-1][1])
        best_c = cost(engine_outcomes(units, true_rul, best.alarms(signal, units), horizon, early_ok), costs)
    return best, best_c
