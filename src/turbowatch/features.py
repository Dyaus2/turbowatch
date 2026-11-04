"""Operating-regime normalisation + causal (past-only) features.

The SAME `FeaturePipeline` object is fitted on training engines only and then used for
validation, test, the API and the live replay — there is no separate serving code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .data import OPS, SENSORS

WINDOWS = (5, 15)


@dataclass
class FeaturePipeline:
    n_regimes: int = 1
    windows: tuple[int, ...] = WINDOWS
    min_std: float = 1e-4
    seed: int = 0
    sensors_: list[str] = field(default_factory=list)
    stats_: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    kmeans_: KMeans | None = None

    # ---------- fitting (train engines only) ----------
    def fit(self, train: pd.DataFrame) -> FeaturePipeline:
        reg = self._fit_regimes(train)
        normed_std = []
        self.stats_ = {}
        for r in range(self.n_regimes):
            x = train.loc[reg == r, SENSORS].to_numpy(float)
            mu, sd = x.mean(0), x.std(0)
            self.stats_[r] = (mu, np.where(sd < 1e-12, 1.0, sd))
            normed_std.append(np.where(sd < self.min_std, 0.0, 1.0))
        # keep sensors that vary within at least one regime
        keep = np.max(normed_std, axis=0) > 0
        self.sensors_ = [s for s, k in zip(SENSORS, keep, strict=True) if k]
        return self

    def _fit_regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_regimes == 1:
            self.kmeans_ = None
            return np.zeros(len(df), dtype=int)
        self.kmeans_ = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed)
        return self.kmeans_.fit_predict(df[OPS].to_numpy(float))

    def regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans_ is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(df[OPS].to_numpy(float))

    # ---------- transforms (any engine, any time) ----------
    def normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-regime z-score of the kept sensors (removes operating-condition effects)."""
        if not self.sensors_:
            raise RuntimeError("FeaturePipeline is not fitted")
        out = df[["unit", "cycle"]].copy()
        reg = self.regimes(df)
        x = df[SENSORS].to_numpy(float)
        z = np.empty_like(x)
        for r, (mu, sd) in self.stats_.items():
            m = reg == r
            z[m] = (x[m] - mu) / sd
        idx = [SENSORS.index(s) for s in self.sensors_]
        out[self.sensors_] = z[:, idx]
        out["regime"] = reg
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal features per row. Row t only uses cycles <= t of the same unit."""
        n = self.normalise(df.sort_values(["unit", "cycle"]).reset_index(drop=True))
        g = n.groupby("unit", group_keys=False)
        feats: dict[str, pd.Series | np.ndarray] = {"unit": n["unit"], "cycle": n["cycle"]}
        for s in self.sensors_:
            feats[s] = n[s]
            for w in self.windows:
                feats[f"{s}_mean{w}"] = g[s].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
                feats[f"{s}_std{w}"] = g[s].transform(lambda x, w=w: x.rolling(w, min_periods=2).std()).fillna(0.0)
            wl = self.windows[-1]
            feats[f"{s}_slope{wl}"] = (n[s] - g[s].transform(lambda x, wl=wl: x.shift(wl)).fillna(n[s])) / wl
            feats[f"{s}_ewm"] = g[s].transform(lambda x: x.ewm(alpha=0.1, adjust=True).mean())
        return pd.DataFrame(feats)

    @property
    def feature_names(self) -> list[str]:
        names = ["cycle"]
        for s in self.sensors_:
            names.append(s)
            for w in self.windows:
                names += [f"{s}_mean{w}", f"{s}_std{w}"]
            names += [f"{s}_slope{self.windows[-1]}", f"{s}_ewm"]
        return names


def last_rows(X: pd.DataFrame) -> pd.DataFrame:
    """Row at the last observed cycle of every unit (official test protocol)."""
    idx = X.groupby("unit")["cycle"].idxmax()
    return X.loc[idx].sort_values("unit").reset_index(drop=True)


def capped(rul: pd.Series | np.ndarray, cap: int = 125) -> np.ndarray:
    return np.minimum(np.asarray(rul), cap)
