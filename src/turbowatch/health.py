"""Unsupervised health monitoring + calibrated risk.

1. An anomaly detector (Isolation Forest or a small autoencoder) is fitted on HEALTHY cycles of
   fit engines only (true RUL > `healthy_rul`).
2. Its raw score is smoothed causally (EWM per engine).
3. A logistic (Platt) calibrator, fitted on CALIBRATION engines, maps the smoothed score to
   P(failure within `horizon` cycles) — the "risk" shown in the dashboard. Health = 1 - risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn


class IFDetector:
    name = "isolation_forest"

    def __init__(self, seed: int = 0, n_estimators: int = 300):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray) -> IFDetector:
        self.model.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(self.scaler.transform(X))  # higher = more anomalous


class _AE(nn.Module):
    def __init__(self, d: int, h: int = 16, z: int = 4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z))
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class AEDetector:
    name = "autoencoder"

    def __init__(self, seed: int = 0, epochs: int = 40, lr: float = 3e-3):
        self.seed, self.epochs, self.lr = seed, epochs, lr
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> AEDetector:
        torch.manual_seed(self.seed)
        Xs = torch.from_numpy(self.scaler.fit_transform(X).astype(np.float32))
        self.net = _AE(Xs.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xs), batch_size=256, shuffle=True, generator=g)
        for _ in range(self.epochs):
            for (xb,) in dl:
                opt.zero_grad()
                nn.functional.mse_loss(self.net(xb), xb).backward()
                opt.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = torch.from_numpy(self.scaler.transform(X).astype(np.float32))
        return ((self.net(Xs) - Xs) ** 2).mean(1).numpy()


def smooth(scores: np.ndarray, units: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Causal EWM of the raw anomaly score within each engine (rows sorted by unit, cycle)."""
    s = pd.Series(scores)
    return s.groupby(units).transform(lambda x: x.ewm(alpha=alpha, adjust=True).mean()).to_numpy()


@dataclass
class RiskCalibrator:
    """Platt scaling: smoothed anomaly score -> P(true RUL <= horizon)."""
    horizon: int = 30

    def fit(self, score: np.ndarray, true_rul: np.ndarray) -> RiskCalibrator:
        y = (np.asarray(true_rul) <= self.horizon).astype(int)
        self.lr_ = LogisticRegression().fit(np.asarray(score).reshape(-1, 1), y)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(np.asarray(score).reshape(-1, 1))[:, 1]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))
