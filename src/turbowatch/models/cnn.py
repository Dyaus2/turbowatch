"""1D-CNN sequence model for RUL (PyTorch). Input: last `window` cycles of normalised sensors."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn


def make_windows(norm: pd.DataFrame, sensors: list[str], window: int, y: np.ndarray | None = None,
                 last_only: bool = False) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Causal windows ending at each row (left-padded with the unit's first row).

    `norm` must be sorted by (unit, cycle). Returns X [n, window, n_sensors], y, and the
    row index each window ends at.
    """
    xs, ys, ends = [], [], []
    for _, g in norm.groupby("unit", sort=True):
        a = g[sensors].to_numpy(np.float32)
        pad = np.repeat(a[:1], window - 1, axis=0)
        a2 = np.vstack([pad, a])
        idx = g.index.to_numpy()
        rng = [len(a) - 1] if last_only else range(len(a))
        for i in rng:
            xs.append(a2[i:i + window])
            ends.append(idx[i])
            if y is not None:
                ys.append(y[idx[i]])
    X = np.stack(xs).astype(np.float32)
    return X, (np.asarray(ys, np.float32) if y is not None else None), np.asarray(ends)


class Net(nn.Module):
    def __init__(self, n_in: int, width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_in, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.LazyLinear(64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [b, window, n_in]
        return self.head(self.conv(x.transpose(1, 2))).squeeze(-1)


@dataclass
class CNNRul:
    window: int = 30
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    history_: list[float] = field(default_factory=list)

    def fit(self, X: np.ndarray, y: np.ndarray) -> CNNRul:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.n_in_ = X.shape[2]
        self.net_ = Net(self.n_in_).to(self.device)
        self.net_(torch.zeros(1, self.window, self.n_in_, device=self.device))  # materialise LazyLinear
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        g = torch.Generator().manual_seed(self.seed)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=g)
        self.history_ = []
        for _ in range(self.epochs):
            self.net_.train()
            tot, n = 0.0, 0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.net_(xb), yb)
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
                n += len(xb)
            sched.step()
            self.history_.append(tot / n)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net_.eval()
        out = []
        for i in range(0, len(X), 2048):
            out.append(self.net_(torch.from_numpy(X[i:i + 2048]).to(self.device)).cpu().numpy())
        return np.clip(np.concatenate(out), 0, None)

    def state(self) -> dict:
        sd = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
        return {"window": self.window, "n_in": self.n_in_, "state_dict": sd}

    @classmethod
    def from_state(cls, st: dict) -> CNNRul:
        m = cls(window=st["window"])
        m.n_in_ = st["n_in"]
        m.net_ = Net(m.n_in_)
        m.net_(torch.zeros(1, m.window, m.n_in_))
        m.net_.load_state_dict(st["state_dict"])
        m.net_.eval()
        return m
