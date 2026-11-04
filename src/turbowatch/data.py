"""NASA C-MAPSS loading, validation and a small synthetic stand-in for tests.

Source: NASA Prognostics Center of Excellence data repository, "Turbofan Engine Degradation
Simulation Data Set" (Saxena et al., 2008, doi:10.1109/PHM.2008.4711414), public mirror:
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
OPS = ["op1", "op2", "op3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *OPS, *SENSORS]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass
class Subset:
    name: str
    train: pd.DataFrame  # run-to-failure trajectories
    test: pd.DataFrame  # trajectories truncated before failure
    rul_test: np.ndarray  # true RUL at the last test cycle of each unit (sorted by unit)


def download(dest: str | Path = "data/raw", url: str = CMAPSS_URL) -> Path:
    """Download + extract the C-MAPSS text files (idempotent)."""
    dest = Path(dest)
    if (dest / "train_FD001.txt").exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=180) as r:
        outer = zipfile.ZipFile(io.BytesIO(r.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for n in inner.namelist():
        if n.endswith(".txt"):
            (dest / Path(n).name).write_bytes(inner.read(n))
    return dest


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df


def validate(df: pd.DataFrame) -> None:
    """Schema + integrity checks; raises ValueError with a clear message."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[COLUMNS].isna().any().any():
        raise ValueError("NaN values in C-MAPSS frame")
    for u, g in df.groupby("unit"):
        c = g["cycle"].to_numpy()
        if c[0] != 1 or not np.all(np.diff(c) == 1):
            raise ValueError(f"unit {u}: cycles must start at 1 and increase by 1")


def load(name: str = "FD001", root: str | Path = "data/raw") -> Subset:
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name}")
    root = Path(root)
    train, test = _read(root / f"train_{name}.txt"), _read(root / f"test_{name}.txt")
    rul = pd.read_csv(root / f"RUL_{name}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    validate(train)
    validate(test)
    if len(rul) != test["unit"].nunique():
        raise ValueError("RUL file length does not match number of test units")
    return Subset(name, train, test, rul)


def add_true_rul(df: pd.DataFrame, rul_at_end: dict[int, int] | None = None) -> pd.Series:
    """True (uncapped) RUL for every row.

    Train (run-to-failure): RUL = last_cycle - cycle.
    Test: RUL = rul_at_end[unit] + last_cycle - cycle.
    """
    last = df.groupby("unit")["cycle"].transform("max")
    rul = last - df["cycle"]
    if rul_at_end is not None:
        rul = rul + df["unit"].map(rul_at_end)
    return rul.astype(int)


def test_rul_map(sub: Subset) -> dict[int, int]:
    units = sorted(sub.test["unit"].unique())
    return {int(u): int(r) for u, r in zip(units, sub.rul_test, strict=True)}


def make_synthetic(n_units: int = 24, seed: int = 0, n_regimes: int = 1) -> pd.DataFrame:
    """Run-to-failure data with the C-MAPSS schema (tests only — not a benchmark)."""
    rng = np.random.default_rng(seed)
    regime_ops = rng.uniform([-1, -1, 60], [1, 1, 100], size=(n_regimes, 3))
    regime_off = rng.normal(0, 3, size=(n_regimes, 21))
    informative = {2: 0.8, 3: 1.0, 4: 1.2, 7: -0.9, 11: 1.1, 12: -0.8, 15: 0.9, 17: 0.7, 20: -0.6, 21: -0.7}
    rows = []
    for u in range(1, n_units + 1):
        life = int(rng.integers(130, 260))
        onset = rng.uniform(0.3, 0.6) * life
        base = rng.normal(0, 1, 21)
        for c in range(1, life + 1):
            r = int(rng.integers(n_regimes))
            deg = np.exp(max(0.0, c - onset) / (life - onset) * 3.0) - 1.0
            s = base + regime_off[r] + rng.normal(0, 0.25, 21)
            for k, w in informative.items():
                s[k - 1] += w * deg / 6.0
            s[0] = 518.67 + regime_off[r][0]
            ops = regime_ops[r] + rng.normal(0, 0.001, 3)
            rows.append([u, c, *ops, *s])
    return pd.DataFrame(rows, columns=COLUMNS)


def truncate(df: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Mimic the official test protocol: cut each unit at a random point; return remaining RUL."""
    rng = np.random.default_rng(seed)
    parts, rul = [], []
    for _, g in df.groupby("unit"):
        cut = int(rng.integers(int(0.3 * len(g)), len(g)))
        parts.append(g.iloc[:cut])
        rul.append(len(g) - cut)
    return pd.concat(parts, ignore_index=True), np.array(rul)
