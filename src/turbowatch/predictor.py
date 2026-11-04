"""Serving-side inference. Uses the SAME fitted FeaturePipeline as training (no re-implementation)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import health
from .explain import Explainer
from .models.cnn import CNNRul, make_windows

STATUS_WATCH_RISK = 0.3


class Predictor:
    def __init__(self, artifact_dir: str | Path):
        d = Path(artifact_dir)
        b = joblib.load(d / "bundle.joblib")
        self.cfg = b["config"]
        self.fp = b["features"]
        self.gbm = b["gbm"]
        self.detector = b["detector"]
        self.detector_name = b["detector_name"]
        self.calibrator = b["calibrator"]
        self.alarm_signal = b["alarm_signal"]
        self.policy = b["policy"]
        self.rul_test: dict[int, int] = b["rul_test"]
        self.cnn = CNNRul.from_state(torch.load(d / "cnn.pt", weights_only=False))
        self.explainer = Explainer(self.gbm.point_, self.fp.feature_names)
        self.cap = int(self.cfg["cap"])

    def timeline(self, engine: pd.DataFrame) -> pd.DataFrame:
        """Per-cycle predictions for ONE engine's history (rows: cycles observed so far)."""
        return self.timelines(engine).drop(columns="unit")

    def timelines(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-cycle predictions for many engines at once (identical to per-engine: all features are causal per unit)."""
        X = self.fp.transform(df)
        engine = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
        feats = self.fp.feature_names
        rul = self.gbm.predict(X[feats])
        lo, hi = self.gbm.predict_interval(X[feats])
        norm = self.fp.normalise(engine)
        Xw, _, _ = make_windows(norm, self.fp.sensors_, self.cnn.window)
        rul_cnn = self.cnn.predict(Xw)
        hcols = [f"{s}_mean{self.fp.windows[-1]}" for s in self.fp.sensors_]
        raw = self.detector.score(X[hcols].to_numpy(float))
        risk = self.calibrator.predict(health.smooth(raw, X["unit"].to_numpy()))
        signal = -rul if self.alarm_signal == "gbm_rul" else risk
        alarm = self.policy.alarms(signal, X["unit"].to_numpy())
        status = np.where(alarm, "alarm", np.where((risk >= STATUS_WATCH_RISK) | (rul <= 2 * self.cfg["horizon"]), "watch", "ok"))
        return pd.DataFrame({"unit": X["unit"], "cycle": X["cycle"], "rul": rul, "rul_lo": lo, "rul_hi": hi, "rul_cnn": rul_cnn,
                             "risk": risk, "health": 1 - risk, "alarm": alarm, "status": status})

    def explain_last(self, engine: pd.DataFrame, k: int = 5) -> dict:
        X = self.fp.transform(engine.sort_values("cycle").reset_index(drop=True))
        return self.explain_row(X.iloc[[-1]], k)

    def explain_row(self, X_row: pd.DataFrame, k: int = 5) -> dict:
        return {"base_value": self.explainer.base_value, "factors": self.explainer.top_factors(X_row, k)}
