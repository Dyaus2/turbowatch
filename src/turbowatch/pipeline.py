"""End-to-end training + honest held-out evaluation for one C-MAPSS subset.

Engines of the official TRAIN file are split (grouped by engine) into
  fit (70%)   -> feature pipeline, RUL models, anomaly detectors
  calib (15%) -> risk calibration + alarm-policy tuning
  holdout (15%) -> reporting (full run-to-failure trajectories)
and the official TEST file (truncated engines + true RUL) is used for reporting only.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from . import alarm, data, health, metrics
from .explain import Explainer
from .features import FeaturePipeline, capped, last_rows
from .models.cnn import CNNRul, make_windows
from .models.gbm import GBMRul
from .split import split_engines

N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass
class Config:
    subset: str = "FD001"
    data_dir: str = "data/raw"
    out_dir: str = "artifacts"
    seed: int = 0
    cap: int = 125
    horizon: int = 30
    early_ok: int = 60
    healthy_rul: int = 125
    cnn_window: int = 30
    cnn_epochs: int = 20
    ae_epochs: int = 40
    cost_missed: float = 100.0
    cost_false_alarm: float = 10.0
    max_far: float = 0.05  # false-alarm-rate constraint for the "detection @ fixed FAR" operating point
    device: str = "cpu"  # "cuda" on Colab T4 (CNN training only)
    mlflow: bool = True


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def _health_inputs(fp: FeaturePipeline, X: pd.DataFrame) -> np.ndarray:
    cols = [f"{s}_mean{fp.windows[-1]}" for s in fp.sensors_]
    return X[cols].to_numpy(float)


def run(cfg: Config, train_df: pd.DataFrame | None = None, test_df: pd.DataFrame | None = None,
        rul_test: np.ndarray | None = None) -> dict:
    t0 = time.time()
    if train_df is None:
        sub = data.load(cfg.subset, cfg.data_dir)
        train_df, test_df, rul_test = sub.train, sub.test, sub.rul_test
    train_df, test_df = _sorted(train_df), _sorted(test_df)
    test_units = sorted(test_df["unit"].unique())
    rul_map = {int(u): int(r) for u, r in zip(test_units, rul_test, strict=True)}

    split = split_engines(train_df["unit"].unique(), seed=cfg.seed)
    parts = {p: split.frame(train_df, p) for p in ("fit", "calib", "holdout")}
    true = {p: data.add_true_rul(d).to_numpy() for p, d in parts.items()}
    true["test"] = data.add_true_rul(test_df, rul_map).to_numpy()
    parts["test"] = test_df

    # ---- features (fitted on FIT engines only) ----
    fp = FeaturePipeline(n_regimes=N_REGIMES.get(cfg.subset, 1), seed=cfg.seed).fit(parts["fit"])
    X = {p: fp.transform(d) for p, d in parts.items()}
    feats = fp.feature_names

    # ---- RUL: gradient boosting + quantile band ----
    gbm = GBMRul(seed=cfg.seed).fit(X["fit"][feats], capped(true["fit"], cfg.cap))
    res: dict = {"subset": cfg.subset, "n_engines": {p: int(parts[p]["unit"].nunique()) for p in parts}}
    Xt_last = last_rows(X["test"])
    y_test = capped(rul_test, cfg.cap)
    g_pred = gbm.predict(Xt_last)
    g_lo, g_hi = gbm.predict_interval(Xt_last)
    g_hold = gbm.predict(X["holdout"])
    res["gbm"] = {
        "test_rmse": metrics.rmse(y_test, g_pred), "test_phm08": metrics.phm08(y_test, g_pred),
        "test_p10_p90_coverage": metrics.interval_coverage(y_test, g_lo, g_hi),
        "holdout_rmse_all_cycles": metrics.rmse(capped(true["holdout"], cfg.cap), g_hold),
    }

    # ---- RUL: 1D-CNN on normalised sensor windows ----
    norm = {p: fp.normalise(d) for p, d in parts.items()}
    Xw, yw, _ = make_windows(norm["fit"], fp.sensors_, cfg.cnn_window, capped(true["fit"], cfg.cap).astype(np.float32))
    cnn = CNNRul(window=cfg.cnn_window, epochs=cfg.cnn_epochs, seed=cfg.seed, device=cfg.device).fit(Xw, yw)
    Xtw, _, _ = make_windows(norm["test"], fp.sensors_, cfg.cnn_window, last_only=True)
    Xhw, _, _ = make_windows(norm["holdout"], fp.sensors_, cfg.cnn_window)
    c_pred, c_hold = cnn.predict(Xtw), cnn.predict(Xhw)
    res["cnn"] = {"test_rmse": metrics.rmse(y_test, c_pred), "test_phm08": metrics.phm08(y_test, c_pred),
                  "holdout_rmse_all_cycles": metrics.rmse(capped(true["holdout"], cfg.cap), c_hold),
                  "epochs": cfg.cnn_epochs, "final_train_mse": cnn.history_[-1]}

    # ---- health: anomaly detectors on healthy FIT cycles -> calibrated risk ----
    H = {p: _health_inputs(fp, X[p]) for p in X}
    healthy = true["fit"] > cfg.healthy_rul
    costs = alarm.Costs(cfg.cost_missed, cfg.cost_false_alarm)
    detectors = {"isolation_forest": health.IFDetector(seed=cfg.seed), "autoencoder": health.AEDetector(seed=cfg.seed, epochs=cfg.ae_epochs)}
    signals: dict[str, dict[str, np.ndarray]] = {}
    res["health"] = {}
    calibrators = {}
    for name, det in detectors.items():
        det.fit(H["fit"][healthy])
        sm = {p: health.smooth(det.score(H[p]), X[p]["unit"].to_numpy()) for p in ("calib", "holdout", "test")}
        cal = health.RiskCalibrator(cfg.horizon).fit(sm["calib"], true["calib"])
        risk = {p: cal.predict(sm[p]) for p in sm}
        signals[name] = risk
        calibrators[name] = cal
        yh = (true["holdout"] <= cfg.horizon).astype(int)
        res["health"][name] = {"holdout_brier": health.brier(risk["holdout"], yh), "holdout_ece": health.ece(risk["holdout"], yh)}
    # RUL-model-based alarm signal (higher = closer to failure)
    signals["gbm_rul"] = {p: -gbm.predict(X[p]) for p in ("calib", "holdout", "test")}

    # ---- alarm policies tuned on CALIB engines, reported on holdout + test ----
    res["alarms"] = {}
    policies = {}
    for name, sig in signals.items():
        u = {p: X[p]["unit"].to_numpy() for p in ("calib", "holdout", "test")}
        entry = {}
        for mode, far in (("min_cost", None), ("far_constrained", cfg.max_far)):
            pol, c = alarm.tune(sig["calib"], u["calib"], true["calib"], costs, horizon=cfg.horizon,
                                early_ok=cfg.early_ok, max_false_alarm_rate=far)
            ev = {}
            for p in ("holdout", "test"):
                out = alarm.engine_outcomes(u[p], true[p], pol.alarms(sig[p], u[p]), cfg.horizon, cfg.early_ok)
                ev[p] = {**alarm.summarize(out), "cost": alarm.cost(out, costs)}
            entry[mode] = {"policy": asdict(pol), "calib_cost": c, **ev}
            policies[(name, mode)] = pol
        res["alarms"][name] = entry

    # ---- SHAP (global importance on test last cycles) ----
    expl = Explainer(gbm.point_, feats)
    res["shap_global_top"] = expl.global_importance(Xt_last, k=8)
    res["train_seconds"] = round(time.time() - t0, 1)
    res["config"] = asdict(cfg)

    # ---- persist serving bundle ----
    out = Path(cfg.out_dir) / cfg.subset
    out.mkdir(parents=True, exist_ok=True)
    # model selection uses CALIBRATION engines only (holdout/test stay untouched for reporting)
    best_det = min(("isolation_forest", "autoencoder"), key=lambda n: res["alarms"][n]["min_cost"]["calib_cost"])
    best_signal = min(("gbm_rul", "isolation_forest", "autoencoder"), key=lambda n: res["alarms"][n]["min_cost"]["calib_cost"])
    bundle = {
        "config": asdict(cfg), "features": fp, "gbm": gbm,
        "detector_name": best_det, "detector": detectors[best_det], "calibrator": calibrators[best_det],
        "alarm_signal": best_signal, "policy": policies[(best_signal, "min_cost")], "rul_test": rul_map,
    }
    joblib.dump(bundle, out / "bundle.joblib", compress=3)
    torch.save(cnn.state(), out / "cnn.pt")
    res["serving_detector"] = best_det
    res["serving_alarm_signal"] = best_signal
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=float))
    if cfg.mlflow:
        _log_mlflow(cfg, res, out)
    return res


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
            flat[key] = float(v)
    return flat


def _log_mlflow(cfg: Config, res: dict, out: Path) -> None:
    import mlflow

    root = Path("mlruns").resolve()
    root.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")  # view with: mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
    name = f"turbowatch-{cfg.subset}"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(root / "artifacts").as_uri())
    mlflow.set_experiment(name)
    with mlflow.start_run(run_name=f"{cfg.subset}-seed{cfg.seed}"):
        mlflow.log_params({k: v for k, v in asdict(cfg).items() if k not in ("data_dir", "out_dir")})
        metrics_flat = {k: v for k, v in _flatten(res).items() if not k.startswith(("config.", "n_engines."))}
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics_flat.items()})
        mlflow.log_artifact(str(out / "metrics.json"))
