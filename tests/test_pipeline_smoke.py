"""End-to-end smoke test on synthetic data: train → artifacts → serving predictor."""
import numpy as np

from turbowatch import data
from turbowatch.pipeline import Config, run
from turbowatch.predictor import Predictor


def test_pipeline_and_predictor(tmp_path):
    train = data.make_synthetic(n_units=30, seed=0)
    test, rul = data.truncate(data.make_synthetic(n_units=10, seed=1), seed=2)
    cfg = Config(subset="SYNTH", out_dir=str(tmp_path), cnn_epochs=2, ae_epochs=3, mlflow=False)
    res = run(cfg, train, test, rul)
    assert res["n_engines"]["test"] == 10
    assert np.isfinite(res["gbm"]["test_rmse"]) and res["gbm"]["test_rmse"] < 80
    assert set(res["alarms"]) == {"isolation_forest", "autoencoder", "gbm_rul"}

    pred = Predictor(tmp_path / "SYNTH")
    eng = test[test.unit == test.unit.iloc[0]]
    tl = pred.timeline(eng)
    assert len(tl) == len(eng)
    assert ((tl.risk >= 0) & (tl.risk <= 1)).all()
    assert (tl.rul_lo <= tl.rul_hi).all()
    assert set(tl.status) <= {"ok", "watch", "alarm"}
    # serving reproduces training-time features: prediction on a prefix equals the prefix of the full timeline
    tl_prefix = pred.timeline(eng.iloc[:20])
    np.testing.assert_allclose(tl_prefix.rul.to_numpy(), tl.rul.to_numpy()[:20], rtol=1e-6)
    # batched fleet inference == per-engine inference
    many = pred.timelines(test)
    one = many[many.unit == eng.unit.iloc[0]].drop(columns="unit").reset_index(drop=True)
    np.testing.assert_allclose(one.rul.to_numpy(), tl.rul.to_numpy(), rtol=1e-6)
    np.testing.assert_allclose(one.risk.to_numpy(), tl.risk.to_numpy(), rtol=1e-6)
    assert (one.status.to_numpy() == tl.status.to_numpy()).all()
    ex = pred.explain_last(eng)
    assert len(ex["factors"]) == 5
    total = ex["base_value"] + sum(f["shap"] for f in pred.explainer.top_factors(pred.fp.transform(eng).iloc[[-1]], k=999))
    assert abs(total - tl.rul.iloc[-1]) < 1e-3 or tl.rul.iloc[-1] == 0  # SHAP additivity (unclipped)
