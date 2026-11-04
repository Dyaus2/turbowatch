import numpy as np
import pandas as pd
import pytest

from turbowatch import alarm, data, health, metrics
from turbowatch.features import FeaturePipeline, capped, last_rows
from turbowatch.models.cnn import CNNRul, make_windows
from turbowatch.split import split_engines


@pytest.fixture(scope="module")
def synth():
    return data.make_synthetic(n_units=20, seed=1)


def test_validate_rejects_gaps(synth):
    data.validate(synth)
    bad = synth.drop(index=synth.index[5])
    with pytest.raises(ValueError):
        data.validate(bad)


def test_true_rul_train_and_test(synth):
    r = data.add_true_rul(synth)
    assert (r.groupby(synth["unit"]).min() == 0).all()
    test, rul_end = data.truncate(synth, seed=2)
    m = {u: int(v) for u, v in zip(sorted(test["unit"].unique()), rul_end, strict=True)}
    rt = data.add_true_rul(test, m)
    last = test.groupby("unit")["cycle"].idxmax()
    assert np.array_equal(rt.loc[last].to_numpy(), rul_end)


def test_engine_split_is_disjoint_and_complete():
    s = split_engines(range(1, 101), seed=3)
    a, b, c = set(s.fit), set(s.calib), set(s.holdout)
    assert not (a & b or a & c or b & c) and len(a | b | c) == 100


def test_features_are_causal(synth):
    fp = FeaturePipeline().fit(synth)
    X1 = fp.transform(synth)
    mod = synth.copy()
    mod.loc[(mod.unit == 1) & (mod.cycle == 60), fp.sensors_[0]] += 50.0  # change the future
    X2 = fp.transform(mod)
    early = (X1.unit == 1) & (X1.cycle < 60)
    pd.testing.assert_frame_equal(X1[early], X2[early])


def test_feature_pipeline_fit_ignores_other_engines(synth):
    """Normalisation statistics depend only on the engines passed to fit (no holdout leakage)."""
    fit_units = synth[synth.unit <= 10]
    fp1 = FeaturePipeline().fit(fit_units)
    fp2 = FeaturePipeline().fit(fit_units.copy())
    other = synth[synth.unit > 10].copy()
    other[fp1.sensors_] += 100.0
    np.testing.assert_allclose(fp1.stats_[0][0], fp2.stats_[0][0])
    assert "rul" not in fp1.feature_names


def test_regime_normalisation_removes_condition_offsets():
    df = data.make_synthetic(n_units=12, seed=4, n_regimes=3)
    fp = FeaturePipeline(n_regimes=3).fit(df)
    z = fp.normalise(df)
    per_regime_means = z.groupby("regime")[fp.sensors_].mean().abs().to_numpy()
    assert per_regime_means.max() < 0.25


def test_constant_sensor_dropped(synth):
    assert "s1" not in FeaturePipeline().fit(synth).sensors_


def test_metrics():
    assert metrics.phm08([50], [60]) > metrics.phm08([50], [40])
    assert metrics.rmse([1, 2], [1, 2]) == 0
    assert capped([200, 10], 125).tolist() == [125, 10]


def test_cnn_windows_are_causal_and_shaped(synth):
    fp = FeaturePipeline().fit(synth)
    n = fp.normalise(synth.sort_values(["unit", "cycle"]).reset_index(drop=True))
    X, y, ends = make_windows(n, fp.sensors_, 10, np.arange(len(n), dtype=np.float32))
    assert X.shape == (len(n), 10, len(fp.sensors_))
    # the window ending at row i contains row i as its last element
    np.testing.assert_allclose(X[100, -1], n.loc[ends[100], fp.sensors_].to_numpy(np.float32))
    Xl, _, _ = make_windows(n, fp.sensors_, 10, last_only=True)
    assert len(Xl) == n["unit"].nunique()


def test_cnn_learns_something(synth):
    fp = FeaturePipeline().fit(synth)
    srt = synth.sort_values(["unit", "cycle"]).reset_index(drop=True)
    n = fp.normalise(srt)
    y = capped(data.add_true_rul(srt), 125).astype(np.float32)
    X, yw, _ = make_windows(n, fp.sensors_, 20, y)
    m = CNNRul(window=20, epochs=4, seed=0).fit(X, yw)
    assert m.history_[-1] < m.history_[0]
    m2 = CNNRul.from_state(m.state())
    np.testing.assert_allclose(m.predict(X[:50]), m2.predict(X[:50]), rtol=1e-5)


def test_alarm_policy_k_of_n():
    units = np.array([1] * 6)
    sig = np.array([0, 1, 0, 1, 1, 0])
    a = alarm.Policy(0.5, 2, 3).alarms(sig, units)
    assert a.tolist() == [False, False, False, True, True, True]


def test_engine_outcomes_definitions():
    units = np.array([1] * 5 + [2] * 5)
    rul = np.array([100, 80, 50, 20, 0, 200, 190, 180, 170, 160])
    al = np.array([0, 0, 1, 1, 1, 0, 1, 0, 0, 0], bool)
    out = alarm.engine_outcomes(units, rul, al, horizon=30, early_ok=60).set_index("unit")
    assert out.loc[1, "detected"] and out.loc[1, "lead"] == 50 and not out.loc[1, "false_alarm"]
    assert not out.loc[2, "positive"] and out.loc[2, "false_alarm"]


def test_risk_calibrator_monotone():
    rng = np.random.default_rng(0)
    rul = rng.integers(0, 200, 2000)
    score = -rul / 50 + rng.normal(0, 0.5, 2000)
    cal = health.RiskCalibrator(30).fit(score, rul)
    p = cal.predict(np.array([-4.0, 0.0]))
    assert p[0] < p[1]
    assert health.ece(cal.predict(score), (rul <= 30).astype(int)) < 0.1


def test_last_rows():
    X = pd.DataFrame({"unit": [1, 1, 2], "cycle": [1, 2, 1], "a": [0, 1, 2]})
    assert last_rows(X)["a"].tolist() == [1, 2]
