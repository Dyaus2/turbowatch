"""API tests on a synthetic dataset written in the real C-MAPSS text format (exercises the full load path)."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from turbowatch import api, data
from turbowatch.pipeline import Config, run


def _write(df, path):
    df[data.COLUMNS].to_csv(path, sep=" ", header=False, index=False)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("tw")
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    train = data.make_synthetic(n_units=30, seed=0)
    test, rul = data.truncate(data.make_synthetic(n_units=8, seed=1), seed=2)
    _write(train, raw / "train_FD001.txt")
    _write(test, raw / "test_FD001.txt")
    np.savetxt(raw / "RUL_FD001.txt", rul, fmt="%d")
    run(Config(subset="FD001", data_dir=str(raw), out_dir=str(root / "artifacts"), cnn_epochs=2, ae_epochs=3, mlflow=False))
    fl = api.Fleet(root=root, subset="FD001")
    from tests.test_rag import DOCS, FakeEmbedder  # tiny corpus, no model downloads
    from turbowatch import rag
    chunks = [{"id": f"{s}:p{p}:0", "source": s, "title": f"Doc {s}", "page": p, "url": "https://example.org", "text": t} for s, p, t in DOCS]
    (root / "corpus").mkdir()
    (root / "corpus" / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    np.save(root / "corpus" / "embeddings.npy", FakeEmbedder().encode([c["text"] for c in chunks]))
    cp = rag.Copilot(rag.Retriever(root / "corpus", embedder=FakeEmbedder(), rerank=False), min_dense=0.2)
    original, original_cp = api.fleet, api.copilot
    api.fleet, api.copilot = (lambda: fl), (lambda: cp)
    yield TestClient(api.app)
    api.fleet, api.copilot = original, original_cp


def test_meta_and_fleet(client):
    m = client.get("/api/meta").json()
    assert m["engines"] == 8 and m["max_cycle"] > 0
    f = client.get("/api/fleet").json()
    assert len(f["engines"]) == 8 and sum(f["counts"].values()) == 8
    early = client.get("/api/fleet", params={"cycle": 5}).json()
    assert all(e["cycle"] <= 5 for e in early["engines"])


def test_engine_detail(client):
    unit = client.get("/api/fleet").json()["engines"][0]["unit"]
    d = client.get(f"/api/engines/{unit}", params={"cycle": 12}).json()
    assert d["cycle"] <= 12 and len(d["timeline"]) == d["cycle"]
    assert len(d["explanation"]["factors"]) == 6
    assert set(d["sensors"]) == set(api.KEY_SENSORS)
    assert client.get("/api/engines/9999").status_code == 404


def test_alarms_and_metrics(client):
    a = client.get("/api/alarms").json()
    assert all(e["status"] == "alarm" and "top_driver" in e for e in a["alarms"])
    assert client.get("/api/metrics").status_code == 200


def test_replay_stream(client):
    with client.stream("GET", "/api/replay/stream", params={"start": 1, "end": 3, "speed": 50}) as r:
        body = "".join(r.iter_text())
    assert body.count("event: snapshot") == 3 and "event: end" in body


def test_copilot_endpoints(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert client.get("/api/copilot/status").json()["llm"] is False
    a = client.post("/api/copilot/ask", json={"question": "What is the purpose of an alert level?"}).json()
    assert a["mode"] == "template" and a["citations"]
    assert client.post("/api/copilot/ask", json={"question": "x"}).status_code == 422  # too short
    unit = client.get("/api/fleet").json()["engines"][0]["unit"]
    e = client.post(f"/api/copilot/explain/{unit}").json()
    assert f"Engine #{unit}" in e["answer"] and e["mode"] == "template"
