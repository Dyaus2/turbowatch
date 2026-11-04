"""Copilot tests with a deterministic fake embedder (no model downloads in CI)."""
import json
import re
import zlib

import numpy as np
import pytest

from turbowatch import rag

DOCS = [
    ("A", 1, "Compressor stall can occur if the fuel flow increases too rapidly during acceleration, producing an over rich mixture."),
    ("A", 2, "During a hot section inspection, inspect the first stage turbine blades and nozzle guide vanes for cracks, warping, or FOD."),
    ("B", 5, "The purpose of an alert level is to identify significant deviations from a previously acceptable standard of performance."),
    ("B", 6, "Alert levels are not minimum acceptable airworthiness levels but a means of identifying increases in failure rates."),
    ("C", 1, "An exponential rate of change for flow and efficiency loss was imposed for each simulated engine data set."),
]


class FakeEmbedder:
    """Hashed bag-of-words, L2-normalised: similar wording -> high cosine."""
    dim = 256

    def encode(self, texts, query=False):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for w in rag.tokenize(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0  # stable across processes (hash() is salted)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)


@pytest.fixture()
def retriever(tmp_path):
    chunks = [{"id": f"{s}:p{p}:0", "source": s, "title": f"Doc {s}", "page": p, "url": "https://example.org", "text": t} for s, p, t in DOCS]
    (tmp_path / "chunks.jsonl").write_text("\n".join(json.dumps(c) for c in chunks))
    emb = FakeEmbedder()
    np.save(tmp_path / "embeddings.npy", emb.encode([c["text"] for c in chunks]))
    return rag.Retriever(tmp_path, embedder=emb, rerank=False)


def test_hybrid_retrieval_finds_relevant(retriever):
    hits = retriever.search("what is the purpose of an alert level", k=2)
    assert hits[0].chunk["source"] == "B" and hits[0].chunk["page"] == 5
    assert [h.rank for h in hits] == [1, 2]


def test_template_answer_cites_and_is_extractive(retriever, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = rag.Copilot(retriever, min_dense=0.2).answer("What should be inspected on turbine blades during a hot section inspection?")
    assert out["mode"] == "template" and out["citations"]
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", out["answer"])]
    assert nums and all(1 <= n <= len(out["citations"]) for n in nums)
    assert "warping" in out["answer"]


def test_refuses_out_of_scope(retriever):
    out = rag.Copilot(retriever, min_dense=0.2).answer("Who won the football world cup?")
    assert out["mode"] == "refusal" and out["citations"] == []


def test_llm_path_and_no_key(retriever, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert rag.call_llm([{"role": "user", "content": "hi"}]) is None
    monkeypatch.setattr(rag, "call_llm", lambda messages, timeout=30.0: "Alert levels flag deviations [1].")
    out = rag.Copilot(retriever, min_dense=0.2).answer("purpose of an alert level")
    assert out["mode"] == "llm" and out["answer"].endswith("[1].")


def test_explain_alarm_template(retriever, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    eng = {"unit": 7, "cycle": 150, "rul": 12.0, "rul_lo": 8.0, "rul_hi": 20.0, "risk": 0.7, "status": "alarm"}
    factors = [{"feature": "s3_mean15", "description": "HPC outlet temperature (T30), 15-cycle mean", "value": 1.2, "shap": -14.0}]
    out = rag.Copilot(retriever, min_dense=0.2).explain_alarm(eng, factors)
    assert out["mode"] == "template" and "Engine #7" in out["answer"] and "HPC outlet temperature" in out["answer"]
    assert out["citations"]
