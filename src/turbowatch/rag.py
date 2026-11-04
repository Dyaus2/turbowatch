"""Maintenance copilot: hybrid retrieval (bge-small dense + BM25, RRF) over public maintenance documents,
grounded answers with citations, an LLM path (Groq, OpenAI-compatible API) and a template fallback.

Decision support only — never a substitute for approved maintenance data.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

EMBED_MODEL = os.environ.get("TURBOWATCH_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-*-en-v1.5 retrieval instruction
_STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its which who what when where how why do does should can i my".split())


def tokenize(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP]


class Embedder:
    def __init__(self, name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name, device="cpu")

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return np.asarray(self.model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False), np.float32)


RERANK_MODEL = os.environ.get("TURBOWATCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")


@dataclass
class Hit:
    chunk: dict
    score: float
    dense: float
    rank: int
    rerank: float | None = None


class Retriever:
    def __init__(self, corpus_dir: str | Path, embedder: Embedder | None = None, rerank: bool = True):
        d = Path(corpus_dir)
        self.chunks = [json.loads(line) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.emb = np.load(d / "embeddings.npy")
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in self.chunks])
        self.embedder = embedder or Embedder()
        self.reranker = None
        if rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    def search(self, query: str, k: int = 5, depth: int = 40, rrf_k: int = 60, rerank_top: int = 20) -> list[Hit]:
        """Hybrid dense+BM25 retrieval fused by reciprocal rank fusion, then cross-encoder reranking of the top candidates."""
        hits = self._hybrid(query, max(k, rerank_top if self.reranker else k), depth, rrf_k)
        if self.reranker is None:
            return hits[:k]
        scores = self.reranker.predict([(query, h.chunk["text"]) for h in hits], show_progress_bar=False)
        for h, sc in zip(hits, scores, strict=True):
            h.rerank = float(sc)
        hits.sort(key=lambda h: -h.rerank)
        for i, h in enumerate(hits, start=1):
            h.rank = i
        return hits[:k]

    def _hybrid(self, query: str, k: int, depth: int, rrf_k: int) -> list[Hit]:
        q = self.embedder.encode([query], query=True)[0]
        dense = self.emb @ q
        sparse = self.bm25.get_scores(tokenize(query))
        d_rank = np.argsort(-dense)[:depth]
        s_rank = np.argsort(-sparse)[:depth]
        fused: dict[int, float] = {}
        for ranking in (d_rank, s_rank):
            for r, i in enumerate(ranking, start=1):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + r)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.chunks[i], fused[i], float(dense[i]), r) for r, i in enumerate(order, start=1)]


# ---------------- answering ----------------
SYSTEM = ("You are TurboWatch's maintenance copilot for turbofan engines. Answer ONLY from the numbered context passages. "
          "Cite passages inline like [1] or [2]. If the context does not contain the answer, say exactly: "
          "\"I can't answer that from the available documents.\" Keep answers under 140 words. "
          "This is decision support, not approved maintenance data.")


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk['source']}, p.{h.chunk['page']}) {h.chunk['text']}" for i, h in enumerate(hits, start=1))


def call_llm(messages: list[dict], timeout: float = 30.0) -> str | None:
    """Groq OpenAI-compatible chat API. Returns None when no key is configured or the call fails."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    import httpx
    try:
        r = httpx.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), "messages": messages,
                             "temperature": 0.1, "max_tokens": 450})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:  # never leak the key or crash the API: fall back to the template answer
        return None


def _best_sentences(query: str, hits: list[Hit], n: int = 3) -> list[tuple[str, int]]:
    q = set(tokenize(query))
    cands = []
    for i, h in enumerate(hits, start=1):
        for s in re.split(r"(?<=[.!?])\s+", h.chunk["text"]):
            toks = set(tokenize(s))
            if len(toks) < 5:
                continue
            cands.append((len(q & toks) / (len(q) ** 0.5 * len(toks) ** 0.5 + 1e-9), s.strip(), i))
    cands.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, s, i in cands:
        if s not in seen:
            out.append((s, i))
            seen.add(s)
        if len(out) == n:
            break
    return out


def citations(hits: list[Hit]) -> list[dict]:
    return [{"n": i, "id": h.chunk["id"], "source": h.chunk["source"], "title": h.chunk["title"], "page": h.chunk["page"],
             "url": h.chunk["url"], "snippet": h.chunk["text"][:280]} for i, h in enumerate(hits, start=1)]


# C-MAPSS sensor -> maintenance concepts used to query the documents for an alarm explanation
SENSOR_CONCEPTS = {
    "s2": ["low pressure compressor"], "s3": ["high pressure compressor", "compressor stall"], "s4": ["exhaust gas temperature", "hot section inspection"],
    "s7": ["compressor discharge pressure"], "s8": ["fan speed rpm"], "s9": ["core speed rpm", "tachometer"], "s11": ["compressor discharge pressure"],
    "s12": ["fuel flow", "fuel control"], "s13": ["fan speed rpm"], "s14": ["core speed rpm"], "s15": ["bypass ratio", "turbofan"],
    "s17": ["bleed air"], "s20": ["turbine cooling", "hot section inspection"], "s21": ["turbine cooling", "hot section inspection"],
}


class Copilot:
    def __init__(self, retriever: Retriever, min_rerank: float = 0.0, min_dense: float = 0.55):
        self.r = retriever
        # Refusal: cross-encoder logit < 0 means the reranker itself judges the best passage < 50% likely relevant.
        # (Without a reranker, fall back to a cosine-similarity floor.)
        self.min_rerank, self.min_dense = min_rerank, min_dense

    def relevant_enough(self, hits: list[Hit]) -> bool:
        if not hits:
            return False
        if hits[0].rerank is not None:
            return hits[0].rerank >= self.min_rerank
        return max(h.dense for h in hits) >= self.min_dense

    def answer(self, question: str, k: int = 4) -> dict:
        hits = self.r.search(question, k=k)
        if not self.relevant_enough(hits):
            return {"answer": "I can't answer that from the available documents.", "citations": [], "mode": "refusal", "grounded": False}
        llm = call_llm([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"Context:\n{_context(hits)}\n\nQuestion: {question}"}])
        if llm:
            return {"answer": llm, "citations": citations(hits), "mode": "llm", "grounded": True}
        sents = _best_sentences(question, hits)
        text = " ".join(f"{s} [{i}]" for s, i in sents)
        return {"answer": text, "citations": citations(hits), "mode": "template", "grounded": True}

    def explain_alarm(self, engine: dict, factors: list[dict], k: int = 4) -> dict:
        """engine: snapshot dict (unit, cycle, rul, rul_lo, rul_hi, risk, status); factors: SHAP drivers."""
        drivers = [f for f in factors if f["shap"] < 0][:3] or factors[:3]
        concepts = []
        for f in drivers:
            concepts += SENSOR_CONCEPTS.get(f["feature"].split("_")[0], [])
        query = "turbine engine inspection troubleshooting " + " ".join(dict.fromkeys(concepts or ["performance deterioration"]))
        # checklist guidance comes from maintenance documents (FAA), not from the simulation paper
        hits = [h for h in self.r.search(query, k=k + 4) if h.chunk["source"].startswith(("FAA", "AC-"))][:k] or self.r.search(query, k=k)
        facts = (f"Engine #{engine['unit']} at cycle {engine['cycle']}: status {engine['status'].upper()}, predicted RUL "
                 f"{engine['rul']:.0f} cycles (P10–P90 {engine['rul_lo']:.0f}–{engine['rul_hi']:.0f}), risk of failure within "
                 f"30 cycles {engine['risk'] * 100:.0f}%. Main model drivers (SHAP, cycles of predicted life): "
                 + "; ".join(f"{f['description']} {f['shap']:+.1f}" for f in drivers) + ".")
        prompt = (f"Engine state:\n{facts}\n\nContext:\n{_context(hits)}\n\nWrite: (1) a 2-sentence explanation of why the model "
                  "flags this engine, using the drivers; (2) a short inspection checklist (3-5 bullet points) grounded in the "
                  "context, each bullet citing [n]. Do not invent part numbers or limits.")
        llm = call_llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        if llm:
            return {"answer": llm, "facts": facts, "citations": citations(hits), "mode": "llm"}
        checklist = [f"- {s} [{i}]" for s, i in _best_sentences(query, hits, n=4)]
        text = (f"{facts}\n\nWhy: the model's predicted remaining life is pulled down mainly by "
                + "; ".join(f["description"] for f in drivers)
                + " — sustained drifts in these signals are typical of gas-path deterioration in the simulation.\n\n"
                "Relevant guidance from the documents:\n" + "\n".join(checklist))
        return {"answer": text, "facts": facts, "citations": citations(hits), "mode": "template"}
