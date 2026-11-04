"""Export everything the dashboard needs as static files, so the demo can run on a static host (GitHub Pages /
static Hugging Face Space) with no Python server.

The outputs are produced by the SAME code paths as the FastAPI backend (turbowatch.api.Fleet / Predictor / explain),
then the browser replays them. Also writes a parity fixture: real API responses for a sample of (cycle, engine)
requests, which `frontend/scripts/parity.ts` checks the static implementation against.

    python scripts/export_static.py            # -> frontend/static-data/
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from turbowatch import api
from turbowatch.explain import describe

OUT = Path("frontend/static-data")
FIX = Path("frontend/scripts/parity_fixture.json")
TOP_K = 6


def r(x, nd=4):
    return [round(float(v), nd) for v in x]


def main() -> None:
    f = api.fleet()
    p = f.pred
    feats = p.fp.feature_names
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "engines").mkdir(parents=True)

    meta = api.meta()
    meta["base_value"] = p.explainer.base_value
    meta["features"] = [{"feature": c, "description": describe(c)} for c in feats]
    (OUT / "meta.json").write_text(json.dumps(meta))
    (OUT / "metrics.json").write_text(json.dumps(api.metrics_view()))

    # exact TreeSHAP for every test row (same explainer as /api/engines), keep the top-k drivers per row
    sv = p.explainer.values(f.X)
    order = np.argsort(-np.abs(sv), axis=1, kind="stable")[:, :TOP_K]
    Xv = f.X[feats].to_numpy(float)

    tl = f.tl
    engines = {}
    for u in f.units:
        idx = np.flatnonzero((tl["unit"] == u).to_numpy())
        rows = tl.iloc[idx]
        assert (rows["cycle"].to_numpy() == np.arange(1, len(rows) + 1)).all(), "test cycles must start at 1 and be consecutive"
        engines[str(u)] = {
            "rul": r(rows["rul"]), "rul_lo": r(rows["rul_lo"]), "rul_hi": r(rows["rul_hi"]), "rul_cnn": r(rows["rul_cnn"]),
            "risk": r(rows["risk"]), "health": r(rows["health"]), "alarm": [int(a) for a in rows["alarm"]],
            "status": "".join({"ok": "o", "watch": "w", "alarm": "a"}[s] for s in rows["status"]),
            "true_rul": [int(t) for t in rows["true_rul"]],
            # 1-decimal values exactly as the /api/fleet snapshot rounds them (avoids double rounding in the browser)
            "snap": {c: r(rows[c], 1) for c in ("rul", "rul_lo", "rul_hi", "rul_cnn")},
            # top-1 SHAP driver per row (alarms panel): [feature index, feature value, shap]
            "top": [[int(order[i, 0]), round(float(Xv[i, order[i, 0]]), 4), round(float(sv[i, order[i, 0]]), 4)] for i in idx],
        }
        raw = f.raw[f.raw["unit"] == u]
        detail = {
            "sensors": {s: r(raw[s]) for s in api.KEY_SENSORS},
            "shap": [[v for j in order[i] for v in (int(j), round(float(Xv[i, j]), 4), round(float(sv[i, j]), 4))] for i in idx],
        }
        (OUT / "engines" / f"{u}.json").write_text(json.dumps(detail, separators=(",", ":")))
    (OUT / "fleet.json").write_text(json.dumps({"subset": f.subset, "max_cycle": f.max_cycle, "rul_cap": p.cap, "engines": engines},
                                               separators=(",", ":")))

    # copilot corpus (text + metadata); embeddings for the browser model are built by frontend/scripts/embed_corpus.ts
    chunks = [json.loads(line) for line in Path("data/corpus/chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    (OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, separators=(",", ":")))

    # parity fixture: real API responses
    fx: dict = {"fleet": {}, "alarms": {}, "engines": [], "explain_facts": []}
    for c in (1, 31, 120, 199, f.max_cycle):
        fx["fleet"][str(c)] = api.fleet_view(c)
        fx["alarms"][str(c)] = api.alarms(c)
    from turbowatch.rag import Copilot

    class _Stub:  # retriever stub: records the query, returns no passages (facts/query are retrieval-independent)
        def search(self, q, k=5):
            self.q = q
            return []

    stub = _Stub()
    cop = Copilot(stub)
    for u, c in ((1, 31), (24, 120), (34, 199), (81, None), (100, 150)):
        fx["engines"].append({"unit": u, "cycle": c, "resp": api.engine(u, c, 6)})
        rows = f.engine_rows(u, c)
        snap = next(e for e in f.snapshot(int(rows["cycle"].iloc[-1])) if e["unit"] == u)
        factors = p.explain_row(f.X.loc[[rows.index[-1]]], 6)["factors"]
        out = cop.explain_alarm(snap, factors)
        fx["explain_facts"].append({"unit": u, "cycle": c, "facts": out["facts"], "query": stub.q})
    FIX.write_text(json.dumps(fx))
    size = sum(q.stat().st_size for q in OUT.rglob("*") if q.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB), {FIX}")


if __name__ == "__main__":
    main()
