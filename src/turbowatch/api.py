"""FastAPI backend: fleet overview, engine timelines + SHAP, alarms, metrics, and a replay stream.

The "fleet" is the official C-MAPSS test set (engines the models never saw). Replay simulates a live
feed: at fleet clock `cycle=c` every engine exposes only the cycles it had observed by then.
Run: uvicorn turbowatch.api:app --port 7860
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import data
from .predictor import Predictor

ROOT = Path(os.environ.get("TURBOWATCH_ROOT", Path(__file__).resolve().parents[2]))
SUBSET = os.environ.get("TURBOWATCH_SUBSET", "FD001")
KEY_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s17", "s20", "s21"]


class Fleet:
    """Loads the serving bundle once and precomputes every test engine's full timeline (causal, so
    the row at cycle c only depends on cycles <= c — exactly what a live system would have shown)."""

    def __init__(self, root: Path = ROOT, subset: str = SUBSET):
        self.subset = subset
        self.pred = Predictor(root / "artifacts" / subset)
        sub = data.load(subset, data.download(root / "data" / "raw"))
        self.raw = sub.test.sort_values(["unit", "cycle"]).reset_index(drop=True)
        self.X = self.pred.fp.transform(self.raw)
        tl = self.pred.timelines(self.raw)
        tl["true_rul"] = data.add_true_rul(self.raw, self.pred.rul_test).to_numpy()
        self.tl = tl
        self.units = sorted(int(u) for u in self.raw["unit"].unique())
        self.lengths = self.raw.groupby("unit")["cycle"].max().astype(int).to_dict()
        self.max_cycle = int(max(self.lengths.values()))
        self.metrics_path = root / "results"

    def engine_rows(self, unit: int, cycle: int | None) -> pd.DataFrame:
        if unit not in self.lengths:
            raise HTTPException(404, f"engine {unit} not found")
        m = self.tl["unit"] == unit
        if cycle is not None:
            m &= self.tl["cycle"] <= cycle
        return self.tl[m]

    def snapshot(self, cycle: int | None) -> list[dict]:
        c = self.max_cycle if cycle is None else cycle
        rows = self.tl[self.tl["cycle"] <= c]
        last = rows.groupby("unit").tail(1)
        first_alarm = rows[rows["alarm"]].groupby("unit")["cycle"].min()
        out = []
        for r in last.itertuples(index=False):
            u = int(r.unit)
            out.append({
                "unit": u, "cycle": int(r.cycle), "total_cycles": int(self.lengths[u]), "finished": bool(r.cycle >= self.lengths[u]),
                "rul": round(float(r.rul), 1), "rul_lo": round(float(r.rul_lo), 1), "rul_hi": round(float(r.rul_hi), 1),
                "rul_cnn": round(float(r.rul_cnn), 1), "risk": round(float(r.risk), 4), "status": str(r.status),
                "alarm_since": int(first_alarm[u]) if u in first_alarm.index and bool(r.alarm) else None,
                "true_rul": int(min(r.true_rul, 125)),
            })
        return out


@lru_cache(maxsize=1)
def fleet() -> Fleet:
    return Fleet()


@lru_cache(maxsize=1)
def copilot():
    from .rag import Copilot, Retriever
    return Copilot(Retriever(ROOT / "data" / "corpus"))


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("TURBOWATCH_WARM", "1") == "1" and (ROOT / "artifacts" / SUBSET).exists():
        fleet()  # precompute the fleet at startup so the first visitor does not wait
        if (ROOT / "data" / "corpus" / "chunks.jsonl").exists():
            copilot()  # load embedding + reranker models once
    yield


app = FastAPI(title="TurboWatch API", version="0.1.0", description="Fleet health & predictive maintenance on NASA C-MAPSS",
              lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    f = fleet()
    p = f.pred
    return {
        "subset": f.subset, "engines": len(f.units), "max_cycle": f.max_cycle, "rul_cap": p.cap,
        "horizon": p.cfg["horizon"], "early_ok": p.cfg["early_ok"],
        "alarm_signal": p.alarm_signal, "health_detector": p.detector_name,
        "policy": {"threshold": p.policy.threshold, "k": p.policy.k, "n": p.policy.n},
        "sensors": [{"id": s, "label": d} for s, d in _sensor_labels(KEY_SENSORS)],
    }


def _sensor_labels(ids):
    from .explain import SENSOR_INFO
    return [(s, SENSOR_INFO[s]) for s in ids]


@app.get("/api/fleet")
def fleet_view(cycle: int | None = Query(None, ge=1)):
    f = fleet()
    snap = f.snapshot(cycle)
    counts = {s: sum(e["status"] == s for e in snap) for s in ("ok", "watch", "alarm")}
    return {"cycle": cycle or f.max_cycle, "max_cycle": f.max_cycle, "counts": counts, "engines": snap}


@app.get("/api/engines/{unit}")
def engine(unit: int, cycle: int | None = Query(None, ge=1), k: int = Query(6, ge=1, le=15)):
    f = fleet()
    rows = f.engine_rows(unit, cycle)
    if rows.empty:
        raise HTTPException(404, f"engine {unit} has no data up to cycle {cycle}")
    raw = f.raw[(f.raw["unit"] == unit) & (f.raw["cycle"] <= int(rows["cycle"].max()))]
    Xrow = f.X.loc[[rows.index[-1]]]  # features at the latest observed cycle (index aligned with tl)
    ex = f.pred.explain_row(Xrow, k)
    tl = rows.assign(true_rul=np.minimum(rows["true_rul"], f.pred.cap))
    return {
        "unit": unit, "cycle": int(rows["cycle"].iloc[-1]), "total_cycles": f.lengths[unit],
        "timeline": json.loads(tl.drop(columns=["unit"]).round(4).to_json(orient="records")),
        "sensors": {s: raw[s].round(4).tolist() for s in KEY_SENSORS},
        "explanation": ex,
    }


@app.get("/api/alarms")
def alarms(cycle: int | None = Query(None, ge=1)):
    f = fleet()
    snap = [e for e in f.snapshot(cycle) if e["status"] == "alarm"]
    snap.sort(key=lambda e: e["rul"])
    for e in snap:
        Xrow = f.X[(f.X["unit"] == e["unit"]) & (f.X["cycle"] == e["cycle"])]
        e["top_driver"] = f.pred.explain_row(Xrow, 1)["factors"][0]
    return {"cycle": cycle or f.max_cycle, "alarms": snap}


@app.get("/api/metrics")
def metrics_view():
    f = fleet()
    out: dict = {}
    p = f.metrics_path / "summary.json"
    if p.exists():
        out["summary"] = json.loads(p.read_text())
    ce = f.metrics_path / "copilot_eval.json"
    if ce.exists():
        out["copilot"] = {k: v for k, v in json.loads(ce.read_text()).items() if k != "answers"}
    dj = f.metrics_path / "drift.json"
    if dj.exists():
        out["drift"] = json.loads(dj.read_text())
    md = f.metrics_path / "RESULTS.md"
    out["results_markdown"] = md.read_text() if md.exists() else ""
    return out


@app.get("/api/copilot/status")
def copilot_status():
    return {"llm": bool(os.environ.get("GROQ_API_KEY")), "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile") if os.environ.get("GROQ_API_KEY") else None}


@app.post("/api/copilot/ask")
def copilot_ask(body: AskIn):
    return copilot().answer(body.question.strip())


@app.post("/api/copilot/explain/{unit}")
def copilot_explain(unit: int, cycle: int | None = Query(None, ge=1)):
    f = fleet()
    rows = f.engine_rows(unit, cycle)
    if rows.empty:
        raise HTTPException(404, f"engine {unit} has no data up to cycle {cycle}")
    snap = next(e for e in f.snapshot(int(rows["cycle"].iloc[-1])) if e["unit"] == unit)
    factors = f.pred.explain_row(f.X.loc[[rows.index[-1]]], 6)["factors"]
    return {**copilot().explain_alarm(snap, factors), "cycle": snap["cycle"]}


@app.get("/api/replay/stream")
async def replay_stream(start: int = Query(1, ge=1), speed: float = Query(4.0, gt=0, le=50), end: int | None = Query(None, ge=1)):
    """Server-sent events: one fleet snapshot per cycle at `speed` cycles/second."""
    f = fleet()
    stop = min(end or f.max_cycle, f.max_cycle)

    async def gen():
        for c in range(start, stop + 1):
            snap = f.snapshot(c)
            counts = {s: sum(e["status"] == s for e in snap) for s in ("ok", "watch", "alarm")}
            yield f"event: snapshot\ndata: {json.dumps({'cycle': c, 'counts': counts, 'engines': snap})}\n\n"
            await asyncio.sleep(1.0 / speed)
        yield "event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- static frontend (built React app) ----------
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = DIST / path
        if path and target.is_file() and DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
