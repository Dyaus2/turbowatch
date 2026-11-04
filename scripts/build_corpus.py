"""Build the copilot's document corpus from public-domain / public-use sources (see docs/SOURCES.md).

Downloads each PDF, extracts the selected pages with pypdf, cleans + chunks the text and writes
data/corpus/chunks.jsonl (committed) and data/corpus/embeddings.npy (bge-small, committed).
Run: python scripts/build_corpus.py
"""
import io
import json
import re
import urllib.request
from pathlib import Path

from pypdf import PdfReader

UA = {"User-Agent": "Mozilla/5.0 (turbowatch corpus builder)"}
SOURCES = [
    {"id": "FAA-H-8083-32B", "title": "FAA Aviation Maintenance Technician Handbook — Powerplant (FAA-H-8083-32B)",
     "url": "https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/amt_powerplant_handbook.pdf",
     "pages": list(range(56, 81)) + [115] + list(range(414, 431)) + list(range(464, 471)),
     "license": "U.S. Government work (FAA) — public domain in the U.S.; text only, third-party images not used"},
    {"id": "AC-120-17B", "title": "FAA Advisory Circular AC 120-17B — Reliability Program Methods",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-17B.pdf", "pages": None,
     "license": "U.S. Government work (FAA) — public domain in the U.S."},
    {"id": "AC-120-16G", "title": "FAA Advisory Circular AC 120-16G — Air Carrier Maintenance Programs",
     "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC_120-16G.pdf", "pages": None,
     "license": "U.S. Government work (FAA) — public domain in the U.S."},
    {"id": "NASA-CMAPSS-2008", "title": "Saxena, Goebel, Simon, Eklund — Damage Propagation Modeling for Aircraft Engine Prognostics (NASA NTRS 20090029214)",
     "url": "https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf", "pages": None,
     "license": "NASA Technical Reports Server — copyright determination: PUBLIC_USE_PERMITTED"},
]


def clean(t: str) -> str:
    t = t.replace("­", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)  # de-hyphenate line breaks
    t = re.sub(r"[ \t]*\n[ \t]*", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def chunk(text: str, max_words: int = 170, overlap: int = 35) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], []
    for s in sents:
        w = s.split()
        if cur and len(cur) + len(w) > max_words:
            out.append(" ".join(cur))
            cur = cur[-overlap:]
        cur.extend(w)
    if len(cur) > 25:
        out.append(" ".join(cur))
    return out


def main(out_dir: str = "data/corpus") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chunks = []
    for src in SOURCES:
        req = urllib.request.Request(src["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=300) as r:
            pdf = PdfReader(io.BytesIO(r.read()))
        pages = src["pages"] or list(range(1, len(pdf.pages) + 1))
        for p in pages:
            text = clean(pdf.pages[p - 1].extract_text() or "")
            if len(text.split()) < 40:
                continue
            for i, c in enumerate(chunk(text)):
                chunks.append({"id": f"{src['id']}:p{p}:{i}", "source": src["id"], "title": src["title"], "page": p, "url": src["url"], "text": c})
        print(src["id"], len(pages), "pages")
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "sources.json").write_text(json.dumps([{k: v for k, v in s.items() if k != "pages"} | {"pages": s["pages"] and [s["pages"][0], s["pages"][-1]]} for s in SOURCES], indent=2))
    print(len(chunks), "chunks")

    import numpy as np

    from turbowatch.rag import Embedder
    emb = Embedder().encode([c["text"] for c in chunks])
    np.save(out / "embeddings.npy", emb.astype(np.float32))
    print("embeddings", emb.shape)


if __name__ == "__main__":
    main()
