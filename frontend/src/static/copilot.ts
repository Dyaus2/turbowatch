/** Browser port of src/turbowatch/rag.py: BM25 (rank_bm25.BM25Okapi) + bge-small dense retrieval fused with RRF,
 *  cross-encoder reranking, refusal rule, extractive template answers and an optional LLM path.
 *  Kept line-for-line close to the Python so both behave the same; evaluated by scripts/eval_copilot.ts. */
import type { EngineSnap, Factor } from "../api.ts";
import { pyFixed } from "./pyfmt.ts";

export interface Chunk { id: string; source: string; title: string; page: number; url: string; text: string }
export interface Hit { chunk: Chunk; score: number; dense: number; rank: number; rerank: number | null }
export interface Citation { n: number; id: string; source: string; title: string; page: number; url: string; snippet: string }
export interface CopilotAnswer { answer: string; citations: Citation[]; mode: "llm" | "template" | "refusal"; facts?: string; grounded?: boolean; cycle?: number }

/** Embedding + reranking models (transformers.js in the browser and in Node). */
export interface Models {
  embed(texts: string[]): Promise<Float32Array>; // row-major, L2-normalised
  rerank(query: string, docs: string[]): Promise<number[]>; // raw cross-encoder logits
}

export type Message = { role: "system" | "user"; content: string };
export type LLM = (messages: Message[]) => Promise<string | null>;

export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";
const STOP = new Set("a an the of to in on for and or is are was were be by with as at from that this it its which who what when where how why do does should can i my".split(" "));

export function tokenize(t: string): string[] {
  return (t.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((w) => !STOP.has(w));
}

/** Port of rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25). */
export class BM25 {
  private docFreqs: Map<string, number>[] = [];
  private docLen: number[] = [];
  private idf = new Map<string, number>();
  private avgdl: number;
  private k1 = 1.5;
  private b = 0.75;

  constructor(corpus: string[][], epsilon = 0.25) {
    const nd = new Map<string, number>();
    let total = 0;
    for (const doc of corpus) {
      this.docLen.push(doc.length);
      total += doc.length;
      const f = new Map<string, number>();
      for (const w of doc) f.set(w, (f.get(w) ?? 0) + 1);
      this.docFreqs.push(f);
      for (const w of f.keys()) nd.set(w, (nd.get(w) ?? 0) + 1);
    }
    this.avgdl = total / corpus.length;
    let idfSum = 0;
    const negative: string[] = [];
    for (const [w, freq] of nd) {
      const idf = Math.log(corpus.length - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(w, idf);
      idfSum += idf;
      if (idf < 0) negative.push(w);
    }
    const eps = epsilon * (idfSum / this.idf.size);
    for (const w of negative) this.idf.set(w, eps);
  }

  scores(query: string[]): Float64Array {
    const out = new Float64Array(this.docLen.length);
    for (const q of query) {
      const idf = this.idf.get(q) ?? 0;
      for (let i = 0; i < out.length; i++) {
        const tf = this.docFreqs[i].get(q) ?? 0;
        out[i] += idf * ((tf * (this.k1 + 1)) / (tf + this.k1 * (1 - this.b + (this.b * this.docLen[i]) / this.avgdl)));
      }
    }
    return out;
  }
}

const argsortDesc = (a: ArrayLike<number>): number[] => Array.from({ length: a.length }, (_, i) => i).sort((i, j) => a[j] - a[i]);

export class Retriever {
  chunks: Chunk[];
  emb: Float32Array;
  dim: number;
  bm25: BM25;
  models: Models;
  useRerank: boolean;

  constructor(chunks: Chunk[], emb: Float32Array, models: Models, useRerank = true) {
    this.chunks = chunks;
    this.emb = emb;
    this.dim = emb.length / chunks.length;
    if (!Number.isInteger(this.dim)) throw new Error("embedding matrix does not match the corpus");
    this.bm25 = new BM25(chunks.map((c) => tokenize(`${c.title} ${c.text}`)));
    this.models = models;
    this.useRerank = useRerank;
  }

  async denseScores(query: string): Promise<Float64Array> {
    const q = await this.models.embed([QUERY_PREFIX + query]);
    const out = new Float64Array(this.chunks.length);
    for (let i = 0; i < out.length; i++) {
      let s = 0;
      for (let j = 0; j < this.dim; j++) s += this.emb[i * this.dim + j] * q[j];
      out[i] = s;
    }
    return out;
  }

  async hybrid(query: string, k: number, depth = 40, rrfK = 60): Promise<Hit[]> {
    const dense = await this.denseScores(query);
    const sparse = this.bm25.scores(tokenize(query));
    const fused = new Map<number, number>();
    for (const ranking of [argsortDesc(dense).slice(0, depth), argsortDesc(sparse).slice(0, depth)]) {
      ranking.forEach((i, r) => fused.set(i, (fused.get(i) ?? 0) + 1 / (rrfK + r + 1)));
    }
    const order = [...fused.keys()].sort((a, b) => (fused.get(b) as number) - (fused.get(a) as number)).slice(0, k);
    return order.map((i, r) => ({ chunk: this.chunks[i], score: fused.get(i) as number, dense: dense[i], rank: r + 1, rerank: null }));
  }

  async search(query: string, k = 5, depth = 40, rrfK = 60, rerankTop = 20): Promise<Hit[]> {
    const hits = await this.hybrid(query, Math.max(k, this.useRerank ? rerankTop : k), depth, rrfK);
    if (!this.useRerank) return hits.slice(0, k);
    const scores = await this.models.rerank(query, hits.map((h) => h.chunk.text));
    hits.forEach((h, i) => { h.rerank = scores[i]; });
    hits.sort((a, b) => (b.rerank as number) - (a.rerank as number));
    hits.forEach((h, i) => { h.rank = i + 1; });
    return hits.slice(0, k);
  }
}

// ---------------- answering (same prompts and template as rag.py) ----------------
export const SYSTEM = "You are TurboWatch's maintenance copilot for turbofan engines. Answer ONLY from the numbered context passages. "
  + "Cite passages inline like [1] or [2]. If the context does not contain the answer, say exactly: "
  + "\"I can't answer that from the available documents.\" Keep answers under 140 words. "
  + "This is decision support, not approved maintenance data.";

const context = (hits: Hit[]) => hits.map((h, i) => `[${i + 1}] (${h.chunk.source}, p.${h.chunk.page}) ${h.chunk.text}`).join("\n\n");

export function bestSentences(query: string, hits: Hit[], n = 3): [string, number][] {
  const q = new Set(tokenize(query));
  const cands: [number, string, number][] = [];
  hits.forEach((h, idx) => {
    for (const s of h.chunk.text.split(/(?<=[.!?])\s+/)) {
      const toks = new Set(tokenize(s));
      if (toks.size < 5) continue;
      let inter = 0;
      for (const t of toks) if (q.has(t)) inter++;
      cands.push([inter / (Math.sqrt(q.size) * Math.sqrt(toks.size) + 1e-9), s.trim(), idx + 1]);
    }
  });
  cands.sort((a, b) => b[0] - a[0]);
  const out: [string, number][] = [];
  const seen = new Set<string>();
  for (const [, s, i] of cands) {
    if (!seen.has(s)) { out.push([s, i]); seen.add(s); }
    if (out.length === n) break;
  }
  return out;
}

export const citations = (hits: Hit[]): Citation[] => hits.map((h, i) => ({
  n: i + 1, id: h.chunk.id, source: h.chunk.source, title: h.chunk.title, page: h.chunk.page, url: h.chunk.url,
  snippet: Array.from(h.chunk.text).slice(0, 280).join(""),
}));

export const SENSOR_CONCEPTS: Record<string, string[]> = {
  s2: ["low pressure compressor"], s3: ["high pressure compressor", "compressor stall"], s4: ["exhaust gas temperature", "hot section inspection"],
  s7: ["compressor discharge pressure"], s8: ["fan speed rpm"], s9: ["core speed rpm", "tachometer"], s11: ["compressor discharge pressure"],
  s12: ["fuel flow", "fuel control"], s13: ["fan speed rpm"], s14: ["core speed rpm"], s15: ["bypass ratio", "turbofan"],
  s17: ["bleed air"], s20: ["turbine cooling", "hot section inspection"], s21: ["turbine cooling", "hot section inspection"],
};

export const REFUSAL = "I can't answer that from the available documents.";

export class Copilot {
  r: Retriever;
  minRerank: number;
  minDense: number;

  constructor(retriever: Retriever, minRerank = 0.0, minDense = 0.55) {
    this.r = retriever;
    this.minRerank = minRerank;
    this.minDense = minDense;
  }

  relevantEnough(hits: Hit[]): boolean {
    if (!hits.length) return false;
    if (hits[0].rerank != null) return hits[0].rerank >= this.minRerank;
    return Math.max(...hits.map((h) => h.dense)) >= this.minDense;
  }

  async answer(question: string, llm?: LLM | null, k = 4): Promise<CopilotAnswer> {
    const hits = await this.r.search(question, k);
    if (!this.relevantEnough(hits)) return { answer: REFUSAL, citations: [], mode: "refusal", grounded: false };
    const out = llm ? await llm([{ role: "system", content: SYSTEM }, { role: "user", content: `Context:\n${context(hits)}\n\nQuestion: ${question}` }]) : null;
    if (out) return { answer: out, citations: citations(hits), mode: "llm", grounded: true };
    const text = bestSentences(question, hits).map(([s, i]) => `${s} [${i}]`).join(" ");
    return { answer: text, citations: citations(hits), mode: "template", grounded: true };
  }

  static query(factors: Factor[]): { drivers: Factor[]; query: string } {
    const neg = factors.filter((f) => f.shap < 0).slice(0, 3);
    const drivers = neg.length ? neg : factors.slice(0, 3);
    const concepts: string[] = [];
    for (const f of drivers) concepts.push(...(SENSOR_CONCEPTS[f.feature.split("_")[0]] ?? []));
    const uniq = [...new Set(concepts.length ? concepts : ["performance deterioration"])];
    return { drivers, query: `turbine engine inspection troubleshooting ${uniq.join(" ")}` };
  }

  static facts(e: EngineSnap, drivers: Factor[]): string {
    return `Engine #${e.unit} at cycle ${e.cycle}: status ${e.status.toUpperCase()}, predicted RUL ${pyFixed(e.rul, 0)} cycles `
      + `(P10–P90 ${pyFixed(e.rul_lo, 0)}–${pyFixed(e.rul_hi, 0)}), risk of failure within 30 cycles ${pyFixed(e.risk * 100, 0)}%. `
      + "Main model drivers (SHAP, cycles of predicted life): " + drivers.map((f) => `${f.description} ${pyFixed(f.shap, 1, true)}`).join("; ") + ".";
  }

  async explainAlarm(engine: EngineSnap, factors: Factor[], llm?: LLM | null, k = 4): Promise<CopilotAnswer> {
    const { drivers, query } = Copilot.query(factors);
    // checklist guidance comes from maintenance documents (FAA), not from the simulation paper
    let hits = (await this.r.search(query, k + 4)).filter((h) => h.chunk.source.startsWith("FAA") || h.chunk.source.startsWith("AC-")).slice(0, k);
    if (!hits.length) hits = await this.r.search(query, k);
    const facts = Copilot.facts(engine, drivers);
    const prompt = `Engine state:\n${facts}\n\nContext:\n${context(hits)}\n\nWrite: (1) a 2-sentence explanation of why the model `
      + "flags this engine, using the drivers; (2) a short inspection checklist (3-5 bullet points) grounded in the "
      + "context, each bullet citing [n]. Do not invent part numbers or limits.";
    const out = llm ? await llm([{ role: "system", content: SYSTEM }, { role: "user", content: prompt }]) : null;
    if (out) return { answer: out, facts, citations: citations(hits), mode: "llm" };
    const checklist = bestSentences(query, hits, 4).map(([s, i]) => `- ${s} [${i}]`);
    const text = `${facts}\n\nWhy: the model's predicted remaining life is pulled down mainly by `
      + drivers.map((f) => f.description).join("; ")
      + " — sustained drifts in these signals are typical of gas-path deterioration in the simulation.\n\n"
      + "Relevant guidance from the documents:\n" + checklist.join("\n");
    return { answer: text, facts, citations: citations(hits), mode: "template" };
  }
}
