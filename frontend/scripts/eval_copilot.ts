/** Evaluate the IN-BROWSER copilot (int8 ONNX models via transformers.js + the TS retrieval/answer code) on the same
 *  question set and with the same metric definitions as scripts/eval_copilot.py.
 *  Writes results/copilot_eval_browser.json and results/COPILOT_BROWSER.md.  Run: node scripts/eval_copilot.ts */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { BM25, Copilot, QUERY_PREFIX, Retriever, tokenize, type Chunk } from "../src/static/copilot.ts";
import { DTYPE, EMBED_ID, RERANK_ID, loadModels } from "../src/static/models.ts";
import { DATA, REPO, loadEmbeddings, loadJSON } from "./node_loader.ts";

interface QA { q: string; source: string | null; page: number | null; key: string | null }
const qa: QA[] = readFileSync(join(REPO, "data", "eval", "copilot_qa.jsonl"), "utf-8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
const ansQ = qa.filter((e) => e.source);
const oosQ = qa.filter((e) => !e.source);

const chunks = (await loadJSON("chunks.json")) as Chunk[];
const models = await loadModels();
const r = new Retriever(chunks, loadEmbeddings(), models);
const byId = new Map(chunks.map((c) => [c.id, c]));
const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
const argsortDesc = (a: ArrayLike<number>) => Array.from({ length: a.length }, (_, i) => i).sort((i, j) => a[j] - a[i]);
const relevant = (c: Chunk, e: QA) => c.source === e.source && c.page === e.page && c.text.toLowerCase().includes((e.key as string).toLowerCase());

const methods = ["dense", "bm25", "hybrid", "hybrid+rerank"] as const;
const stats: Record<string, Record<string, number[]>> = Object.fromEntries(methods.map((m) => [m, { "hit@1": [], "hit@3": [], "hit@5": [], "mrr@10": [] }]));
const bm25 = new BM25(chunks.map((c) => tokenize(`${c.title} ${c.text}`)));
for (const e of ansQ) {
  const dense = await r.denseScores(e.q);
  const rk: Record<string, Chunk[]> = {
    dense: argsortDesc(dense).slice(0, 20).map((i) => chunks[i]),
    bm25: argsortDesc(bm25.scores(tokenize(e.q))).slice(0, 20).map((i) => chunks[i]),
    hybrid: (await r.hybrid(e.q, 20, 40, 60)).map((h) => h.chunk),
  };
  const cands = rk.hybrid.slice(0, 20);
  const sc = await models.rerank(e.q, cands.map((c) => c.text));
  rk["hybrid+rerank"] = argsortDesc(sc).map((i) => cands[i]);
  for (const m of methods) {
    const rel = rk[m].slice(0, 10).map((c) => relevant(c, e));
    for (const k of [1, 3, 5]) stats[m][`hit@${k}`].push(rel.slice(0, k).some(Boolean) ? 1 : 0);
    const first = rel.indexOf(true);
    stats[m]["mrr@10"].push(first >= 0 ? 1 / (first + 1) : 0);
  }
}
const retrieval = Object.fromEntries(methods.map((m) => [m, Object.fromEntries(Object.entries(stats[m]).map(([k, v]) => [k, mean(v)]))]));

const topScore = async (q: string) => (await r.search(q, 1))[0].rerank as number;
const mdAns: number[] = [];
for (const e of ansQ) mdAns.push(await topScore(e.q));
const mdOos: number[] = [];
for (const e of oosQ) mdOos.push(await topScore(e.q));

const cop = new Copilot(r);
const answers = [];
for (const e of qa) {
  const out = await cop.answer(e.q);
  const citedFull = out.citations.map((c) => (byId.get(c.id) as Chunk).text).join(" ");
  const citedToks = new Set(tokenize(citedFull));
  const sents = out.answer.replace(/\[\d+\]/g, "").split(/(?<=[.!?])\s+/).filter((s) => tokenize(s).length >= 3);
  const support = sents.length ? sents.map((s) => mean(tokenize(s).map((t) => (citedToks.has(t) ? 1 : 0))) >= 0.8) : [false];
  const nums = [...out.answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  answers.push({
    q: e.q, in_scope: !!e.source, mode: out.mode,
    key_in_answer: !!e.key && out.answer.toLowerCase().includes(e.key.toLowerCase()),
    key_in_citations: !!e.key && out.citations.some((c) => (byId.get(c.id) as Chunk).text.toLowerCase().includes((e.key as string).toLowerCase())),
    citations_valid: nums.every((n) => n >= 1 && n <= out.citations.length) && (nums.length > 0 || out.mode === "refusal"),
    sentence_support: mean(support.map(Number)), refused: out.mode === "refusal", answer: out.answer.slice(0, 400),
  });
}
const ins = answers.filter((a) => a.in_scope);
const oos = answers.filter((a) => !a.in_scope);
const answered = ins.filter((a) => !a.refused);
const res = {
  runtime: `transformers.js (ONNX Runtime), ${EMBED_ID} + ${RERANK_ID}, dtype ${DTYPE}`, query_prefix: QUERY_PREFIX,
  n_questions: qa.length, n_in_scope: ins.length, n_out_of_scope: oos.length, answer_mode: ins[0]?.mode ?? "n/a", retrieval,
  refusal_threshold_rerank_logit: cop.minRerank, top_rerank_in_scope_min: Math.min(...mdAns), top_rerank_out_of_scope_max: Math.max(...mdOos),
  in_scope_answered: mean(ins.map((a) => Number(!a.refused))), out_of_scope_refused: mean(oos.map((a) => Number(a.refused))),
  key_fact_in_answer: mean(ins.map((a) => Number(a.key_in_answer))), key_fact_in_cited_passages: mean(ins.map((a) => Number(a.key_in_citations))),
  citations_valid: answered.length ? mean(answered.map((a) => Number(a.citations_valid))) : 0,
  sentence_support: answered.length ? mean(answered.map((a) => a.sentence_support)) : 0,
  answers,
};
writeFileSync(join(REPO, "results", "copilot_eval_browser.json"), JSON.stringify(res, null, 2));
const { answers: _a, ...summary } = res;
writeFileSync(join(DATA, "copilot_browser.json"), JSON.stringify(summary));
const pct = (x: number) => `${Math.round(100 * x)}%`;
const md = [
  "# Copilot evaluation — in-browser version (generated by `frontend/scripts/eval_copilot.ts`)", "",
  `Same ${ins.length} in-scope + ${oos.length} out-of-scope questions and metric definitions as \`scripts/eval_copilot.py\`, run on the code and models the static demo ships: ${res.runtime}. Answer mode: **${res.answer_mode}**.`, "",
  "## Retrieval (in-scope questions)", "", "| Retriever | hit@1 | hit@3 | hit@5 | MRR@10 |", "|---|---|---|---|---|",
  ...methods.map((m) => `| ${m} | ${pct(retrieval[m]["hit@1"])} | ${pct(retrieval[m]["hit@3"])} | ${pct(retrieval[m]["hit@5"])} | ${retrieval[m]["mrr@10"].toFixed(3)} |`),
  "", "## Answers", "", "| Metric | Value |", "|---|---|",
  `| In-scope questions answered (not refused) | ${pct(res.in_scope_answered)} |`,
  `| Out-of-scope questions refused | ${pct(res.out_of_scope_refused)} |`,
  `| Gold key fact present in the cited passages | ${pct(res.key_fact_in_cited_passages)} |`,
  `| Gold key fact present in the answer text | ${pct(res.key_fact_in_answer)} |`,
  `| Answers whose citation markers point to retrieved passages | ${pct(res.citations_valid)} |`,
  `| Answer sentences supported by cited passages (≥80% content-word overlap) | ${pct(res.sentence_support)} |`, "",
  `Refusal rule: top reranker logit < ${cop.minRerank}. Lowest in-scope top logit: ${res.top_rerank_in_scope_min.toFixed(2)}; highest out-of-scope: ${res.top_rerank_out_of_scope_max.toFixed(2)}.`, "",
];
writeFileSync(join(REPO, "results", "COPILOT_BROWSER.md"), md.join("\n"));
console.log(md.join("\n"));
