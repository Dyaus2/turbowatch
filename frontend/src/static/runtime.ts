/** Static-hosting runtime: answers the dashboard's /api/* requests in the browser from exported data
 *  (no Python server). Selected at build time with `vite build --mode static`. */
import type { LLM, Message } from "./copilot.ts";
import { Copilot, Retriever, type Chunk, type CopilotAnswer } from "./copilot.ts";
import { HttpError, StaticFleet } from "./fleet.ts";
import type { Progress } from "./models.ts";

const BASE = `${import.meta.env.BASE_URL}data/`;

async function load(path: string): Promise<unknown> {
  const r = await fetch(BASE + path);
  if (!r.ok) throw new HttpError(r.status, `${r.statusText} — ${path}`);
  return r.json();
}

let fleetP: Promise<StaticFleet> | null = null;
const getFleet = () => (fleetP ??= StaticFleet.create(load).catch((e) => { fleetP = null; throw e; }));

// ---- copilot models: downloaded once on first use, then cached by the browser ----
type Listener = (p: { loaded: number; total: number }) => void;
const listeners = new Set<Listener>();
const files = new Map<string, Progress>();
export function onModelProgress(l: Listener): () => void { listeners.add(l); return () => listeners.delete(l); }

let copilotP: Promise<Copilot> | null = null;
export function getCopilot(): Promise<Copilot> {
  copilotP ??= (async () => {
    const { loadModels } = await import("./models.ts");
    const [chunks, emb, models] = await Promise.all([
      load("chunks.json") as Promise<Chunk[]>,
      fetch(`${BASE}embeddings.bin`).then((r) => { if (!r.ok) throw new HttpError(r.status, "embeddings.bin"); return r.arrayBuffer(); }),
      loadModels((p) => {
        files.set(p.file, p);
        let loaded = 0, total = 0;
        for (const f of files.values()) { loaded += f.loaded; total += f.total; }
        listeners.forEach((l) => l({ loaded, total }));
      }),
    ]);
    return new Copilot(new Retriever(chunks, new Float32Array(emb), models));
  })().catch((e) => { copilotP = null; throw e; });
  return copilotP;
}

// ---- optional bring-your-own LLM key: kept in memory only (never stored, never sent anywhere but the provider) ----
export const PROVIDERS = {
  groq: { label: "Groq", base: "https://api.groq.com/openai/v1", preferred: "llama-3.3-70b-versatile" },
  xai: { label: "xAI (Grok)", base: "https://api.x.ai/v1", preferred: "" },
} as const;
export type Provider = keyof typeof PROVIDERS;
export interface LLMConfig { provider: Provider; key: string; model: string }

let llmCfg: LLMConfig | null = null;
let llmError: string | null = null;
export const getLLMConfig = () => llmCfg;
export const setLLMConfig = (c: LLMConfig | null) => { llmCfg = c && c.key && c.model ? c : null; };

export async function listModels(provider: Provider, key: string): Promise<string[]> {
  const r = await fetch(`${PROVIDERS[provider].base}/models`, { headers: { Authorization: `Bearer ${key}` } });
  if (!r.ok) throw new Error(r.status === 401 ? "key rejected (401)" : `${r.status} ${r.statusText}`);
  const j = (await r.json()) as { data?: { id: string }[] };
  return (j.data ?? []).map((m) => m.id).sort();
}

function currentLLM(): LLM | null {
  const cfg = llmCfg;
  if (!cfg) return null;
  return async (messages: Message[]) => {
    try {
      const r = await fetch(`${PROVIDERS[cfg.provider].base}/chat/completions`, {
        method: "POST", headers: { Authorization: `Bearer ${cfg.key}`, "Content-Type": "application/json" },
        body: JSON.stringify({ model: cfg.model, messages, temperature: 0.1, max_tokens: 450 }),
      });
      if (!r.ok) { llmError = `LLM call failed (${r.status}) — showing the template answer`; return null; }
      const j = (await r.json()) as { choices?: { message?: { content?: string } }[] };
      return j.choices?.[0]?.message?.content?.trim() || null;
    } catch {
      llmError = "LLM call failed (network) — showing the template answer";
      return null;
    }
  };
}

// ---- request router (same paths and parameters as the FastAPI app) ----
const num = (v: string | null) => (v == null || v === "" ? null : Number(v));

export async function staticGet(url: string): Promise<unknown> {
  const u = new URL(url, location.href);
  const p = u.pathname.replace(/^.*?\/api\//, "/api/");
  const cycle = num(u.searchParams.get("cycle"));
  if (p === "/api/health") return { status: "ok" };
  if (p === "/api/metrics") {
    const [m, browser] = await Promise.all([load("metrics.json"), load("copilot_browser.json").catch(() => null)]);
    return { ...(m as object), copilot_browser: browser };
  }
  const f = await getFleet();
  if (p === "/api/meta") return f.meta;
  if (p === "/api/fleet") return f.fleetView(cycle);
  if (p === "/api/alarms") return f.alarms(cycle);
  const m = p.match(/^\/api\/engines\/(\d+)$/);
  if (m) return f.engine(Number(m[1]), cycle, Math.min(num(u.searchParams.get("k")) ?? 6, 6));
  if (p === "/api/copilot/status") return { llm: !!llmCfg, model: llmCfg?.model ?? null, static: true };
  throw new HttpError(404, `not found: ${p}`);
}

export async function staticPost(url: string, body?: unknown): Promise<CopilotAnswer & { llm_error?: string }> {
  const u = new URL(url, location.href);
  const p = u.pathname.replace(/^.*?\/api\//, "/api/");
  const cop = await getCopilot();
  llmError = null;
  let out: CopilotAnswer;
  if (p === "/api/copilot/ask") {
    const q = String((body as { question?: string })?.question ?? "").trim();
    if (q.length < 3 || q.length > 500) throw new HttpError(422, "question must be 3–500 characters");
    out = await cop.answer(q, currentLLM());
  } else {
    const m = p.match(/^\/api\/copilot\/explain\/(\d+)$/);
    if (!m) throw new HttpError(404, `not found: ${p}`);
    const f = await getFleet();
    const { snap, factors } = await f.explainInputs(Number(m[1]), num(u.searchParams.get("cycle")));
    out = { ...(await cop.explainAlarm(snap, factors, currentLLM())), cycle: snap.cycle };
  }
  return llmError ? { ...out, llm_error: llmError } : out;
}
