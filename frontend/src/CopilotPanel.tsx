import { Bot, FileText, KeyRound, Send, Sparkles } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { apiPost, STATIC } from "./api";
import { useReplay } from "./replay";
import { Panel } from "./ui";

interface Citation { n: number; id: string; source: string; title: string; page: number; url: string; snippet: string }
interface CopilotResp { answer: string; citations: Citation[]; mode: "llm" | "template" | "refusal"; facts?: string; cycle?: number; llm_error?: string }

const SUGGESTED = [
  "What should be inspected during a hot section inspection?",
  "What can cause a compressor stall?",
  "What is the purpose of an alert level in a reliability program?",
];

/** Renders text with [n] citation markers as superscript links to the cited source page. */
function WithCitations({ text, cites }: { text: string; cites: Citation[] }) {
  const parts: ReactNode[] = [];
  text.split(/(\[\d+\])/g).forEach((p, i) => {
    const m = p.match(/^\[(\d+)\]$/);
    const c = m ? cites.find((x) => x.n === Number(m[1])) : undefined;
    if (m && c) parts.push(<a key={i} href={`${c.url}#page=${c.page}`} target="_blank" rel="noreferrer" title={`${c.source}, p.${c.page}`} className="mx-0.5 align-super text-[10px] font-semibold text-cyan-500 hover:underline">[{c.n}]</a>);
    else parts.push(<span key={i}>{p}</span>);
  });
  return <div className="whitespace-pre-wrap text-sm leading-relaxed">{parts}</div>;
}

function Sources({ cites }: { cites: Citation[] }) {
  if (!cites.length) return null;
  return (
    <ol className="mt-3 space-y-1.5 border-t border-slate-200 pt-3 dark:border-slate-800">
      {cites.map((c) => (
        <li key={c.id} className="text-xs">
          <a href={`${c.url}#page=${c.page}`} target="_blank" rel="noreferrer" className="inline-flex items-start gap-1.5 text-slate-500 hover:text-cyan-500">
            <FileText size={12} className="mt-0.5 shrink-0" /><span><b>[{c.n}]</b> {c.title}, p.{c.page}</span>
          </a>
        </li>
      ))}
    </ol>
  );
}

function ModeTag({ mode }: { mode: CopilotResp["mode"] }) {
  const label = mode === "llm" ? "LLM · grounded" : mode === "template" ? "template · no LLM" : "refused";
  return <span className="rounded-full border border-slate-300 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-500 dark:border-slate-700">{label}</span>;
}

export function CopilotPanel({ unit }: { unit: number }) {
  const { cycle } = useReplay();
  const [tab, setTab] = useState<"explain" | "ask">("explain");
  const [explain, setExplain] = useState<CopilotResp | null>(null);
  const [explainAt, setExplainAt] = useState<number | null>(null);
  const [q, setQ] = useState("");
  const [ans, setAns] = useState<CopilotResp | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const loading = useModelProgress(busy);

  useEffect(() => { setExplain(null); setExplainAt(null); }, [unit]);

  const post = (url: string, body?: unknown) => apiPost<CopilotResp>(url, body);
  const runExplain = async () => {
    setBusy(true); setErr(null);
    try { const r = await post(`/api/copilot/explain/${unit}?cycle=${cycle}`); setExplain(r); setExplainAt(r.cycle ?? cycle); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const ask = async (question: string) => {
    if (question.trim().length < 3) return;
    setBusy(true); setErr(null); setQ(question);
    try { setAns(await post("/api/copilot/ask", { question })); } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };

  return (
    <Panel title="Maintenance copilot" right={<Bot size={15} className="text-cyan-500" />}>
      <div className="mb-3 flex rounded-lg border border-slate-300 p-0.5 text-xs dark:border-slate-700">
        {(["explain", "ask"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)} className={`flex-1 rounded-md px-2 py-1 ${tab === t ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "text-slate-500"}`}>
            {t === "explain" ? "Explain this engine" : "Ask the manuals"}
          </button>
        ))}
      </div>

      {tab === "explain" ? (
        <div>
          <button onClick={runExplain} disabled={busy} className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-cyan-500 px-3 py-2 text-sm font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-60">
            <Sparkles size={15} /> {busy ? (loading ?? "Thinking…") : explain ? `Re-explain at cycle ${cycle}` : `Explain engine #${unit} at cycle ${cycle}`}
          </button>
          {explain && (
            <div className="mt-3">
              <div className="mb-2 flex items-center justify-between text-[11px] text-slate-500"><span>engine state at cycle {explainAt}</span><ModeTag mode={explain.mode} /></div>
              <WithCitations text={explain.answer} cites={explain.citations} />
              <Sources cites={explain.citations} />
            </div>
          )}
        </div>
      ) : (
        <div>
          <form onSubmit={(e) => { e.preventDefault(); ask(q); }} className="flex gap-2">
            <input value={q} onChange={(e) => setQ(e.target.value)} maxLength={500} placeholder="e.g. What causes high EGT?" aria-label="Question for the maintenance documents"
              className="min-w-0 flex-1 rounded-lg border border-slate-300 bg-transparent px-3 py-2 text-sm outline-none focus:border-cyan-500 dark:border-slate-700" />
            <button disabled={busy} className="rounded-lg bg-cyan-500 px-3 text-slate-950 hover:bg-cyan-400 disabled:opacity-60" aria-label="Ask"><Send size={15} /></button>
          </form>
          {!ans && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {SUGGESTED.map((s) => <button key={s} onClick={() => ask(s)} className="rounded-full border border-slate-300 px-2.5 py-1 text-left text-[11px] text-slate-500 hover:border-cyan-500 hover:text-cyan-500 dark:border-slate-700">{s}</button>)}
            </div>
          )}
          {busy && <p className="mt-3 text-sm text-slate-500">{loading ?? "Searching the documents…"}</p>}
          {ans && !busy && (
            <div className="mt-3">
              <div className="mb-2 flex justify-end"><ModeTag mode={ans.mode} /></div>
              <WithCitations text={ans.answer} cites={ans.citations} />
              <Sources cites={ans.citations} />
            </div>
          )}
        </div>
      )}
      {err && <p className="mt-2 text-xs text-red-500">{err}</p>}
      {(tab === "explain" ? explain : ans)?.llm_error && <p className="mt-2 text-xs text-amber-500">{(tab === "explain" ? explain : ans)?.llm_error}</p>}
      {STATIC && <KeySettings />}
      <p className="mt-3 text-[10px] leading-snug text-slate-500">{STATIC && "Runs in your browser: retrieval models (~58 MB) download once on first use. "}Decision support only — answers are grounded in public FAA/NASA documents and are not approved maintenance data. Always follow the manufacturer's manuals.</p>
    </Panel>
  );
}

/** Static build only: progress of the one-time model download, as a short label (null when not downloading). */
function useModelProgress(active: boolean): string | null {
  const [p, setP] = useState<{ loaded: number; total: number } | null>(null);
  useEffect(() => {
    if (!STATIC || !active) return;
    let off = () => {};
    import("./static/runtime").then((m) => { off = m.onModelProgress(setP); });
    return () => off();
  }, [active]);
  if (!STATIC || !active || !p || p.loaded >= p.total) return null;
  return `Downloading retrieval models ${Math.round(p.loaded / 1e6)} / ${Math.round(p.total / 1e6)} MB (first use only)…`;
}

/** Static build only: optional bring-your-own LLM key, kept in memory for this tab and sent only to the provider. */
function KeySettings() {
  const [open, setOpen] = useState(false);
  const [provider, setProvider] = useState<"groq" | "xai">("groq");
  const [key, setKey] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [state, setState] = useState<string | null>(null);
  const [active, setActive] = useState(false);

  const connect = async () => {
    const rt = await import("./static/runtime");
    setState("Checking key…");
    try {
      const list = await rt.listModels(provider, key.trim());
      const pref = rt.PROVIDERS[provider].preferred;
      const chosen = model && list.includes(model) ? model : pref && list.includes(pref) ? pref : list[0] ?? "";
      setModels(list); setModel(chosen);
      rt.setLLMConfig({ provider, key: key.trim(), model: chosen });
      setActive(!!chosen); setState(chosen ? null : "No models available for this key");
    } catch (e) { rt.setLLMConfig(null); setActive(false); setState(String((e as Error).message ?? e)); }
  };
  const pick = async (m: string) => {
    setModel(m);
    (await import("./static/runtime")).setLLMConfig({ provider, key: key.trim(), model: m });
  };
  const clear = async () => {
    (await import("./static/runtime")).setLLMConfig(null);
    setKey(""); setModels([]); setModel(""); setActive(false); setState(null);
  };

  return (
    <div className="mt-3 rounded-lg border border-dashed border-slate-300 p-2 text-xs dark:border-slate-700">
      <button onClick={() => setOpen(!open)} className="inline-flex items-center gap-1.5 text-slate-500 hover:text-cyan-500">
        <KeyRound size={12} /> {active ? `LLM answers on · ${model}` : "Optional: use your own LLM key for written answers"}
      </button>
      {open && (
        <div className="mt-2 space-y-2">
          <div className="flex gap-2">
            <select value={provider} onChange={(e) => { setProvider(e.target.value as "groq" | "xai"); setModels([]); setActive(false); }}
              aria-label="LLM provider" className="rounded border border-slate-300 bg-transparent px-1.5 py-1 dark:border-slate-700">
              <option value="groq">Groq</option><option value="xai">xAI (Grok)</option>
            </select>
            <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="API key" autoComplete="off" spellCheck={false}
              aria-label="API key" className="min-w-0 flex-1 rounded border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700" />
            <button onClick={connect} disabled={key.trim().length < 8} className="rounded bg-cyan-500 px-2 font-semibold text-slate-950 disabled:opacity-50">Use</button>
          </div>
          {models.length > 0 && (
            <div className="flex items-center gap-2">
              <select value={model} onChange={(e) => pick(e.target.value)} aria-label="Model" className="min-w-0 flex-1 rounded border border-slate-300 bg-transparent px-1.5 py-1 dark:border-slate-700">
                {models.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
              <button onClick={clear} className="text-slate-500 underline">forget key</button>
            </div>
          )}
          {state && <p className="text-amber-500">{state}</p>}
          <p className="text-[10px] leading-snug text-slate-500">The key stays in this tab's memory (gone on reload) and is sent only to the provider's API. Without a key the copilot uses extractive template answers.</p>
        </div>
      )}
    </div>
  );
}
