import { ArrowUpDown, Bell, LayoutGrid, List } from "lucide-react";
import { useMemo, useState } from "react";
import type { AlarmItem, EngineSnap, FleetResp, Status } from "./api";
import { useLiveJSON, useReplay } from "./replay";
import { fmt, Kpi, Panel, RiskBar, STATUS_COLOR, StatusBadge } from "./ui";

type SortKey = "unit" | "rul" | "risk" | "cycle";
const ORDER: Record<Status, number> = { alarm: 0, watch: 1, ok: 2 };

export function FleetView({ open }: { open: (unit: number) => void }) {
  const { cycle } = useReplay();
  const { data, error } = useLiveJSON<FleetResp>(`/api/fleet?cycle=${cycle}`);
  const alarms = useLiveJSON<{ alarms: AlarmItem[] }>(`/api/alarms?cycle=${cycle}`).data;
  const [mode, setMode] = useState<"grid" | "table">("grid");
  const [filter, setFilter] = useState<Status | "all">("all");
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "rul", asc: true });

  const engines = useMemo(() => {
    const list = (data?.engines ?? []).filter((e) => filter === "all" || e.status === filter);
    return [...list].sort((a, b) => {
      if (mode === "grid" && sort.key === "rul") return ORDER[a.status] - ORDER[b.status] || a.rul - b.rul;
      const d = (a[sort.key] as number) - (b[sort.key] as number);
      return sort.asc ? d : -d;
    });
  }, [data, filter, sort, mode]);

  if (error && !data) return <div className="panel p-6 text-red-500">API error: {error}</div>;
  if (!data) return <div className="panel p-6 text-slate-500">Loading fleet…</div>;
  const live = data.engines.filter((e) => !e.finished).length;
  const meanRul = data.engines.reduce((s, e) => s + e.rul, 0) / Math.max(1, data.engines.length);

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_340px]">
      <div className="space-y-4 min-w-0">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <Kpi label="Engines streaming" value={live} sub={`${data.engines.length - live} records ended`} />
          <Kpi label="Normal" value={data.counts.ok} color={STATUS_COLOR.ok} sub="no action" />
          <Kpi label="Watch" value={data.counts.watch} color={STATUS_COLOR.watch} sub="risk ≥ 30% or RUL ≤ 60" />
          <Kpi label="Alarm" value={data.counts.alarm} color={STATUS_COLOR.alarm} sub={`mean fleet RUL ${fmt(meanRul)} cycles`} />
        </div>

        <Panel title={`Fleet — ${data.engines.length} engines (official C-MAPSS FD001 test set, never seen in training)`}
          right={
            <div className="flex items-center gap-2">
              <div className="flex rounded-lg border border-slate-300 p-0.5 text-xs dark:border-slate-700">
                {(["all", "alarm", "watch", "ok"] as const).map((f) => (
                  <button key={f} onClick={() => setFilter(f)} className={`rounded-md px-2 py-1 capitalize ${filter === f ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "text-slate-500"}`}>{f}</button>
                ))}
              </div>
              <button onClick={() => setMode(mode === "grid" ? "table" : "grid")} className="rounded-lg border border-slate-300 p-1.5 dark:border-slate-700" aria-label="Toggle grid/table">
                {mode === "grid" ? <List size={15} /> : <LayoutGrid size={15} />}
              </button>
            </div>
          }>
          {mode === "grid" ? (
            <div className="grid grid-cols-[repeat(auto-fill,minmax(132px,1fr))] gap-2">
              {engines.map((e) => <Tile key={e.unit} e={e} onClick={() => open(e.unit)} />)}
            </div>
          ) : (
            <Table engines={engines} sort={sort} setSort={setSort} open={open} />
          )}
        </Panel>
      </div>

      <Panel title="Active alarms" right={<Bell size={14} className="text-red-500" />} className="h-fit xl:sticky xl:top-20">
        {!alarms?.alarms.length && <p className="text-sm text-slate-500">No engine in alarm at this cycle.</p>}
        <ul className="space-y-2">
          {alarms?.alarms.map((a) => (
            <li key={a.unit}>
              <button onClick={() => open(a.unit)} className="w-full rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-left hover:border-red-500/70">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">Engine #{a.unit}</span>
                  <span className="mono text-sm text-red-500 tick-num">RUL {fmt(a.rul)}</span>
                </div>
                <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">since cycle {a.alarm_since} · top driver: {a.top_driver.description}</div>
              </button>
            </li>
          ))}
        </ul>
        <p className="mt-3 text-[11px] leading-snug text-slate-500">Alarm policy tuned on calibration engines with a cost matrix (missed failure = 100, needless inspection = 10).</p>
      </Panel>
    </div>
  );
}

function Tile({ e, onClick }: { e: EngineSnap; onClick: () => void }) {
  const c = STATUS_COLOR[e.status];
  return (
    <button onClick={onClick} className="group relative overflow-hidden rounded-lg border bg-white p-2.5 text-left transition hover:-translate-y-0.5 hover:shadow-lg dark:bg-slate-950/60"
      style={{ borderColor: `${c}55` }}>
      <span className="absolute inset-x-0 top-0 h-0.5" style={{ backgroundColor: c }} />
      <div className="flex items-center justify-between text-xs">
        <span className="font-semibold">#{e.unit}</span>
        <span className={`h-2 w-2 rounded-full ${e.status === "alarm" ? "alarm-pulse" : ""}`} style={{ backgroundColor: c }} />
      </div>
      <div className="mt-1 mono text-xl font-semibold tick-num" style={{ color: e.status === "ok" ? undefined : c }}>{fmt(e.rul)}</div>
      <div className="text-[10px] text-slate-500 tick-num">RUL · {fmt(e.rul_lo)}–{fmt(e.rul_hi)}</div>
      <div className="mt-2"><RiskBar risk={e.risk} /></div>
      <div className="mt-1 flex justify-between text-[10px] text-slate-500 tick-num">
        <span>risk {fmt(e.risk * 100)}%</span><span>{e.finished ? "ended" : `c${e.cycle}`}</span>
      </div>
    </button>
  );
}

function Table({ engines, sort, setSort, open }: {
  engines: EngineSnap[]; sort: { key: SortKey; asc: boolean }; setSort: (s: { key: SortKey; asc: boolean }) => void; open: (u: number) => void;
}) {
  const H = ({ k, label }: { k: SortKey; label: string }) => (
    <th className="px-2 py-2 text-left">
      <button onClick={() => setSort({ key: k, asc: sort.key === k ? !sort.asc : true })} className="inline-flex items-center gap-1 hover:text-cyan-500">
        {label}<ArrowUpDown size={11} className={sort.key === k ? "text-cyan-500" : ""} />
      </button>
    </th>
  );
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm tick-num">
        <thead className="text-[11px] uppercase tracking-wider text-slate-500">
          <tr><H k="unit" label="Engine" /><th className="px-2 py-2 text-left">Status</th><H k="cycle" label="Cycle" /><H k="rul" label="RUL (GBM)" />
            <th className="px-2 py-2 text-left">P10–P90</th><th className="px-2 py-2 text-left">RUL (CNN)</th><H k="risk" label="Risk ≤30c" /><th className="px-2 py-2 text-left">True RUL</th></tr>
        </thead>
        <tbody>
          {engines.map((e) => (
            <tr key={e.unit} onClick={() => open(e.unit)} className="cursor-pointer border-t border-slate-200 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800/40">
              <td className="px-2 py-1.5 font-semibold">#{e.unit}</td>
              <td className="px-2 py-1.5"><StatusBadge status={e.status} small /></td>
              <td className="px-2 py-1.5 mono">{e.cycle}/{e.total_cycles}</td>
              <td className="px-2 py-1.5 mono font-semibold">{fmt(e.rul)}</td>
              <td className="px-2 py-1.5 mono text-slate-500">{fmt(e.rul_lo)}–{fmt(e.rul_hi)}</td>
              <td className="px-2 py-1.5 mono">{fmt(e.rul_cnn)}</td>
              <td className="px-2 py-1.5 w-32"><div className="flex items-center gap-2"><RiskBar risk={e.risk} /><span className="mono text-xs">{fmt(e.risk * 100)}%</span></div></td>
              <td className="px-2 py-1.5 mono text-slate-500">{e.true_rul}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
