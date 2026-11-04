import { ChevronLeft } from "lucide-react";
import { useMemo } from "react";
import { Area, AreaChart, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { EngineResp, Meta } from "./api";
import { useLiveJSON, useReplay } from "./replay";
import { fmt, Kpi, Panel, STATUS_COLOR, StatusBadge } from "./ui";

const AX = { fontSize: 11, fill: "#94a3b8" };
const GRID = "#33415555";
const TT = { contentStyle: { background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 12, color: "#e2e8f0" }, labelStyle: { color: "#94a3b8" } };

export function EngineDetail({ unit, meta, back, children }: { unit: number; meta: Meta; back: () => void; children?: React.ReactNode }) {
  const { cycle } = useReplay();
  const { data, error } = useLiveJSON<EngineResp>(`/api/engines/${unit}?cycle=${cycle}&k=6`);

  const series = useMemo(() => (data?.timeline ?? []).map((p) => ({ ...p, band: [p.rul_lo, p.rul_hi] as [number, number], riskPct: p.risk * 100 })), [data]);
  const firstAlarm = useMemo(() => series.find((p) => p.alarm)?.cycle, [series]);

  if (error && !data) {
    return <div className="panel p-6"><button onClick={back} className="text-cyan-500">← Fleet</button><p className="mt-3 text-slate-500">Engine #{unit} has no data before cycle {cycle} — press play or move the fleet clock.</p></div>;
  }
  if (!data) return <div className="panel p-6 text-slate-500">Loading engine #{unit}…</div>;
  const last = data.timeline[data.timeline.length - 1];
  const maxAbs = Math.max(...data.explanation.factors.map((f) => Math.abs(f.shap)), 1e-6);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <button onClick={back} className="inline-flex items-center gap-1 rounded-lg border border-slate-300 px-2.5 py-1.5 text-sm hover:border-cyan-500 dark:border-slate-700"><ChevronLeft size={15} /> Fleet</button>
        <h1 className="text-xl font-semibold">Engine #{unit}</h1>
        <StatusBadge status={last.status} />
        <span className="mono text-sm text-slate-500">cycle {data.cycle} / {data.total_cycles} recorded</span>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Kpi label="Predicted RUL (GBM)" value={`${fmt(last.rul)} cyc`} sub={`P10–P90: ${fmt(last.rul_lo)}–${fmt(last.rul_hi)}`} color={STATUS_COLOR[last.status]} />
        <Kpi label="Predicted RUL (1D-CNN)" value={`${fmt(last.rul_cnn)} cyc`} sub="sequence model, 30-cycle window" />
        <Kpi label="True RUL (test labels)" value={`${fmt(last.true_rul)} cyc`} sub={`capped at ${meta.rul_cap}; shown for validation`} />
        <Kpi label="Risk of failure ≤ 30 cycles" value={`${fmt(last.risk * 100)}%`} sub={`calibrated ${meta.health_detector.replace("_", " ")}`} />
        <Kpi label="Alarm" value={firstAlarm ? `since c${firstAlarm}` : "none"} sub={`policy: ${meta.alarm_signal === "gbm_rul" ? `RUL ≤ ${fmt(-meta.policy.threshold)}` : `risk ≥ ${fmt(meta.policy.threshold, 2)}`} · ${meta.policy.k}-of-${meta.policy.n}`}
          color={firstAlarm ? STATUS_COLOR.alarm : undefined} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <Panel title="Remaining useful life — prediction vs ground truth">
          <div className="h-72">
            <ResponsiveContainer>
              <ComposedChart data={series} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="cycle" tick={AX} type="number" domain={[1, "dataMax"]} />
                <YAxis tick={AX} domain={[0, 140]} />
                <Tooltip {...TT} formatter={(v: unknown, n: unknown) => (Array.isArray(v) ? [`${fmt(v[0])}–${fmt(v[1])}`, n as string] : [fmt(Number(v), 1), n as string])} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {firstAlarm && <ReferenceArea x1={firstAlarm} x2={data.cycle} fill="#ef4444" fillOpacity={0.08} />}
                <ReferenceLine y={meta.horizon} stroke="#ef4444" strokeDasharray="4 4" label={{ value: "30-cycle danger zone", fill: "#ef4444", fontSize: 10, position: "insideTopRight" }} />
                <Area dataKey="band" name="P10–P90 band" stroke="none" fill="#22d3ee" fillOpacity={0.15} isAnimationActive={false} />
                <Line dataKey="true_rul" name="True RUL" stroke="#94a3b8" strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                <Line dataKey="rul_cnn" name="1D-CNN" stroke="#a78bfa" dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line dataKey="rul" name="Gradient boosting" stroke="#22d3ee" dot={false} strokeWidth={2.2} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Why this prediction? (SHAP, latest cycle)">
          <p className="mb-3 text-xs text-slate-500">Baseline {fmt(data.explanation.base_value, 1)} cycles. Red bars shorten predicted life, green bars extend it.</p>
          <ul className="space-y-2">
            {data.explanation.factors.map((f) => (
              <li key={f.feature}>
                <div className="flex justify-between gap-2 text-xs"><span className="truncate" title={f.feature}>{f.description}</span><span className="mono tick-num" style={{ color: f.shap < 0 ? "#ef4444" : "#10b981" }}>{f.shap > 0 ? "+" : ""}{fmt(f.shap, 1)}</span></div>
                <div className="mt-1 h-2 rounded bg-slate-200 dark:bg-slate-800">
                  <div className="h-2 rounded transition-[width] duration-500" style={{ width: `${(Math.abs(f.shap) / maxAbs) * 100}%`, backgroundColor: f.shap < 0 ? "#ef4444" : "#10b981" }} />
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <div className={`grid gap-4 ${children ? "xl:grid-cols-[1fr_380px]" : ""}`}>
        <Panel title="Health risk — P(failure within 30 cycles)">
          <div className="h-44">
            <ResponsiveContainer>
              <AreaChart data={series} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#ef4444" stopOpacity={0.5} /><stop offset="1" stopColor="#ef4444" stopOpacity={0.02} /></linearGradient></defs>
                <CartesianGrid stroke={GRID} vertical={false} />
                <XAxis dataKey="cycle" tick={AX} type="number" domain={[1, "dataMax"]} />
                <YAxis tick={AX} domain={[0, 100]} unit="%" />
                <Tooltip {...TT} formatter={(v: unknown) => [`${fmt(Number(v), 1)}%`, "risk"]} />
                <ReferenceLine y={30} stroke="#f59e0b" strokeDasharray="4 4" />
                <Area dataKey="riskPct" stroke="#ef4444" fill="url(#rg)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        {children}
      </div>

      <Panel title="Key sensors (raw values)">
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
          {meta.sensors.map((s) => {
            const vals = data.sensors[s.id] ?? [];
            const pts = vals.map((v, i) => ({ c: i + 1, v }));
            return (
              <div key={s.id} className="rounded-lg border border-slate-200 p-2 dark:border-slate-800">
                <div className="flex items-baseline justify-between gap-1"><span className="truncate text-[11px] text-slate-500" title={s.label}>{s.label}</span><span className="mono text-xs tick-num">{vals.length ? fmt(vals[vals.length - 1], 2) : "–"}</span></div>
                <div className="h-14">
                  <ResponsiveContainer>
                    <LineChart data={pts} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
                      <YAxis hide domain={["dataMin", "dataMax"]} />
                      <Line dataKey="v" stroke="#22d3ee" dot={false} strokeWidth={1.2} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
