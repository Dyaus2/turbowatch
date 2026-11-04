import type { ReactNode } from "react";
import type { Status } from "./api";

export const STATUS_COLOR: Record<Status, string> = { ok: "#10b981", watch: "#f59e0b", alarm: "#ef4444" };
export const STATUS_LABEL: Record<Status, string> = { ok: "Normal", watch: "Watch", alarm: "Alarm" };

export function StatusBadge({ status, small }: { status: Status; small?: boolean }) {
  const c = STATUS_COLOR[status];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-semibold uppercase tracking-wide ${small ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs"}`}
      style={{ color: c, borderColor: `${c}66`, backgroundColor: `${c}14` }}>
      <span className={`h-1.5 w-1.5 rounded-full ${status === "alarm" ? "alarm-pulse" : ""}`} style={{ backgroundColor: c }} />
      {STATUS_LABEL[status]}
    </span>
  );
}

export function Kpi({ label, value, sub, color }: { label: string; value: ReactNode; sub?: ReactNode; color?: string }) {
  return (
    <div className="panel px-4 py-3">
      <div className="panel-title">{label}</div>
      <div className="mt-1 text-2xl font-semibold tick-num" style={color ? { color } : undefined}>{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  );
}

export function Panel({ title, right, children, className = "" }: { title: string; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel p-4 ${className}`}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="panel-title">{title}</h2>
        {right}
      </div>
      {children}
    </section>
  );
}

export function RiskBar({ risk }: { risk: number }) {
  const c = risk >= 0.6 ? STATUS_COLOR.alarm : risk >= 0.3 ? STATUS_COLOR.watch : STATUS_COLOR.ok;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
      <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.max(2, risk * 100)}%`, backgroundColor: c }} />
    </div>
  );
}

export const fmt = (x: number, d = 0) => x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
