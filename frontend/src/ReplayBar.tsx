import { FastForward, Pause, Play, RotateCcw } from "lucide-react";
import { SPEEDS, useReplay } from "./replay";

export function ReplayBar() {
  const { cycle, maxCycle, playing, speed, setCycle, setSpeed, toggle, restart } = useReplay();
  return (
    <div className="panel flex flex-wrap items-center gap-3 px-3 py-2">
      <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
        <span className={`h-2 w-2 rounded-full ${playing ? "bg-emerald-500 alarm-pulse" : "bg-slate-500"}`} />
        Replay
      </span>
      <button onClick={toggle} className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500 text-slate-950 hover:bg-cyan-400" aria-label={playing ? "Pause replay" : "Play replay"}>
        {playing ? <Pause size={16} /> : <Play size={16} />}
      </button>
      <button onClick={restart} className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-300 hover:border-cyan-500 dark:border-slate-700" aria-label="Restart replay">
        <RotateCcw size={15} />
      </button>
      <div className="flex items-center gap-1" role="group" aria-label="Replay speed">
        <FastForward size={14} className="text-slate-500" />
        {SPEEDS.map((s) => (
          <button key={s} onClick={() => setSpeed(s)}
            className={`rounded-md px-2 py-1 text-xs mono ${s === speed ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "text-slate-500 hover:text-slate-900 dark:hover:text-white"}`}>
            {s}×
          </button>
        ))}
      </div>
      <input type="range" min={1} max={maxCycle} value={cycle} onChange={(e) => setCycle(Number(e.target.value))}
        className="min-w-[140px] flex-1 accent-cyan-500" aria-label="Fleet clock (cycle)" />
      <span className="mono text-sm tick-num">cycle <b>{String(cycle).padStart(3, " ")}</b><span className="text-slate-500"> / {maxCycle}</span></span>
    </div>
  );
}
