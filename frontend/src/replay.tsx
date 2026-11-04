import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

interface ReplayCtx {
  cycle: number;
  maxCycle: number;
  playing: boolean;
  speed: number;
  setCycle: (c: number) => void;
  setSpeed: (s: number) => void;
  toggle: () => void;
  restart: () => void;
}

const Ctx = createContext<ReplayCtx | null>(null);
export const SPEEDS = [1, 4, 8, 16, 32];

/** Fleet clock for the replay: every engine exposes only the cycles it had observed by `cycle`. */
export function ReplayProvider({ maxCycle, children }: { maxCycle: number; children: ReactNode }) {
  const [cycle, setCycleState] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(8);
  const timer = useRef<number | null>(null);

  const setCycle = useCallback((c: number) => setCycleState(Math.max(1, Math.min(maxCycle, Math.round(c)))), [maxCycle]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => setCycleState((c) => Math.min(maxCycle, c + 1)), 1000 / speed);
    return () => { if (timer.current) window.clearInterval(timer.current); };
  }, [playing, speed, maxCycle]);

  useEffect(() => { if (playing && cycle >= maxCycle) setPlaying(false); }, [playing, cycle, maxCycle]);

  const value = useMemo<ReplayCtx>(() => ({
    cycle, maxCycle, playing, speed, setCycle, setSpeed,
    toggle: () => { if (!playing && cycle >= maxCycle) setCycleState(1); setPlaying(!playing); },
    restart: () => { setCycleState(1); setPlaying(true); },
  }), [cycle, maxCycle, playing, speed, setCycle]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useReplay(): ReplayCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useReplay outside ReplayProvider");
  return c;
}

/** Fetch JSON whenever `url` changes; keeps the last good value while the next one loads. */
export function useLiveJSON<T>(url: string | null): { data: T | null; error: string | null } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!url) return;
    const ac = new AbortController();
    fetch(url, { signal: ac.signal })
      .then((r) => { if (!r.ok) throw new Error(`${r.status} ${r.statusText}`); return r.json(); })
      .then((j) => { setData(j as T); setError(null); })
      .catch((e: unknown) => { if ((e as Error).name !== "AbortError") setError(String(e)); });
    return () => ac.abort();
  }, [url]);
  return { data, error };
}
