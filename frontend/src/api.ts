export type Status = "ok" | "watch" | "alarm";

export interface EngineSnap {
  unit: number;
  cycle: number;
  total_cycles: number;
  finished: boolean;
  rul: number;
  rul_lo: number;
  rul_hi: number;
  rul_cnn: number;
  risk: number;
  status: Status;
  alarm_since: number | null;
  true_rul: number;
}

export interface FleetResp {
  cycle: number;
  max_cycle: number;
  counts: Record<Status, number>;
  engines: EngineSnap[];
}

export interface Meta {
  subset: string;
  engines: number;
  max_cycle: number;
  rul_cap: number;
  horizon: number;
  early_ok: number;
  alarm_signal: string;
  health_detector: string;
  policy: { threshold: number; k: number; n: number };
  sensors: { id: string; label: string }[];
}

export interface TimelinePoint {
  cycle: number;
  rul: number;
  rul_lo: number;
  rul_hi: number;
  rul_cnn: number;
  risk: number;
  health: number;
  alarm: boolean;
  status: Status;
  true_rul: number;
}

export interface Factor {
  feature: string;
  description: string;
  value: number;
  shap: number;
}

export interface EngineResp {
  unit: number;
  cycle: number;
  total_cycles: number;
  timeline: TimelinePoint[];
  sensors: Record<string, number[]>;
  explanation: { base_value: number; factors: Factor[] };
}

export interface AlarmItem extends EngineSnap {
  top_driver: Factor;
}

export interface MetricsResp {
  summary?: Record<string, Record<string, unknown>>;
  table?: { metric: string; test: string; holdout: string }[];
  results_markdown: string;
}

export async function getJSON<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${url}`);
  return r.json() as Promise<T>;
}
