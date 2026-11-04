/** Static re-implementation of the FastAPI fleet endpoints over data exported by scripts/export_static.py.
 *  Model outputs are NOT recomputed here — they are the Python pipeline's outputs; this file only reproduces the
 *  API's slicing logic (fleet clock, snapshots, alarms). Checked against real API responses by scripts/parity.ts. */
import type { AlarmItem, EngineResp, EngineSnap, Factor, FleetResp, Meta, Status, TimelinePoint } from "../api.ts";

export type Loader = (path: string) => Promise<unknown>;

interface EngineCols {
  rul: number[]; rul_lo: number[]; rul_hi: number[]; rul_cnn: number[]; risk: number[]; health: number[];
  alarm: number[]; status: string; true_rul: number[];
  snap: { rul: number[]; rul_lo: number[]; rul_hi: number[]; rul_cnn: number[] };
  top: [number, number, number][];
}
interface FleetFile { subset: string; max_cycle: number; rul_cap: number; engines: Record<string, EngineCols> }
export interface StaticMeta extends Meta { base_value: number; features: { feature: string; description: string }[] }
interface Detail { sensors: Record<string, number[]>; shap: number[][] }

const STATUS: Record<string, Status> = { o: "ok", w: "watch", a: "alarm" };

export class HttpError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(`${status} ${message}`);
    this.status = status;
  }
}

export class StaticFleet {
  meta: StaticMeta;
  data: FleetFile;
  units: number[];
  private load: Loader;
  private firstAlarm = new Map<number, number>();
  private details = new Map<number, Promise<Detail>>();

  constructor(meta: StaticMeta, data: FleetFile, load: Loader) {
    this.meta = meta;
    this.data = data;
    this.load = load;
    this.units = Object.keys(data.engines).map(Number).sort((a, b) => a - b);
    for (const u of this.units) this.firstAlarm.set(u, this.cols(u).alarm.indexOf(1));
  }

  static async create(load: Loader): Promise<StaticFleet> {
    const [meta, data] = await Promise.all([load("meta.json"), load("fleet.json")]);
    return new StaticFleet(meta as StaticMeta, data as FleetFile, load);
  }

  private cols(u: number): EngineCols {
    const e = this.data.engines[String(u)];
    if (!e) throw new HttpError(404, `engine ${u} not found`);
    return e;
  }

  /** Row index of engine u at fleet clock `cycle` (every engine starts at cycle 1, cycles are consecutive). */
  private rowAt(u: number, cycle: number | null): number {
    const n = this.cols(u).rul.length;
    return cycle == null ? n - 1 : Math.min(cycle, n) - 1;
  }

  private factor(fi: number, value: number, shap: number): Factor {
    const f = this.meta.features[fi];
    return { feature: f.feature, description: f.description, value, shap };
  }

  snapshot(cycle: number | null): EngineSnap[] {
    const c = cycle ?? this.data.max_cycle;
    return this.units.map((u) => {
      const e = this.cols(u);
      const n = e.rul.length;
      const i = Math.min(c, n) - 1;
      const alarm = e.alarm[i] === 1;
      return {
        unit: u, cycle: i + 1, total_cycles: n, finished: i + 1 >= n,
        rul: e.snap.rul[i], rul_lo: e.snap.rul_lo[i], rul_hi: e.snap.rul_hi[i], rul_cnn: e.snap.rul_cnn[i],
        risk: e.risk[i], status: STATUS[e.status[i]],
        alarm_since: alarm ? (this.firstAlarm.get(u) as number) + 1 : null,
        true_rul: Math.min(e.true_rul[i], 125),
      };
    });
  }

  fleetView(cycle: number | null): FleetResp {
    const engines = this.snapshot(cycle);
    const counts = { ok: 0, watch: 0, alarm: 0 } as Record<Status, number>;
    for (const e of engines) counts[e.status] += 1;
    return { cycle: cycle ?? this.data.max_cycle, max_cycle: this.data.max_cycle, counts, engines };
  }

  alarms(cycle: number | null): { cycle: number; alarms: AlarmItem[] } {
    const snap = this.snapshot(cycle).filter((e) => e.status === "alarm");
    snap.sort((a, b) => a.rul - b.rul); // stable, like Python's list.sort
    const alarms = snap.map((e) => {
      const [fi, v, s] = this.cols(e.unit).top[e.cycle - 1];
      return { ...e, top_driver: this.factor(fi, v, s) };
    });
    return { cycle: cycle ?? this.data.max_cycle, alarms };
  }

  private detail(u: number): Promise<Detail> {
    let d = this.details.get(u);
    if (!d) {
      d = this.load(`engines/${u}.json`) as Promise<Detail>;
      d.catch(() => this.details.delete(u));
      this.details.set(u, d);
    }
    return d;
  }

  async factors(u: number, cycle: number | null, k = 6): Promise<Factor[]> {
    const i = this.rowAt(u, cycle);
    const row = (await this.detail(u)).shap[i];
    const out: Factor[] = [];
    for (let j = 0; j < Math.min(k, row.length / 3); j++) out.push(this.factor(row[3 * j], row[3 * j + 1], row[3 * j + 2]));
    return out;
  }

  async engine(u: number, cycle: number | null, k = 6): Promise<EngineResp> {
    const e = this.cols(u);
    const i = this.rowAt(u, cycle);
    const d = await this.detail(u);
    const timeline: TimelinePoint[] = [];
    for (let j = 0; j <= i; j++) {
      timeline.push({
        cycle: j + 1, rul: e.rul[j], rul_lo: e.rul_lo[j], rul_hi: e.rul_hi[j], rul_cnn: e.rul_cnn[j], risk: e.risk[j],
        health: e.health[j], alarm: e.alarm[j] === 1, status: STATUS[e.status[j]], true_rul: Math.min(e.true_rul[j], this.data.rul_cap),
      });
    }
    const sensors: Record<string, number[]> = {};
    for (const [s, v] of Object.entries(d.sensors)) sensors[s] = v.slice(0, i + 1);
    return {
      unit: u, cycle: i + 1, total_cycles: e.rul.length, timeline, sensors,
      explanation: { base_value: this.meta.base_value, factors: await this.factors(u, cycle, k) },
    };
  }

  /** Same engine state the API's /api/copilot/explain uses: snapshot at the engine's latest observed cycle. */
  async explainInputs(u: number, cycle: number | null): Promise<{ snap: EngineSnap; factors: Factor[] }> {
    const i = this.rowAt(u, cycle);
    const snap = this.snapshot(i + 1).find((s) => s.unit === u) as EngineSnap;
    return { snap, factors: await this.factors(u, cycle, 6) };
  }
}
