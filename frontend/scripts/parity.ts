/** Parity check: the static (browser) API must reproduce real FastAPI responses recorded by
 *  scripts/export_static.py (frontend/scripts/parity_fixture.json). Exits 1 on any mismatch.
 *  Run: node scripts/parity.ts */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { Copilot } from "../src/static/copilot.ts";
import { StaticFleet } from "../src/static/fleet.ts";
import { pyFixed, pyRound } from "../src/static/pyfmt.ts";
import { FRONTEND, loadJSON } from "./node_loader.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type J = any;
const fx: J = JSON.parse(readFileSync(join(FRONTEND, "scripts", "parity_fixture.json"), "utf-8"));
const fleet = await StaticFleet.create(loadJSON);
let checks = 0;
const fails: string[] = [];

/** Deep compare. Numbers: exact unless `tol` applies (timeline values are stored at 4 decimals; SHAP values of
 *  alarm drivers are full precision in the API and 4 decimals in the static export). */
function same(a: J, b: J, path: string, tol = 0): void {
  checks++;
  if (typeof a === "number" && typeof b === "number") {
    if (Math.abs(a - b) > tol) fails.push(`${path}: static=${a} api=${b}`);
    return;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) { fails.push(`${path}: array length ${a?.length} vs ${b?.length}`); return; }
    a.forEach((x, i) => same(x, b[i], `${path}[${i}]`, tol));
    return;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a).sort().join(","), kb = Object.keys(b).sort().join(",");
    if (ka !== kb) { fails.push(`${path}: keys {${ka}} vs {${kb}}`); return; }
    for (const k of Object.keys(b)) same(a[k], b[k], `${path}.${k}`, ["value", "shap", "base_value"].includes(k) || path.includes("timeline") ? 1.5e-4 : tol);
    return;
  }
  if (a !== b) fails.push(`${path}: static=${JSON.stringify(a)} api=${JSON.stringify(b)}`);
}

for (const [c, resp] of Object.entries(fx.fleet)) same(fleet.fleetView(Number(c)), resp, `fleet@${c}`);
for (const [c, resp] of Object.entries(fx.alarms)) same(fleet.alarms(Number(c)), resp, `alarms@${c}`);
for (const e of fx.engines) same(await fleet.engine(e.unit, e.cycle, 6), e.resp, `engine${e.unit}@${e.cycle}`);
for (const e of fx.explain_facts) {
  const { snap, factors } = await fleet.explainInputs(e.unit, e.cycle);
  const { drivers, query } = Copilot.query(factors);
  same(query, e.query, `explain${e.unit}@${e.cycle}.query`);
  same(Copilot.facts(snap, drivers), e.facts, `explain${e.unit}@${e.cycle}.facts`);
}
// Python formatting semantics (values checked against CPython: f"{x:.1f}", f"{x:+.1f}", round)
const py: [string, string][] = [
  [pyFixed(1.25, 1), "1.2"], [pyFixed(0.35, 1), "0.3"], [pyFixed(2.5, 0), "2"], [pyFixed(3.5, 0), "4"], [pyFixed(-2.5, 0), "-2"],
  [pyFixed(-0.04, 1, true), "-0.0"], [pyFixed(11.04, 1, true), "+11.0"], [pyFixed(125, 0), "125"], [String(pyRound(0.12345, 4)), "0.1235"],
  [pyFixed(1e-7, 2), "0.00"], [pyFixed(0.125, 2), "0.12"], [pyFixed(0.375, 2), "0.38"],
];
py.forEach(([a, b], i) => same(a, b, `pyfmt[${i}]`));

if (fails.length) {
  console.error(`PARITY FAILED: ${fails.length}/${checks} checks\n` + fails.slice(0, 25).join("\n"));
  process.exit(1);
}
console.log(`parity OK: ${checks} checks against real API responses`);
