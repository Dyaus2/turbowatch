import { BookOpen, Gauge, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { STATIC, type Meta } from "./api";
import { CopilotPanel } from "./CopilotPanel";
import { EngineDetail } from "./EngineDetail";
import { FleetView } from "./FleetView";
import { HowItWorks } from "./HowItWorks";
import { ReplayBar } from "./ReplayBar";
import { ReplayProvider, useLiveJSON } from "./replay";

type Route = { page: "fleet" } | { page: "engine"; unit: number } | { page: "how" };

function parse(hash: string): Route {
  const m = hash.match(/^#\/engine\/(\d+)/);
  if (m) return { page: "engine", unit: Number(m[1]) };
  if (hash.startsWith("#/how")) return { page: "how" };
  return { page: "fleet" };
}

function useRoute(): [Route, (h: string) => void] {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  useEffect(() => {
    const on = () => { setRoute(parse(location.hash)); window.scrollTo({ top: 0 }); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return [route, (h) => { location.hash = h; }];
}

function useTheme(): [boolean, () => void] {
  const [dark, setDark] = useState(() => {
    try { const s = localStorage.getItem("tw-theme"); if (s) return s === "dark"; } catch { /* ignore */ }
    return true;
  });
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("tw-theme", dark ? "dark" : "light"); } catch { /* ignore */ }
  }, [dark]);
  return [dark, () => setDark((d) => !d)];
}

export default function App() {
  const { data: meta, error } = useLiveJSON<Meta>("/api/meta");
  const [route, go] = useRoute();
  const [dark, toggleTheme] = useTheme();

  const nav = (h: string, label: string, active: boolean, icon: React.ReactNode) => (
    <a href={h} className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm ${active ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "text-slate-500 hover:text-slate-900 dark:hover:text-white"}`}>{icon}{label}</a>
  );

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/85 backdrop-blur dark:border-slate-800 dark:bg-[#070b14]/85">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-3 px-4 py-2.5">
          <a href="#/" className="flex items-center gap-2">
            <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" className="h-7 w-7" />
            <span className="font-semibold tracking-tight">TurboWatch</span>
            <span className="hidden text-xs text-slate-500 sm:inline">Fleet health & predictive maintenance</span>
          </a>
          <nav className="ml-auto flex items-center gap-1">
            {nav("#/", "Fleet", route.page !== "how", <Gauge size={15} />)}
            {nav("#/how", "How it works", route.page === "how", <BookOpen size={15} />)}
            <a href="https://github.com/Dyaus2/turbowatch" target="_blank" rel="noreferrer" className="rounded-lg px-3 py-1.5 text-sm text-slate-500 hover:text-slate-900 dark:hover:text-white">GitHub ↗</a>
            <button onClick={toggleTheme} className="rounded-lg border border-slate-300 p-1.5 dark:border-slate-700" aria-label="Toggle dark/light theme">
              {dark ? <Sun size={15} /> : <Moon size={15} />}
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-[1500px] space-y-4 px-4 py-4">
        {error && !meta && <div className="panel p-6 text-red-500">{STATIC ? "Could not load the demo data" : "Cannot reach the API"}: {error}</div>}
        {!meta && !error && <div className="panel p-6 text-slate-500">Loading models…</div>}
        {meta && (
          <ReplayProvider maxCycle={meta.max_cycle}>
            {route.page !== "how" && <div className="sticky top-[53px] z-10"><ReplayBar /></div>}
            {route.page === "fleet" && <FleetView open={(u) => go(`#/engine/${u}`)} />}
            {route.page === "engine" && <EngineDetail unit={route.unit} meta={meta} back={() => go("#/")}><CopilotPanel unit={route.unit} /></EngineDetail>}
            {route.page === "how" && <HowItWorks />}
          </ReplayProvider>
        )}
        <footer className="pt-4 pb-8 text-center text-xs text-slate-500">
          Built by <a className="underline" href="https://dyaus2.github.io" target="_blank" rel="noreferrer">Hazem Ilahi</a> · Data: NASA C-MAPSS (simulated) · Decision support only, not certified maintenance advice.
        </footer>
      </main>
    </div>
  );
}
