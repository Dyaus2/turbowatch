# Data card

## NASA C-MAPSS turbofan degradation simulation
- **What:** multivariate time series from NASA's Commercial Modular Aero-Propulsion System Simulation. Each row = one operating cycle of one engine: unit id, cycle, 3 operating settings, 21 sensor channels (temperatures, pressures, speeds, flows — descriptions in `src/turbowatch/explain.py`).
- **Subsets:** FD001 (1 operating condition, HPC degradation; 100 train / 100 test engines), FD002 (6 conditions, HPC; 260/259), FD003 (1 condition, HPC + fan; 100/100), FD004 (6 conditions, HPC + fan; 249/248).
- **Labels:** train trajectories run to failure (RUL = cycles left); test trajectories are truncated, true RUL at the last cycle given in `RUL_FD00x.txt`.
- **Source & license:** NASA Prognostics Center of Excellence, public research data; cite Saxena et al., PHM 2008 (doi:10.1109/PHM.2008.4711414). Downloaded at run/build time, not redistributed here.
- **Processing:** schema + continuity validation; per-operating-regime z-scoring (KMeans on operating settings, fitted on fit engines only); constant sensors dropped (FD001 keeps 15/21); causal rolling features. RUL target capped at 125 cycles.
- **Known issues:** simulated, noise-only sensors (no sensor faults or missing data), failure defined by a health-index threshold, no maintenance actions. Small held-out sets (15 engines per seed).

## Copilot document corpus
556 text chunks from public FAA/NASA documents (FAA-H-8083-32B turbine pages, AC 120-17B, AC 120-16G, NASA NTRS 20090029214). Sources, page ranges and license determinations: [`SOURCES.md`](SOURCES.md). Built by `scripts/build_corpus.py` (text only; no images).

## Evaluation set
`data/eval/copilot_qa.jsonl` — 26 questions written from the corpus (each with gold source, page and key phrase verified to exist) + 4 out-of-scope questions. Self-written and small: indicative only.
