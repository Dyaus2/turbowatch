# Model card — TurboWatch (FD001 serving bundle)

**Intended use:** portfolio demonstration of condition-based maintenance analytics on simulated turbofan data — remaining-useful-life (RUL) estimation, health-risk scoring, alarm triage and explanation. **Not** for operational, airworthiness or safety decisions.

## Models
| Component | Model | Inputs | Output |
|---|---|---|---|
| RUL (served) | scikit-learn HistGradientBoostingRegressor (300 iters) + P10/P90 quantile models | 126 causal features (15 sensors × rolling mean/std 5 & 15, 15-cycle slope, EWM) + cycle | RUL (cycles, capped at 125) + band |
| RUL (comparison) | 1D-CNN (3 conv layers, PyTorch), 30-cycle window | per-regime normalised sensors | RUL |
| Health risk | Isolation Forest on healthy cycles → causal EWM → Platt calibration | 15-cycle sensor means | P(failure within 30 cycles) |
| Alarm | k-of-n policy on predicted RUL, tuned on calibration engines with costs (missed failure 100, false alarm 10) | predicted RUL | alarm on/off |
| Explanations | exact TreeSHAP on the RUL model | features | per-feature contribution (cycles) |

## Training & evaluation data
NASA C-MAPSS FD001 (100 train engines, run-to-failure; 100 test engines, truncated). Engine-grouped split of the train engines: fit 70 / calibration 15 / holdout 15. See [`DATA_CARD.md`](DATA_CARD.md).

## Performance (held-out only; FD001, mean ± std over 5 seeds — `results/RESULTS.md`)
| Metric | Official test set | Holdout engines |
|---|---|---|
| RUL RMSE — gradient boosting (cycles) | 16.26 ± 0.66 | 15.94 ± 1.50 |
| RUL RMSE — 1D-CNN (cycles) | 15.46 ± 0.61 | 16.62 ± 0.79 |
| PHM08 score — gradient boosting | 455 ± 70 | – |
| Alarm detection rate | 97 ± 5% | 100 ± 0% |
| False-alarm rate (engines) | 2 ± 2% | 8 ± 5% |
| Median warning before failure (cycles) | 36 ± 5 | 37 ± 5 |
| P10–P90 interval coverage (nominal 80%) | 66 ± 3% | – |

Copilot retrieval (26 self-written questions): hybrid + rerank hit@1 100%, out-of-scope refusal 100% (4 questions) — `results/COPILOT.md`.

## Known limitations & failure modes
- Trained on **simulated** degradation of one fault mode in one operating condition (FD001). On FD003 (an extra fan-degradation mode) the FD001 model's test RMSE rises to 20.7; a challenger retrained on FD001+FD003 reaches 15.0 (`results/DRIFT.md`).
- The P10–P90 band is over-confident (66 ± 3% coverage vs 80% nominal).
- The autoencoder health score was evaluated and rejected (poor separation); Isolation Forest risk reacts late on truncated engines.
- No sensor-fault handling, no maintenance-action resets, no censoring model.

## Ethical / safety considerations
Outputs are decision support for demonstration. Real maintenance decisions must follow the manufacturer's approved data and the operator's approved maintenance programme. The copilot only quotes public FAA/NASA documents and refuses out-of-scope questions; it can still mis-retrieve.

## Reproduce
`python scripts/train.py --subsets FD001 --seeds 0 1 2 3 4 && python scripts/make_results.py` (≈50 s per seed on 4 CPU cores) or the Colab notebook.
