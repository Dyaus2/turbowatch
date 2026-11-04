"""Generates notebooks/turbowatch_colab.ipynb (run: python notebooks/build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# TurboWatch — reproduce training end to end (Colab T4)
Fleet health & predictive maintenance on NASA C-MAPSS. This notebook clones the repo, downloads the public data,
trains every model (gradient boosting + quantile band, 1D-CNN on GPU, Isolation Forest / autoencoder health
models, calibrated risk, cost-tuned alarm policy), evaluates **only on held-out engines and the official test set**,
and regenerates the results table.

*Runtime → Change runtime type → T4 GPU.* Full run (FD001, 3 seeds, 60 CNN epochs): roughly 10 minutes on a T4."""),
    code("""import os, pathlib, subprocess
QUICK = os.environ.get("TW_QUICK") == "1"   # CI/CPU smoke mode: tiny run
if not pathlib.Path("src/turbowatch").exists():
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Dyaus2/turbowatch"], check=True)
    os.chdir("turbowatch")
print(os.getcwd())"""),
    code("""# Colab already ships a CUDA build of torch; install the rest (pinned)
%pip install -q -r requirements.txt -r requirements-dev.txt
%pip install -q -e . --no-deps"""),
    code("""import sys, torch
PY = sys.executable  # same interpreter as the kernel (Colab and local Jupyter)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = "0" if QUICK else "0 1 2"
EPOCHS = 2 if QUICK else 60
print("device:", DEVICE, "| seeds:", SEEDS, "| CNN epochs:", EPOCHS)"""),
    md("## Train + evaluate (logs to MLflow in `mlruns/`)"),
    code("""!MLFLOW_DISABLE_AGENT_HINT=1 {PY} scripts/train.py --subsets FD001 --seeds {SEEDS} --cnn-epochs {EPOCHS} --device {DEVICE}
!{PY} scripts/make_results.py > /dev/null
from IPython.display import Markdown
Markdown(open("results/RESULTS.md").read())"""),
    md("## Inspect one held-out test engine: RUL vs ground truth, health risk, alarms, SHAP drivers"),
    code("""import matplotlib.pyplot as plt
from turbowatch import data
from turbowatch.predictor import Predictor

pred = Predictor("artifacts/FD001")
sub = data.load("FD001")
unit = max(pred.rul_test, key=lambda u: -pred.rul_test[u] + len(sub.test[sub.test.unit == u]))  # long history, close to failure
eng = sub.test[sub.test.unit == unit]
tl = pred.timeline(eng)
true = data.add_true_rul(eng.sort_values("cycle").reset_index(drop=True), {unit: pred.rul_test[unit]}).clip(upper=125)

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].fill_between(tl.cycle, tl.rul_lo, tl.rul_hi, alpha=.2, label="P10–P90 band")
ax[0].plot(tl.cycle, tl.rul, label="GBM RUL"); ax[0].plot(tl.cycle, tl.rul_cnn, label="1D-CNN RUL")
ax[0].plot(tl.cycle, true, "k--", label="true RUL (capped)"); ax[0].legend(); ax[0].set_ylabel("cycles")
ax[1].plot(tl.cycle, tl.risk, color="crimson", label="P(failure ≤ 30 cycles)")
for c in tl.cycle[tl.alarm]: ax[1].axvline(c, color="orange", alpha=.15)
ax[1].legend(); ax[1].set_xlabel("cycle"); ax[0].set_title(f"FD001 test engine {unit}")
plt.tight_layout(); plt.show()"""),
    code("""ex = pred.explain_last(eng)
print(f"Predicted RUL {tl.rul.iloc[-1]:.1f} = base {ex['base_value']:.1f} + SHAP contributions; top drivers:")
for f in ex["factors"]:
    print(f"  {f['shap']:+7.2f}  {f['description']}  ({f['feature']} = {f['value']:.2f})")"""),
    md("""### Notes
- View experiments: `!mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` (or download `mlruns/`).
- Other subsets: `--subsets FD002 FD003 FD004` (FD002/FD004 use per-regime normalisation with 6 operating regimes).
- Data: NASA PCoE C-MAPSS (public), downloaded by `turbowatch.data.download`."""),
]
nbf.write(nb, "notebooks/turbowatch_colab.ipynb")
print("ok")
