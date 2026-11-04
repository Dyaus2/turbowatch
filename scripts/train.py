"""Train + evaluate. Example:
    python scripts/train.py --subsets FD001 --seeds 0 1 2 3 4
Writes artifacts/<subset>/ (serving bundle from the FIRST seed), results/runs/<subset>_seed<k>.json and logs to MLflow.
"""
import argparse
import json
import shutil
from pathlib import Path

from turbowatch import data
from turbowatch.pipeline import Config, run

p = argparse.ArgumentParser()
p.add_argument("--subsets", nargs="+", default=["FD001"])
p.add_argument("--seeds", nargs="+", type=int, default=[0])
p.add_argument("--cnn-epochs", type=int, default=20)
p.add_argument("--data-dir", default="data/raw")
p.add_argument("--no-mlflow", action="store_true")
p.add_argument("--device", default="cpu", help="cuda on a GPU runtime (CNN only)")
a = p.parse_args()

data.download(a.data_dir)
Path("results/runs").mkdir(parents=True, exist_ok=True)
for subset in a.subsets:
    for i, seed in enumerate(a.seeds):
        out_dir = "artifacts" if i == 0 else f"artifacts/_seed{seed}"
        cfg = Config(subset=subset, seed=seed, data_dir=a.data_dir, out_dir=out_dir, cnn_epochs=a.cnn_epochs, device=a.device, mlflow=not a.no_mlflow)
        res = run(cfg)
        Path(f"results/runs/{subset}_seed{seed}.json").write_text(json.dumps(res, indent=2, default=float))
        print(f"{subset} seed={seed}: GBM test RMSE {res['gbm']['test_rmse']:.2f} | CNN {res['cnn']['test_rmse']:.2f} | {res['train_seconds']}s", flush=True)
        if i > 0:
            shutil.rmtree(out_dir, ignore_errors=True)  # keep only the serving bundle of the first seed
