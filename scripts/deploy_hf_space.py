"""Deploy the repo to a Hugging Face Space (Docker SDK).

    HF_TOKEN=... python scripts/deploy_hf_space.py <user>/turbowatch      # optional: GROQ_API_KEY=... to set the Space secret

Uploads only git-tracked files (no data, no secrets). The token is read from the environment and never printed.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_HEADER = """---
title: TurboWatch
emoji: 🛩️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Predictive maintenance copilot on NASA C-MAPSS turbofan data
---

# TurboWatch — Fleet Health & Predictive Maintenance Copilot

Live demo of [github.com/Dyaus2/turbowatch](https://github.com/Dyaus2/turbowatch): remaining-useful-life prediction, calibrated risk,
cost-tuned alarms, SHAP explanations and a cited RAG maintenance copilot on NASA C-MAPSS data. Decision support only.
Full documentation, results and code: see the GitHub repository.
"""


def main(space_id: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.split()
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            if f == "README.md":
                continue
            dst = Path(tmp) / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(Path(f).read_bytes())
        (Path(tmp) / "README.md").write_text(SPACE_HEADER, encoding="utf-8")
        (Path(tmp) / "REPO_README.md").write_bytes(Path("README.md").read_bytes())
        api.upload_folder(folder_path=tmp, repo_id=space_id, repo_type="space", commit_message="Deploy from GitHub main")
    if os.environ.get("GROQ_API_KEY"):
        api.add_space_secret(space_id, "GROQ_API_KEY", os.environ["GROQ_API_KEY"])
        print("GROQ_API_KEY secret set")
    print(f"Uploaded. Build logs: https://huggingface.co/spaces/{space_id}  ·  App: https://{space_id.replace('/', '-').lower()}.hf.space")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Dyaus2/turbowatch")
