"""Utility to upload local model files to HuggingFace Hub."""

import os
import json
import glob
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "Arko007/trustworthy-gnn-fraud-models"

login(token=HF_TOKEN)
api = HfApi()

try:
    api.create_repo(HF_REPO, exist_ok=True)
except Exception:
    pass

for f in glob.glob("*.safetensors"):
    print(f"Uploading {f}...")
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=HF_REPO)

for f in ["model_config.json", "conformal_calibration.json"]:
    if os.path.exists(f):
        print(f"Uploading {f}...")
        api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=HF_REPO)

print("Done!")
