"""Utility to upload local model files to HuggingFace Hub."""

import os
import json
import glob
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

if not HF_TOKEN:
    local_path = "/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token"
    if os.path.exists(local_path):
        try:
            with open(local_path) as f:
                HF_TOKEN = f.read().strip()
        except Exception:
            pass

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in environment or Kaggle User Secrets!")

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
