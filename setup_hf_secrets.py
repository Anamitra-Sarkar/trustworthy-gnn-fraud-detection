"""Run this script to set HuggingFace Space secrets for the backend deployment."""
import base64
from huggingface_hub import HfApi, login

HF_TOKEN_ARKO006 = open("/home/anamitra/Downloads/API_Keys_and_Secrets/hf-token-arko006.txt").read().strip()
login(token=HF_TOKEN_ARKO006)
api = HfApi()

GROQ_KEY = open("/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt").read().strip()
HF_MODEL_TOKEN = open("/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token").read().strip()

with open("/home/anamitra/Downloads/API_Keys_and_Secrets/Foila/plant-cloud-cd461-firebase-adminsdk-fbsvc-133486961f.json") as f:
    firebase_b64 = base64.b64encode(f.read().encode()).decode()

space = "Arko006/trustworthy-gnn-fraud-api"
for key, val in [("GROQ_API_KEY", GROQ_KEY), ("HF_TOKEN", HF_MODEL_TOKEN), ("FIREBASE_KEY_JSON", firebase_b64)]:
    api.add_space_secret(space, key, val)
    print(f"Set {key}")

print("All secrets configured!")
