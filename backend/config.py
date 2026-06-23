import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Arko007/trustworthy-gnn-fraud-models")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_CACHE_DIR = os.getenv("MODEL_DIR", "/model")

FIREBASE_KEY_JSON = os.getenv("FIREBASE_KEY_JSON", "")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,https://*.vercel.app").split(",")

CONFORMAL_ALPHA = float(os.getenv("CONFORMAL_ALPHA", "0.1"))
MC_DROPOUT_T = int(os.getenv("MC_DROPOUT_T", "50"))

VACUITY_ESCALATION_THRESHOLD = 0.7
