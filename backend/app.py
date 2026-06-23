import os
import threading
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning, module="google")
load_dotenv()

from firebase_auth import init_firebase
from models.loader import ModelLoader
from uncertainty.conformal import ConformalPredictor
from uncertainty.evidential import EvidentialAnalyzer
from uncertainty.mc_dropout import MCDropoutAnalyzer
from config import CORS_ORIGINS, CONFORMAL_ALPHA, MC_DROPOUT_T

model_loader = ModelLoader()
conformal_predictor = ConformalPredictor(alpha=CONFORMAL_ALPHA)
evidential_analyzer = EvidentialAnalyzer()
mc_analyzer = MCDropoutAnalyzer(num_passes=MC_DROPOUT_T)


def _warmup():
    try:
        model_loader.load_config()
        cal_data = model_loader.load_calibration()
        if cal_data:
            conformal_predictor.load_calibration(cal_data)
        print("Model loader warmed up successfully")
    except Exception as e:
        print(f"Warmup warning (non-fatal): {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_firebase()
    threading.Thread(target=_warmup, daemon=True).start()
    print("Trustworthy GNN Fraud Detection API started")
    yield
    print("Shutting down")


app = FastAPI(
    title="Trustworthy GNN Fraud Detection API",
    description="Graph Neural Network-based financial fraud detection with uncertainty quantification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.health import router as health_router
from routes.inference import router as inference_router
from routes.uncertainty import router as uncertainty_router
from routes.compliance import router as compliance_router

app.include_router(health_router)
app.include_router(inference_router)
app.include_router(uncertainty_router)
app.include_router(compliance_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
