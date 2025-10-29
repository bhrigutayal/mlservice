from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import SensorStatsRequest, PredictionResult
from app.model_loader import load_model
from app.predictor import predict as run_predict
import logging

app = FastAPI(title="Physio ML Prediction Service", version="1.0.0")

# CORS - allow your frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# Load model once (returns real model or DummyModel fallback)
model = load_model()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/stats", response_model=PredictionResult)
def predict_stats(req: SensorStatsRequest):
    try:
        result = run_predict(model, req)
        return result
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail={"isSuccessful": False, "error": "Prediction failed", "details": str(e)})
