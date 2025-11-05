from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# --- Updated Imports ---
from app.schemas import SensorStatsRequest, MLApiResponse
from app.model_loader import load_model_and_scaler
from app.predictor import predict as run_predict
# ---
import logging
from datetime import datetime

app = FastAPI(title="Pico ML Prediction Service", version="2.0.0") # Updated title

# CORS - allow your frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

# --- Updated to load model AND scaler ---
try:
    model, scaler = load_model_and_scaler()
    logger.info("Service startup: Model and scaler loaded successfully.")
except Exception as e:
    logger.exception("--- FATAL STARTUP ERROR ---")
    logger.exception(f"Could not load model or scaler: {e}")
    # Set to None to handle in endpoint, or just exit
    model, scaler = None, None 

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

# --- *** ENDPOINT UPDATED *** ---
# 1. Path changed from /api/stats to /api/v1/predict_features
# 2. Response model changed to MLApiResponse
@app.post("/api/v1/predict_features", response_model=MLApiResponse)
def predict_from_features(req: SensorStatsRequest):
    
    if model is None or scaler is None:
        logger.error("Prediction attempt failed: Model/Scaler not loaded.")
        raise HTTPException(
            status_code=500, 
            detail={
                "success": False, 
                "error": "Model or scaler is not loaded on the server.",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    try:
        # The predictor function now returns the correctly formatted object
        result = run_predict(model, scaler, req)
        return result
        
    except Exception as e:
        logger.exception("Prediction failed")
        # Return a response that matches the MLApiResponse schema
        raise HTTPException(
            status_code=500, 
            detail={
                "success": False, 
                "error": "Prediction failed", 
                "details": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )