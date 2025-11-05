import os
import joblib
import numpy as np
from threading import Lock
import logging

logger = logging.getLogger("uvicorn.error")

# --- Updated File Names ---
MODEL_PATH = os.getenv("MODEL_PATH", "/app/pico_stress_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "/app/pico_feature_scaler.pkl")

# --- DummyModel remains the same ---
class DummyModel:
    def predict(self, features):
        logger.info(f"DummyModel received features shape: {np.array(features).shape}")
        # Return 0 (Baseline) or 1 (Stress)
        result = np.array([int(np.random.choice([0, 1]))])
        logger.info(f"DummyModel predict returning: {result}")
        return result

    def predict_proba(self, features):
        logger.info(f"DummyModel proba received features shape: {np.array(features).shape}")
        # Return probabilities for 2 classes
        result = np.random.dirichlet([1, 1], 1)
        logger.info(f"DummyModel predict_proba returning: {result}")
        return result

# --- NEW DummyScaler ---
class DummyScaler:
    def transform(self, features):
        logger.info("DummyScaler is passing features through without scaling.")
        return features

_load_lock = Lock()
_cache = {} # Cache for model and scaler

def load_model_and_scaler():
    """
    Loads both the model and the scaler.
    Returns (model, scaler) tuple.
    """
    global _cache
    with _load_lock:
        if "model" in _cache and "scaler" in _cache:
            return _cache["model"], _cache["scaler"]

        model, scaler = None, None

        # Load Model
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                logger.info(f"Loaded model from {MODEL_PATH}: {type(model)}")
            except Exception as e:
                logger.exception(f"Failed to load model file: {e}")
        
        if model is None:
            logger.warning(f"Model path not found or failed to load: {MODEL_PATH}. Using DummyModel.")
            model = DummyModel()

        # Load Scaler
        if os.path.exists(SCALER_PATH):
            try:
                scaler = joblib.load(SCALER_PATH)
                logger.info(f"Loaded scaler from {SCALER_PATH}: {type(scaler)}")
            except Exception as e:
                logger.exception(f"Failed to load scaler file: {e}")
        
        if scaler is None:
            logger.warning(f"Scaler path not found or failed to load: {SCALER_PATH}. Using DummyScaler.")
            scaler = DummyScaler()
        
        _cache["model"] = model
        _cache["scaler"] = scaler
        
        return model, scaler