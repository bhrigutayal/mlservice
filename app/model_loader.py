import os
import joblib
import numpy as np
import pickle
from threading import Lock
import logging

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model.pkl")  # default path inside Docker image

class DummyModel:
    def predict(self, features):
        logger.info(f"DummyModel received features shape: {np.array(features).shape}")
        result = np.array([int(np.random.choice([0,1,2]))])
        logger.info(f"DummyModel predict returning: {result}")
        return result

    def predict_proba(self, features):
        logger.info(f"DummyModel proba received features shape: {np.array(features).shape}")
        result = np.random.dirichlet([1,1,1], 1)
        logger.info(f"DummyModel predict_proba returning: {result}")
        return result

_load_lock = Lock()

def load_model():
    # thread-safe single load
    with _load_lock:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                logger.info(f"Loaded model from {MODEL_PATH}: {type(model)}")
                # quick sanity check (optional)
                test_features = np.random.rand(1, 24)
                try:
                    logger.info(f"Test predict: {model.predict(test_features)}")
                except Exception:
                    logger.warning("Model doesn't support predict(test) in check.")
                return model
            except Exception as e:
                logger.exception(f"Failed to load model file: {e}")
                return DummyModel()
        else:
            logger.warning(f"Model path not found: {MODEL_PATH}. Using DummyModel.")
            return DummyModel()
