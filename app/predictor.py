import numpy as np
from app.schemas import SensorStatsRequest, MLApiResponse, MLPredictionData
from datetime import datetime
import logging

logger = logging.getLogger("uvicorn.error")

# --- Definitions from the new ML service ---
# This feature order MUST match your Java MLServiceRequest and your trained model
FEATURE_ORDER = [
    # ACC
    'X_std', 'Y_std', 'Z_std',
    'X_min', 'Y_min', 'Z_min',
    'X_max', 'Y_max', 'Z_max',
    'X_mean', 'Y_mean', 'Z_mean',
    # EDA
    'EDA_std', 'EDA_min', 'EDA_max', 'EDA_mean',
    # HR (BPM)
    'HR_std', 'HR_min', 'HR_max', 'HR_mean',
    # TEMP
    'TEMP_std', 'TEMP_min', 'TEMP_max', 'TEMP_mean'
]

# Map integer predictions from the model (e.g., 0, 1)
CLASS_NAMES = {
    0: "Baseline",
    1: "Stress",
}

# Define the response structure
STRESS_LEVELS = {
    0: {"level": "Low", "description": "Relaxed baseline state", "severity": 1},
    1: {"level": "High", "description": "Experiencing stress", "severity": 4},
}
# --- End definitions ---


def predict(model, scaler, req: SensorStatsRequest) -> MLApiResponse:
    """
    Runs prediction and formats the response to match the new API spec.
    """
    
    # 1. Build feature array *in the correct order*
    try:
        features_dict = req.model_dump()
        feature_values = [features_dict[feature] for feature in FEATURE_ORDER]
        features_array = np.array(feature_values).reshape(1, -1)
    except KeyError as e:
        logger.error(f"Missing feature in request: {e}")
        return MLApiResponse(
            success=False,
            error=f"Missing feature in request: {e}",
            timestamp=datetime.utcnow().isoformat()
        )
    
    # 2. Scale features
    features_scaled = scaler.transform(features_array)
    
    # 3. Make prediction
    prediction_probs = model.predict_proba(features_scaled)[0]
    predicted_class = int(np.argmax(prediction_probs))
    confidence = float(np.max(prediction_probs))
    
    # 4. Build probabilities dictionary
    probabilities = {CLASS_NAMES[i]: float(prediction_probs[i]) 
                     for i in range(len(CLASS_NAMES))}

    # 5. Build the full response object
    
    # Check if prediction class is valid
    if predicted_class not in CLASS_NAMES:
         logger.error(f"Model predicted invalid class: {predicted_class}")
         return MLApiResponse(
            success=False,
            error=f"Model predicted invalid class: {predicted_class}",
            features=features_dict,
            timestamp=datetime.utcnow().isoformat()
        )

    pred_data = MLPredictionData(
        stress_state=CLASS_NAMES[predicted_class],
        stress_level=STRESS_LEVELS[predicted_class]["level"],
        description=STRESS_LEVELS[predicted_class]["description"],
        severity=STRESS_LEVELS[predicted_class]["severity"],
        confidence=confidence,
        probabilities=probabilities
    )
    
    response = MLApiResponse(
        success=True,
        prediction=pred_data,
        features=features_dict,
        timestamp=datetime.utcnow().isoformat()
    )
    
    logger.info(f"Prediction successful: {pred_data.stress_state} (conf: {confidence:.2f})")
    
    return response