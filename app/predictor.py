import numpy as np
from app.schemas import SensorStatsRequest, PredictionResult

def predict(model, req: SensorStatsRequest) -> PredictionResult:
    # keep same order as in your Flask version
    fields = [
        'X_mean','Y_mean','Z_mean','EDA_mean','HR_mean','TEMP_mean',
        'X_std','Y_std','Z_std','EDA_std','HR_std','TEMP_std',
        'X_min','Y_min','Z_min','EDA_min','HR_min','TEMP_min',
        'X_max','Y_max','Z_max','EDA_max','HR_max','TEMP_max'
    ]
    # build feature array
    arr = np.array([getattr(req, f) for f in fields], dtype=float).reshape(1, -1)
    # predict
    prediction_result = model.predict(arr)
    if hasattr(prediction_result, '__iter__'):
        prediction = int(prediction_result[0])
    else:
        prediction = int(prediction_result)
    # probabilities
    probabilities = model.predict_proba(arr)
    probabilities = np.array(probabilities)
    if probabilities.ndim == 2:
        confidence = float(probabilities[0, prediction])
    else:
        confidence = float(probabilities[prediction])
    return PredictionResult(
        isSuccessful=True,
        prediction=prediction,
        confidence=confidence,
        features_received={f: float(getattr(req, f)) for f in fields}
    )
