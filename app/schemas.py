from pydantic import BaseModel, Field
from typing import Dict

# This class remains the same, it's the 24-feature input
class SensorStatsRequest(BaseModel):
    X_mean: float = Field(...)
    Y_mean: float = Field(...)
    Z_mean: float = Field(...)
    EDA_mean: float = Field(...)
    HR_mean: float = Field(...)
    TEMP_mean: float = Field(...)

    X_std: float = Field(...)
    Y_std: float = Field(...)
    Z_std: float = Field(...)
    EDA_std: float = Field(...)
    HR_std: float = Field(...)
    TEMP_std: float = Field(...)

    X_min: float = Field(...)
    Y_min: float = Field(...)
    Z_min: float = Field(...)
    EDA_min: float = Field(...)
    HR_min: float = Field(...)
    TEMP_min: float = Field(...)

    X_max: float = Field(...)
    Y_max: float = Field(...)
    Z_max: float = Field(...)
    EDA_max: float = Field(...)
    HR_max: float = Field(...)
    TEMP_max: float = Field(...)


# --- NEW RESPONSE MODELS ---
# These replace the old 'PredictionResult'

class MLPredictionData(BaseModel):
    """
    Corresponds to the nested 'prediction' object.
    Matches the Java 'MLPredictionData' DTO.
    """
    stress_state: str
    stress_level: str
    description: str
    severity: int
    confidence: float
    probabilities: Dict[str, float]

class MLApiResponse(BaseModel):
    """
    Corresponds to the top-level response.
    Matches the Java 'MLApiResponse' DTO.
    """
    success: bool
    prediction: MLPredictionData = None
    features: Dict[str, float] = None
    error: str = None
    timestamp: str

# This old class is no longer used by the main endpoint
# class PredictionResult(BaseModel):
#     isSuccessful: bool
#     prediction: int
#     confidence: float
#     features_received: Dict[str, float]