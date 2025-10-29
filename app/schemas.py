from pydantic import BaseModel, Field
from typing import Dict

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

class PredictionResult(BaseModel):
    isSuccessful: bool
    prediction: int
    confidence: float
    features_received: Dict[str, float]
