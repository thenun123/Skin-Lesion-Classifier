"""
Pydantic response schemas for all API endpoints.

Defining explicit schemas here:
- Generates accurate, typed OpenAPI / Swagger documentation automatically.
- Makes the API contract explicit and testable without reading implementation code.
- Prevents accidental leakage of internal fields into responses.
"""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = Field(
        ...,
        examples=["ok", "model_not_loaded"],
        description="'ok' when the model is loaded and ready to serve predictions.",
    )
    model_path: str = Field(
        ...,
        description="Absolute path to the model file that was loaded.",
    )

    model_config = {"json_schema_extra": {"example": {"status": "ok", "model_path": "/app/models/Custom_CNN.keras"}}}


class MetricsResponse(BaseModel):
    """Response schema for GET /metrics."""

    total_requests: int = Field(..., description="Total prediction requests served since startup.")
    total_errors: int = Field(..., description="Total inference errors since startup.")
    uptime_seconds: float = Field(..., description="Seconds since the API process started.")
    recent_predictions_tracked: int = Field(
        ..., description="Number of predictions in the current rolling window."
    )
    avg_inference_time_ms: float | None = Field(
        None, description="Average inference latency over the rolling window (ms)."
    )
    recent_malignant_rate: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of recent predictions classified as malignant.",
    )


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    label: str = Field(
        ...,
        examples=["benign", "malignant"],
        description="Predicted class label.",
    )
    probability_malignant: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw sigmoid output — probability that the lesion is malignant.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the predicted label (= probability of the winning class).",
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision threshold used: probability_malignant >= threshold → malignant.",
    )
    inference_time_ms: float = Field(
        ...,
        ge=0.0,
        description="End-to-end inference latency in milliseconds (preprocessing + forward pass).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "label": "malignant",
                "probability_malignant": 0.7812,
                "confidence": 0.7812,
                "threshold": 0.34,
                "inference_time_ms": 147.3,
            }
        }
    }
