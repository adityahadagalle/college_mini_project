"""Pydantic schemas for sentiment analysis API requests and responses."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# ---------------------------------------------------------------------------
# Auth schemas (imported by app.api.v1.auth)
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    """Schema for user registration requests."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "user@example.com",
                "password": "securepassword",
                "full_name": "Jane Doe",
            }
        }
    )


class UserOut(BaseModel):
    """Schema for user data returned in API responses."""

    id: str
    email: str
    full_name: str
    is_active: bool

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "email": "user@example.com",
                "full_name": "Jane Doe",
                "is_active": True,
            }
        },
    )


class Token(BaseModel):
    """JWT token response schema."""

    access_token: str
    token_type: str = "bearer"
    user: Optional[UserOut] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
            }
        }
    )


class PredictionRequest(BaseModel):
    """Single text sentiment analysis request schema."""

    text: str = Field(..., min_length=1, max_length=5000)
    temperature: float = Field(default=1.0, gt=0.0, le=5.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The platform provided an exceptional experience!",
                "temperature": 1.0,
            }
        }
    )


class PredictionMetadata(BaseModel):
    """Model versioning and execution latency metadata."""

    model_name: str
    version: str
    inference_latency_ms: float
    device: str = "CPU"


class PredictionResponse(BaseModel):
    """Single text sentiment analysis response schema."""

    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]
    metadata: PredictionMetadata

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "The platform provided an exceptional experience!",
                "label": "POSITIVE",
                "confidence": 0.985,
                "probabilities": {
                    "POSITIVE": 0.985,
                    "NEUTRAL": 0.010,
                    "NEGATIVE": 0.005,
                },
                "metadata": {
                    "model_name": "TF-Sentiment-DistilBERT",
                    "version": "1.0.0",
                    "inference_latency_ms": 12.4,
                    "device": "CPU",
                },
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Batch text sentiment analysis request schema."""

    texts: List[str] = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(default=1.0, gt=0.0, le=5.0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": ["Loved the service!", "Dreadful customer care."],
                "temperature": 1.0,
            }
        }
    )


class PredictionStatistics(BaseModel):
    """Summary statistics for batch predictions."""

    total_samples: int
    positive_count: int
    neutral_count: int
    negative_count: int
    avg_latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch text sentiment analysis response schema."""

    predictions: List[PredictionResponse]
    statistics: PredictionStatistics


# ---------------------------------------------------------------------------
# Predict route schemas (imported by app.api.v1.predict & history)
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Schema for a single prediction request via the predict API route."""

    text: str = Field(..., min_length=1, max_length=5000)

    model_config = ConfigDict(json_schema_extra={"example": {"text": "I love this product!"}})


class ExplainabilityToken(BaseModel):
    """Token-level attribution score for explainability output."""

    token: str
    score: float


class PredictResponse(BaseModel):
    """Schema for a single prediction API response."""

    id: str
    text_content: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    latency_ms: float
    explainability: List[ExplainabilityToken] = []
    created_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Batch job schemas (imported by app.api.v1.batch)
# ---------------------------------------------------------------------------


class BatchJobResponse(BaseModel):
    """Schema for batch processing job status and metadata."""

    job_id: str
    filename: str
    total_rows: int
    processed_rows: int
    status: str
    result_file_path: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Health schemas (imported by app.api.v1.health)
# ---------------------------------------------------------------------------


class SystemHealthResponse(BaseModel):
    """Schema for system health check response."""

    status: str
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    uptime_seconds: float
    model_loaded: bool


# ---------------------------------------------------------------------------
# Model info schemas (imported by app.api.v1.model)
# ---------------------------------------------------------------------------


class ModelInfoResponse(BaseModel):
    """Schema for model metadata and evaluation metrics."""

    model_version: str
    architecture: str
    parameter_count: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "v1.0.0",
                "architecture": "TFDistilBertForSequenceClassification",
                "parameter_count": 66955779,
                "accuracy": 0.932,
                "precision": 0.928,
                "recall": 0.925,
                "f1_score": 0.9265,
                "training_date": "2026-08-01",
            }
        }
    )


# ---------------------------------------------------------------------------
# Analytics schemas – re-export from app.schemas.analytics for backward compat
# ---------------------------------------------------------------------------

from app.schemas.analytics import AnalyticsOverview as AnalyticsOverview  # noqa: E402, F401
from app.schemas.analytics import SentimentCounts as SentimentDistribution  # noqa: E402, F401
