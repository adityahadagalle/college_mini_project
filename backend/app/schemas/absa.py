"""Pydantic v2 schemas for Aspect-Based Sentiment Analysis (ABSA)."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AspectResult(BaseModel):
    """Normalized aspect extraction and classification result DTO."""

    aspect: str = Field(..., description="Extracted aspect term text")
    aspect_category: Optional[str] = Field(
        default=None,
        description="Deterministic rule-based category normalization tag (e.g. CAMERA, BATTERY, DISPLAY). Note: Heuristic category tag for analytics grouping, not a neural classification output.",
    )
    is_implicit: bool = Field(False, description="Whether the aspect was implicit rather than text span")
    start_char: int = Field(..., description="Character start offset in the original review")
    end_char: int = Field(..., description="Character end offset in the original review")
    sentiment: str = Field(..., description="POSITIVE, NEUTRAL, or NEGATIVE")
    confidence: float = Field(..., description="Softmax confidence score (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(
        ..., description="Class probability distribution for POSITIVE, NEUTRAL, and NEGATIVE"
    )
    evidence_text: Optional[str] = Field(
        None, description="Exact substring slice from original text text[start_char:end_char]"
    )
    model_name: str = Field(default="DeBERTa-v3-ABSA", description="Model architecture identifier")
    model_version: str = Field(default="v1.0.0", description="Model version tag")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp")


class ABSARequest(BaseModel):
    """Input payload for single text ABSA analysis."""

    text: str = Field(..., min_length=1, max_length=5000, description="Review text to analyze")
    prediction_id: Optional[str] = Field(
        None, description="Optional ID of existing prediction record to associate aspects with"
    )
    save_to_db: bool = Field(default=True, description="Whether to persist extracted aspects to database")


class ABSAResponse(BaseModel):
    """Output response for single text ABSA analysis."""

    text: str = Field(..., description="Original input text")
    aspects: List[AspectResult] = Field(default_factory=list, description="List of extracted aspects")
    aspect_count: int = Field(..., description="Total number of aspects identified")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class ABSABatchRequest(BaseModel):
    """Input payload for batch ABSA analysis."""

    texts: List[str] = Field(..., min_length=1, max_length=500, description="List of texts to analyze")
    save_to_db: bool = Field(default=False, description="Whether to persist aspects in database")


class ABSABatchResponse(BaseModel):
    """Output response for batch ABSA analysis."""

    results: List[ABSAResponse] = Field(..., description="Per-review ABSA result items")
    total_samples: int = Field(..., description="Number of texts processed")
    total_aspects: int = Field(..., description="Total aspects extracted across all samples")
    latency_ms: float = Field(..., description="Total processing latency in milliseconds")


class AspectFrequencyItem(BaseModel):
    """Aspect aggregation item with counts and sentiment breakdown."""

    aspect: str
    count: int
    positive_count: int
    neutral_count: int
    negative_count: int
    net_sentiment_score: float  # (pos - neg) / total


class AspectAnalyticsResponse(BaseModel):
    """Aggregated enterprise aspect sentiment intelligence statistics."""

    total_aspects: int
    unique_aspects_count: int
    sentiment_distribution: Dict[str, int]
    sentiment_ratios: Dict[str, float]
    top_positive_aspects: List[AspectFrequencyItem]
    top_negative_aspects: List[AspectFrequencyItem]
    top_neutral_aspects: List[AspectFrequencyItem]
    recent_aspect_records: List[Dict[str, Any]]
