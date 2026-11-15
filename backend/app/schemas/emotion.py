"""Pydantic v2 schemas for Emotion Detection and CX Taxonomy Rollup."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DetectedCXCategory(BaseModel):
    """CX Category item exceeding calibrated thresholds."""

    category: str = Field(..., description="CX Category Name (e.g. Frustration / Anger, Delight / Excitement)")
    score: float = Field(..., description="Max sigmoid score across constituent emotions in this category")
    threshold: float = Field(..., description="Calibrated decision threshold for this category")


class DetectedEmotionItem(BaseModel):
    """Raw GoEmotions discrete label exceeding calibrated threshold."""

    label: str = Field(..., description="GoEmotions label name (e.g. anger, joy, gratitude)")
    probability: float = Field(..., description="Sigmoid probability (0.0 to 1.0)")
    threshold: float = Field(..., description="Calibrated decision threshold applied for this label")
    cx_category: str = Field(..., description="Mapped 6-category Customer Experience taxonomy label")


class EmotionRequest(BaseModel):
    """Input payload for single text emotion detection."""

    text: str = Field(..., min_length=1, max_length=5000, description="Customer review text to analyze")
    prediction_id: Optional[str] = Field(
        None, description="Optional ID of existing prediction record to associate emotion with"
    )
    save_to_db: bool = Field(default=True, description="Whether to persist emotion record in database")


class EmotionResponse(BaseModel):
    """Output contract for single text emotion detection."""

    text: str = Field(..., description="Original input text")
    primary_emotion: str = Field(
        ...,
        description="Top active CX category, or 'Neutral / Informative' / 'None' if no active emotions exceed thresholds",
    )
    confidence: float = Field(
        ...,
        description="Confidence score defined as the maximum sigmoid probability of active constituent emotions for the primary category",
    )
    detected_cx_categories: List[DetectedCXCategory] = Field(
        default_factory=list, description="Active CX categories exceeding calibrated thresholds"
    )
    detected_emotions: List[DetectedEmotionItem] = Field(
        default_factory=list, description="Active raw GoEmotions labels exceeding per-class thresholds"
    )
    raw_scores: Dict[str, float] = Field(
        ..., description="Complete probability distribution across all 28 GoEmotions labels"
    )
    model_name: str = Field(default="RoBERTa-GoEmotions-MultiLabel", description="Model architecture identifier")
    model_version: str = Field(default="v1.0.0", description="Model version tag")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO timestamp")


class EmotionBatchRequest(BaseModel):
    """Input payload for batch emotion analysis."""

    texts: List[str] = Field(..., min_length=1, max_length=500, description="List of review texts to analyze")
    save_to_db: bool = Field(default=False, description="Whether to persist results in database")


class EmotionBatchResponse(BaseModel):
    """Output contract for batch emotion analysis."""

    results: List[EmotionResponse] = Field(..., description="Per-review emotion detection results")
    total_samples: int = Field(..., description="Total reviews analyzed")
    latency_ms: float = Field(..., description="Total batch latency in milliseconds")


class EmotionFrequencyItem(BaseModel):
    """Aggregated emotion label count and ratio."""

    label: str
    count: int
    ratio: float
    cx_category: str


class EmotionAnalyticsResponse(BaseModel):
    """Aggregated enterprise emotion analytics across recorded customer feedback."""

    total_records: int
    cx_distribution: Dict[str, int]
    cx_ratios: Dict[str, float]
    top_raw_emotions: List[EmotionFrequencyItem]
    recent_emotion_records: List[Dict[str, Any]]
