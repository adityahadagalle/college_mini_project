"""Pydantic v2 schemas for Unified Intelligence and Cross-Feature Analytics."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class CapabilityStatusItem(BaseModel):
    """Explicit per-capability operational and execution status."""

    status: Literal["completed", "unavailable", "pending", "failed"] = Field(
        ...,
        description="Execution status: completed, unavailable, pending, or failed",
    )
    model_name: Optional[str] = Field(None, description="Underlying neural model identifier")
    model_version: Optional[str] = Field(None, description="Model deployment version")
    latency_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    error: Optional[str] = Field(None, description="Sanitized error reason if unavailable or failed")


class OverallSentimentResult(BaseModel):
    """Overall sentiment inference result."""

    sentiment: str = Field(..., description="Predicted sentiment: POSITIVE, NEUTRAL, or NEGATIVE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    probabilities: Dict[str, float] = Field(
        ..., description="Class probability distribution (positive, neutral, negative)"
    )
    latency_ms: Optional[float] = Field(None, description="Inference latency in milliseconds")
    explainability: Optional[List[Dict[str, Any]]] = Field(
        None, description="Token-level salience weights if available"
    )


class UnifiedAspectItem(BaseModel):
    """Extracted aspect term with conditioned sentiment."""

    aspect_term: str = Field(..., description="Extracted aspect term from review text")
    aspect_category: Optional[str] = Field(None, description="Standardized category tag if available")
    sentiment: str = Field(..., description="Aspect sentiment: POSITIVE, NEUTRAL, or NEGATIVE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Sentiment classification confidence")
    start_char: int = Field(..., description="Character start offset in text")
    end_char: int = Field(..., description="Character end offset in text")
    evidence_text: Optional[str] = Field(None, description="Contextual snippet around aspect")


class UnifiedEmotionResult(BaseModel):
    """Multi-label customer experience emotion detection result."""

    primary_cx_category: str = Field(
        ..., description="Primary CX category (e.g. DELIGHT, FRUSTRATION, CRITICISM, INQUIRY)"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    detected_emotions: List[Dict[str, Any]] = Field(
        default_factory=list, description="List of detected fine-grained emotion labels and scores"
    )
    all_probabilities: Dict[str, float] = Field(
        default_factory=dict, description="Raw probability distribution across calibrated labels"
    )


class UnifiedTopicProjection(BaseModel):
    """Lightweight projection against frozen topic discovery centroids."""

    topic_id: Optional[str] = Field(None, description="Target discovered topic UUID")
    topic_index: int = Field(..., description="Cluster index (-1 for unassigned/outlier)")
    topic_label: str = Field(..., description="Human-curated or c-TF-IDF keyword label")
    similarity_score: Optional[float] = Field(
        None, description="Cosine similarity score against the topic centroid"
    )
    topic_run_version: Optional[str] = Field(
        None, description="UUID of the completed discovery run used for projection"
    )
    is_outlier: bool = Field(
        default=False, description="True if cosine similarity fell below the 0.45 drift threshold"
    )

    @field_validator("topic_index", mode="before")
    @classmethod
    def _parse_int_bytes(cls, v: Any) -> Any:
        if isinstance(v, bytes):
            return int.from_bytes(v, byteorder="little", signed=True)
        return v


class UnifiedReviewResponse(BaseModel):
    """Complete multi-task response for a single review."""

    review_id: str = Field(..., description="Unique prediction identifier (UUID)")
    text_content: str = Field(..., description="Input review text")
    overall_sentiment: Optional[OverallSentimentResult] = Field(
        None, description="Overall sentiment result from TensorFlow engine"
    )
    aspects: List[UnifiedAspectItem] = Field(
        default_factory=list,
        description="Extracted aspect terms and polarities (empty if none or unavailable)",
    )
    emotions: Optional[UnifiedEmotionResult] = Field(
        None, description="Customer experience emotion classification"
    )
    topic: Optional[UnifiedTopicProjection] = Field(
        None, description="Lightweight projection into discovered topic space"
    )
    capability_status: Dict[str, CapabilityStatusItem] = Field(
        ...,
        description="Explicit per-capability execution status (sentiment, absa, emotion, topics)",
    )
    total_latency_ms: float = Field(..., description="Total wall-clock execution time in milliseconds")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation ISO timestamp",
    )


class IntelligenceAnalyzeRequest(BaseModel):
    """Request payload for unified multi-capability single review analysis."""

    text: str = Field(
        ...,
        min_length=3,
        max_length=5000,
        description="Review text to analyze across sentiment, ABSA, emotion, and topics",
    )
    save_to_db: bool = Field(
        default=True,
        description="Whether to persist prediction, aspects, emotions, and topic assignments to DB",
    )
    project_topic: bool = Field(
        default=True,
        description="Whether to project onto latest completed TopicDiscoveryRun centroids",
    )


class IntelligenceBatchRequest(BaseModel):
    """Batch analysis request payload."""

    reviews: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of review texts (max 500 items per request)",
    )
    save_to_db: bool = Field(
        default=True,
        description="Whether to persist results in the database",
    )
    sync: bool = Field(
        default=True,
        description="Whether to run synchronously (recommended for <= 25 items)",
    )


class IntelligenceBatchJobStatus(BaseModel):
    """Status tracking model for unified batch operations."""

    job_id: str = Field(..., description="Unique batch job UUID")
    status: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        ..., description="Job execution state"
    )
    progress_pct: float = Field(..., ge=0.0, le=100.0, description="Job progress percentage")
    total_records: int = Field(default=0, description="Total input records")
    processed_records: int = Field(default=0, description="Processed records count")
    succeeded_records: int = Field(default=0, description="Succeeded records count")
    failed_records: int = Field(default=0, description="Failed records count")
    partial_records: int = Field(
        default=0, description="Records where one or more capabilities were unavailable"
    )
    error_message: Optional[str] = Field(None, description="Error detail if batch failed")
    created_at: str = Field(..., description="Job creation ISO timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion ISO timestamp")
    results: Optional[List[UnifiedReviewResponse]] = Field(
        None, description="List of unified review results if returned synchronously"
    )


# ==============================================================================
# Cross-Feature Analytics Schemas
# ==============================================================================


class AspectSentimentByTopicItem(BaseModel):
    """Aspect sentiment breakdown grouped by discovered topic."""

    topic_id: str = Field(..., description="Discovered topic UUID")
    topic_name: str = Field(..., description="Topic label")
    topic_index: int = Field(..., description="Cluster index")
    aspect_term: str = Field(..., description="Aspect term")
    aspect_category: Optional[str] = Field(None, description="Aspect category")
    positive_count: int = Field(default=0)
    neutral_count: int = Field(default=0)
    negative_count: int = Field(default=0)
    total_count: int = Field(default=0)
    positive_pct: float = Field(default=0.0)
    negative_pct: float = Field(default=0.0)
    sample_count: int = Field(..., description="Total reviews contributing to this cell")
    low_sample_warning: bool = Field(
        default=False, description="Flagged true if sample_count < 5"
    )


class EmotionBySentimentItem(BaseModel):
    """CX emotion category distribution grouped by overall sentiment."""

    sentiment_label: str = Field(..., description="POSITIVE, NEUTRAL, or NEGATIVE")
    emotion_category: str = Field(..., description="CX emotion category")
    count: int = Field(default=0)
    percentage: float = Field(default=0.0)
    sample_count: int = Field(..., description="Total predictions for this sentiment class")
    low_sample_warning: bool = Field(
        default=False, description="Flagged true if sample_count < 5"
    )


class EmotionByAspectCategoryItem(BaseModel):
    """CX emotion distribution grouped by aspect term or category."""

    aspect_category: str = Field(..., description="Aspect category or term")
    emotion_category: str = Field(..., description="CX emotion category")
    count: int = Field(default=0)
    percentage: float = Field(default=0.0)
    sample_count: int = Field(..., description="Total co-occurrences")
    low_sample_warning: bool = Field(
        default=False, description="Flagged true if sample_count < 5"
    )


class TopicSummaryItem(BaseModel):
    """Per-topic review count, sentiment distribution, and dominant emotion."""

    topic_id: str
    topic_name: str
    topic_index: int
    total_reviews: int
    sentiment_breakdown: Dict[str, float]
    dominant_emotion: str
    sample_count: int
    low_sample_warning: bool


class NegativeAspectTrendItem(BaseModel):
    """Time-bucketed negative aspect occurrences and rates."""

    date: str = Field(..., description="Date formatted as YYYY-MM-DD")
    aspect_term: str = Field(..., description="Aspect term")
    negative_count: int = Field(default=0)
    total_aspect_count: int = Field(default=0)
    negative_rate: float = Field(default=0.0)
    sample_count: int = Field(...)
    low_sample_warning: bool = Field(default=False)


class IntelligenceAnalyticsResponse(BaseModel):
    """Comprehensive cross-feature analytics response."""

    aspect_sentiment_by_topic: List[AspectSentimentByTopicItem]
    emotion_by_overall_sentiment: List[EmotionBySentimentItem]
    emotion_by_aspect_category: List[EmotionByAspectCategoryItem]
    topic_sentiment_and_emotion_summary: List[TopicSummaryItem]
    negative_aspect_trends: List[NegativeAspectTrendItem]
    methodology_note: str = Field(
        default=(
            "Statistical co-occurrence groupings reflect multi-model association across review corpora "
            "rather than direct causal attribution."
        ),
        description="Mandatory methodological disclosure",
    )
    min_sample_size_applied: int = Field(default=1)
    total_analyzed_reviews: int = Field(default=0)


class EvidenceItem(BaseModel):
    """Individual review evidence item matching drilldown filters."""

    prediction_id: str
    review_text: str
    overall_sentiment: str
    confidence_score: Optional[float] = None
    aspect_term: Optional[str] = None
    aspect_sentiment: Optional[str] = None
    emotion_category: Optional[str] = None
    topic_name: Optional[str] = None
    created_at: str


class EvidenceDrilldownResponse(BaseModel):
    """Paginated evidence drilldown query results."""

    filters: Dict[str, Any]
    total_matched: int
    page: int
    page_size: int
    items: List[EvidenceItem]
