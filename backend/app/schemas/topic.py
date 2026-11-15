"""Pydantic v2 schemas for Collection-Level Automatic Topic Discovery."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TopicKeywordItem(BaseModel):
    """Keyword extracted via c-TF-IDF representation."""

    word: str = Field(..., description="Keyword token or unigram/bigram")
    score: float = Field(..., description="c-TF-IDF importance score")


class DiscoveredTopicItem(BaseModel):
    """Discovered topic metadata and summary statistics."""

    id: str = Field(..., description="Unique UUID for discovered topic")
    run_id: str = Field(..., description="Discovery run UUID")
    topic_index: int = Field(
        ...,
        description="Cluster index (-1 for Outliers / Unassigned, 0..N for discovered clusters)",
    )
    auto_label: str = Field(
        ..., description="Automatically generated label from top c-TF-IDF keywords"
    )
    human_label: Optional[str] = Field(
        None, description="Human curated / renamed topic label"
    )
    display_label: str = Field(
        ..., description="Display label (human_label if present, else auto_label)"
    )
    keywords: List[TopicKeywordItem] = Field(
        default_factory=list, description="Top c-TF-IDF keywords for the topic"
    )
    review_count: int = Field(default=0, description="Total reviews assigned to this topic")
    cohesion_score: Optional[float] = Field(
        None, description="Average cosine similarity of member reviews to the topic centroid"
    )
    sentiment_breakdown: Optional[Dict[str, float]] = Field(
        None,
        description="Sentiment distribution ratio (e.g. {'POSITIVE': 0.6, 'NEGATIVE': 0.4})",
    )
    is_archived: bool = Field(
        default=False, description="Whether topic has been archived/hidden by human curator"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation ISO timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Last update ISO timestamp",
    )

    @field_validator("topic_index", "review_count", mode="before")
    @classmethod
    def _parse_int_bytes(cls, v: Any) -> Any:
        if isinstance(v, bytes):
            return int.from_bytes(v, byteorder="little", signed=True)
        return v


class TopicReviewAssignmentItem(BaseModel):
    """Review item assigned to a topic."""

    id: str = Field(..., description="Unique assignment UUID")
    topic_id: str = Field(..., description="Discovered topic UUID")
    prediction_id: Optional[str] = Field(
        None, description="Associated prediction record UUID if ingested from DB"
    )
    review_text: str = Field(..., description="Cleaned review content")
    distance_to_centroid: Optional[float] = Field(
        None, description="Cosine distance (1 - cosine_similarity) to cluster centroid"
    )
    is_outlier: bool = Field(
        default=False, description="Whether review was marked as an unclustered outlier"
    )
    is_representative: bool = Field(
        default=False,
        description="Whether review is one of the top 3 closest examples to the cluster centroid",
    )
    sentiment_label: Optional[str] = Field(
        None, description="Sentiment label (POSITIVE, NEUTRAL, NEGATIVE) if available"
    )
    confidence_score: Optional[float] = Field(
        None, description="Sentiment confidence score if available"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Assignment ISO timestamp",
    )


class TopicDiscoveryRequest(BaseModel):
    """Request payload to initiate a topic discovery run."""

    source: str = Field(
        default="all_predictions",
        description="Data source: 'all_predictions', 'date_range', or 'custom_texts'",
    )
    custom_texts: Optional[List[str]] = Field(
        default=None,
        description="Explicit list of review texts (required if source='custom_texts')",
    )
    start_date: Optional[str] = Field(
        default=None, description="Start date ISO string filter for DB predictions"
    )
    end_date: Optional[str] = Field(
        default=None, description="End date ISO string filter for DB predictions"
    )
    sentiment_filter: Optional[str] = Field(
        default=None,
        description="Optional sentiment filter ('POSITIVE', 'NEGATIVE', 'NEUTRAL')",
    )
    min_cluster_size: int = Field(
        default=3,
        ge=2,
        le=50,
        description="Minimum cluster size parameter for HDBSCAN clustering",
    )
    min_samples: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="min_samples parameter for HDBSCAN conservative clustering",
    )
    n_neighbors: int = Field(
        default=15,
        ge=2,
        le=50,
        description="n_neighbors parameter for UMAP dimensionality reduction",
    )
    n_components: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Target dimensions for UMAP (default 5 for clustering)",
    )
    random_state: int = Field(
        default=42,
        description="Fixed random seed for deterministic UMAP reduction",
    )


class TopicDiscoveryJobStatus(BaseModel):
    """Job status and progress tracking response."""

    job_id: str = Field(..., description="Unique discovery run / job UUID")
    status: str = Field(..., description="Job status: PENDING, PROCESSING, COMPLETED, FAILED")
    progress: float = Field(
        ..., ge=0.0, le=1.0, description="Job completion ratio from 0.0 to 1.0"
    )
    stage: str = Field(..., description="Current pipeline execution stage description")
    corpus_size: int = Field(default=0, description="Total unique reviews in corpus")
    num_topics: int = Field(
        default=0, description="Number of discovered topics (excluding outlier cluster)"
    )
    outlier_count: int = Field(default=0, description="Number of reviews classified as outliers")
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Quality metrics including intra-cluster cohesion and keyword diversity",
    )
    error_message: Optional[str] = Field(None, description="Error detail if job failed")
    created_at: str = Field(..., description="Job creation ISO timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion ISO timestamp")


class TopicListResponse(BaseModel):
    """Response containing discovered topics for a run."""

    run_id: str = Field(..., description="Topic discovery run UUID")
    corpus_size: int = Field(..., description="Total unique reviews analyzed")
    num_topics: int = Field(..., description="Number of meaningful topics discovered")
    outlier_count: int = Field(..., description="Number of unassigned outlier reviews")
    metrics: Optional[Dict[str, Any]] = Field(
        None, description="Run-level quality metrics (cohesion, diversity)"
    )
    topics: List[DiscoveredTopicItem] = Field(
        default_factory=list, description="List of discovered topics"
    )
    all_runs: List[Dict[str, Any]] = Field(
        default_factory=list, description="Historical discovery runs summary"
    )


class TopicReviewsResponse(BaseModel):
    """Paginated review assignments for a specific topic."""

    topic_id: str = Field(..., description="Topic UUID")
    topic_name: str = Field(..., description="Topic display name")
    topic_index: int = Field(..., description="Cluster index (-1 for outliers)")
    total_reviews: int = Field(..., description="Total reviews in this topic")
    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Page size limit")
    reviews: List[TopicReviewAssignmentItem] = Field(
        default_factory=list, description="Paginated review assignments"
    )


class TopicRenameRequest(BaseModel):
    """Request payload to rename a topic."""

    human_label: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="New human curated label for this topic",
    )


class TopicMergeRequest(BaseModel):
    """Request payload to merge two topics together."""

    source_topic_id: str = Field(
        ..., description="Topic UUID to merge from (will be marked archived)"
    )
    target_topic_id: str = Field(
        ..., description="Topic UUID to merge into (receives assignments and updated count)"
    )
    new_label: Optional[str] = Field(
        None, description="Optional new human label for the target topic"
    )


class MatchedTopicPair(BaseModel):
    """Matched topic pair across two discovery runs."""

    topic_a_id: str
    topic_a_name: str
    topic_b_id: str
    topic_b_name: str
    similarity: float = Field(
        ..., description="Cosine similarity between cluster centroids (0.0 to 1.0)"
    )


class CandidateEmergingTopic(BaseModel):
    """Candidate emerging theme with similarity < 0.45 across runs."""

    topic_id: str
    topic_name: str
    topic_index: int
    review_count: int
    max_similarity_to_prior_run: float
    keywords: List[TopicKeywordItem]
    uncertainty_note: str = Field(
        default="Candidate emerging theme - requires human review due to unsupervised cluster drift",
        description="Mandatory disclaimer distinguishing unsupervised drift from confirmed business trends",
    )

    @field_validator("topic_index", "review_count", mode="before")
    @classmethod
    def _parse_int_bytes(cls, v: Any) -> Any:
        if isinstance(v, bytes):
            return int.from_bytes(v, byteorder="little", signed=True)
        return v


class TopicComparisonResponse(BaseModel):
    """Comparison result between two topic discovery runs."""

    run_id_a: str
    run_id_b: str
    matched_topics: List[MatchedTopicPair] = Field(
        default_factory=list, description="Topics shared between runs (similarity >= 0.45)"
    )
    emerging_topics_in_b: List[CandidateEmergingTopic] = Field(
        default_factory=list,
        description="Novel candidate clusters in Run B with max similarity < 0.45",
    )
    faded_topics_in_a: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clusters present in Run A with low alignment in Run B",
    )


class TopicMetadataResponse(BaseModel):
    """Model card and benchmark metadata for Automatic Topic Discovery."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    architecture: str = "CPU-Optimized Embeddings + UMAP + HDBSCAN + c-TF-IDF"
    device: str = "cpu"
    vram_usage_mb: float = 0.0
    embedding_dimension: int = 384
    min_reviews_required: int = 15
    local_weights_verified: bool = True
    benchmark_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Comparative algorithm benchmarks (UMAP+HDBSCAN vs KMeans vs Agglomerative)",
    )
