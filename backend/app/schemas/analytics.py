"""Enterprise Analytics & Business Intelligence schemas.

Provides Pydantic models for analytics endpoints: overview, trends, charts,
reports, filtering, and user/system analytics.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


class AnalyticsFilter(BaseModel):
    """Shared filter parameters accepted by analytics endpoints."""

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    user_id: Optional[str] = None
    sentiment: Optional[str] = None
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    batch_job_id: Optional[str] = None
    search: Optional[str] = None
    prediction_id: Optional[str] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


class SentimentCounts(BaseModel):
    """Absolute counts per sentiment label."""

    positive: int = 0
    neutral: int = 0
    negative: int = 0


class SentimentRatio(BaseModel):
    """Percentage ratios per sentiment label (0.0‑1.0)."""

    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0


class AnalyticsOverview(BaseModel):
    """Dashboard overview response."""

    total_predictions: int = 0
    predictions_today: int = 0
    predictions_this_week: int = 0
    predictions_this_month: int = 0
    sentiment_counts: SentimentCounts = Field(default_factory=SentimentCounts)
    sentiment_ratio: SentimentRatio = Field(default_factory=SentimentRatio)
    average_confidence: float = 0.0
    average_latency_ms: float = 0.0
    active_users: int = 0
    active_sessions: int = 0
    batch_jobs_total: int = 0
    batch_jobs_failed: int = 0
    system_uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------


class TrendPoint(BaseModel):
    """Single data point on a trend line."""

    period: str
    total: int = 0
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0


class AnalyticsTrend(BaseModel):
    """Trend data over a time range."""

    granularity: str  # "daily" | "weekly" | "monthly" | "yearly"
    data: List[TrendPoint] = []


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


class ChartDataset(BaseModel):
    """A single series inside a chart."""

    label: str
    data: List[float] = []


class AnalyticsChart(BaseModel):
    """Chart-ready JSON response."""

    chart_type: str  # "line" | "bar" | "pie" | "area" | "heatmap" | "trend"
    title: str
    labels: List[str] = []
    datasets: List[ChartDataset] = []
    metadata: Dict[str, Any] = {}


class AnalyticsChartBundle(BaseModel):
    """Collection of all charts returned by ``GET /analytics/charts``."""

    sentiment_pie: AnalyticsChart
    volume_line: AnalyticsChart
    confidence_bar: AnalyticsChart
    latency_area: AnalyticsChart
    sentiment_trend: AnalyticsChart
    hourly_heatmap: AnalyticsChart


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


class ReportRow(BaseModel):
    """Single row inside a JSON report."""

    id: str
    text_content: str
    sentiment: str
    confidence: float
    latency_ms: float
    created_at: Optional[str] = None
    user_id: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AnalyticsReport(BaseModel):
    """Report response envelope."""

    report_type: str  # "predictions" | "analytics" | "users" | "batch"
    generated_at: str
    total_records: int
    page: int
    page_size: int
    data: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# User analytics
# ---------------------------------------------------------------------------


class UserAnalytics(BaseModel):
    """Per-user statistics."""

    user_id: str
    email: str
    total_predictions: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    last_active: Optional[str] = None


class UserAnalyticsResponse(BaseModel):
    """Collection of user analytics rows."""

    total_users: int
    data: List[UserAnalytics] = []


# ---------------------------------------------------------------------------
# System analytics
# ---------------------------------------------------------------------------


class SystemAnalytics(BaseModel):
    """System-level performance metrics."""

    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    uptime_seconds: float = 0.0
    model_loaded: bool = False
    total_predictions: int = 0
    total_users: int = 0
    total_batch_jobs: int = 0
    active_sessions: int = 0
    db_size_mb: float = 0.0


# ---------------------------------------------------------------------------
# Prediction statistics (extended)
# ---------------------------------------------------------------------------


class PredictionStatistics(BaseModel):
    """Extended prediction statistics."""

    total: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    positive_pct: float = 0.0
    neutral_pct: float = 0.0
    negative_pct: float = 0.0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    success_count: int = 0
    failure_count: int = 0
