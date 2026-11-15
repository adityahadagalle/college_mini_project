"""Enterprise Analytics Service – Phase 7.

Orchestrates analytics calculations by delegating all database access to
:class:`AnalyticsRepository` and returning rich, typed response schemas.
No duplicate code, no raw SQL, no business logic in the repository.
"""

import csv
import io
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import psutil
from app.repositories.analytics_repository import AnalyticsRepository
from app.schemas.analytics import (
    AnalyticsChart,
    AnalyticsChartBundle,
    AnalyticsFilter,
    AnalyticsOverview,
    AnalyticsReport,
    AnalyticsTrend,
    ChartDataset,
    PredictionStatistics,
    SentimentCounts,
    SentimentRatio,
    SystemAnalytics,
    TrendPoint,
    UserAnalytics,
    UserAnalyticsResponse,
)
from app.services.tf_inference import tf_engine
from sqlalchemy.ext.asyncio import AsyncSession

_APP_START_TIME = time.time()


class EnterpriseAnalyticsService:
    """Business logic for enterprise analytics & BI."""

    def __init__(self, repo: AnalyticsRepository) -> None:
        self._repo = repo

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _start_of_today() -> datetime:
        return datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=timezone.utc)

    @staticmethod
    def _start_of_week() -> datetime:
        today = date.today()
        start = today - timedelta(days=today.weekday())
        return datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)

    @staticmethod
    def _start_of_month() -> datetime:
        today = date.today()
        start = today.replace(day=1)
        return datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)

    def _filter_dates(self, f: AnalyticsFilter) -> tuple[Optional[datetime], Optional[datetime]]:
        start: Optional[datetime] = None
        end: Optional[datetime] = None
        if f.start_date:
            start = datetime.combine(f.start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        if f.end_date:
            end = datetime.combine(f.end_date + timedelta(days=1), datetime.min.time()).replace(
                tzinfo=timezone.utc
            )
        return start, end

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    async def get_overview(self, user_id: Optional[str] = None) -> AnalyticsOverview:
        now = self._now()
        total = await self._repo.total_predictions(user_id)
        today_count = await self._repo.predictions_in_range(self._start_of_today(), now, user_id)
        week_count = await self._repo.predictions_in_range(self._start_of_week(), now, user_id)
        month_count = await self._repo.predictions_in_range(self._start_of_month(), now, user_id)

        counts = await self._repo.sentiment_counts(user_id)
        avg_conf = await self._repo.avg_confidence(user_id)
        avg_lat = await self._repo.avg_latency(user_id)
        active_users = await self._repo.active_user_count()
        active_sessions = await self._repo.active_session_count()
        batch = await self._repo.batch_job_counts(user_id)

        s_total = counts["POSITIVE"] + counts["NEUTRAL"] + counts["NEGATIVE"]
        ratio = SentimentRatio(
            positive=round(counts["POSITIVE"] / s_total, 4) if s_total else 0.0,
            neutral=round(counts["NEUTRAL"] / s_total, 4) if s_total else 0.0,
            negative=round(counts["NEGATIVE"] / s_total, 4) if s_total else 0.0,
        )

        return AnalyticsOverview(
            total_predictions=total,
            predictions_today=today_count,
            predictions_this_week=week_count,
            predictions_this_month=month_count,
            sentiment_counts=SentimentCounts(
                positive=counts["POSITIVE"],
                neutral=counts["NEUTRAL"],
                negative=counts["NEGATIVE"],
            ),
            sentiment_ratio=ratio,
            average_confidence=avg_conf,
            average_latency_ms=avg_lat,
            active_users=active_users,
            active_sessions=active_sessions,
            batch_jobs_total=batch["total"],
            batch_jobs_failed=batch["failed"],
            system_uptime_seconds=round(time.time() - _APP_START_TIME, 2),
        )

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------

    async def get_trends(
        self,
        granularity: str = "daily",
        days: int = 30,
        user_id: Optional[str] = None,
    ) -> AnalyticsTrend:
        raw = await self._repo.daily_trend(days=days, user_id=user_id)
        points: List[TrendPoint] = []
        for row in raw:
            points.append(
                TrendPoint(
                    period=row["period"],
                    total=row["total"],
                    positive=row["positive"],
                    neutral=row["neutral"],
                    negative=row["negative"],
                    avg_confidence=row["avg_confidence"],
                    avg_latency_ms=row["avg_latency_ms"],
                )
            )

        if granularity == "weekly":
            points = self._aggregate_weekly(points)
        elif granularity == "monthly":
            points = self._aggregate_monthly(points)
        elif granularity == "yearly":
            points = self._aggregate_yearly(points)

        return AnalyticsTrend(granularity=granularity, data=points)

    @staticmethod
    def _aggregate_weekly(points: List[TrendPoint]) -> List[TrendPoint]:
        buckets: Dict[str, TrendPoint] = {}
        for p in points:
            try:
                d = date.fromisoformat(p.period)
            except ValueError:
                continue
            week_start = d - timedelta(days=d.weekday())
            key = week_start.isoformat()
            if key not in buckets:
                buckets[key] = TrendPoint(period=key)
            b = buckets[key]
            b.total += p.total
            b.positive += p.positive
            b.neutral += p.neutral
            b.negative += p.negative
        # Compute weighted averages
        for key, b in buckets.items():
            if b.total > 0:
                matching = [pt for pt in points if _date_in_week(pt.period, key)]
                total_count = sum(pt.total for pt in matching)
                if total_count > 0:
                    b.avg_confidence = round(
                        sum(pt.avg_confidence * pt.total for pt in matching) / total_count,
                        4,
                    )
                    b.avg_latency_ms = round(
                        sum(pt.avg_latency_ms * pt.total for pt in matching) / total_count,
                        2,
                    )
        return sorted(buckets.values(), key=lambda x: x.period)

    @staticmethod
    def _aggregate_monthly(points: List[TrendPoint]) -> List[TrendPoint]:
        buckets: Dict[str, TrendPoint] = {}
        for p in points:
            key = p.period[:7]  # "YYYY-MM"
            if key not in buckets:
                buckets[key] = TrendPoint(period=key)
            b = buckets[key]
            b.total += p.total
            b.positive += p.positive
            b.neutral += p.neutral
            b.negative += p.negative
        for key, b in buckets.items():
            matching = [pt for pt in points if pt.period.startswith(key)]
            total_count = sum(pt.total for pt in matching)
            if total_count > 0:
                b.avg_confidence = round(
                    sum(pt.avg_confidence * pt.total for pt in matching) / total_count, 4
                )
                b.avg_latency_ms = round(
                    sum(pt.avg_latency_ms * pt.total for pt in matching) / total_count, 2
                )
        return sorted(buckets.values(), key=lambda x: x.period)

    @staticmethod
    def _aggregate_yearly(points: List[TrendPoint]) -> List[TrendPoint]:
        buckets: Dict[str, TrendPoint] = {}
        for p in points:
            key = p.period[:4]  # "YYYY"
            if key not in buckets:
                buckets[key] = TrendPoint(period=key)
            b = buckets[key]
            b.total += p.total
            b.positive += p.positive
            b.neutral += p.neutral
            b.negative += p.negative
        for key, b in buckets.items():
            matching = [pt for pt in points if pt.period.startswith(key)]
            total_count = sum(pt.total for pt in matching)
            if total_count > 0:
                b.avg_confidence = round(
                    sum(pt.avg_confidence * pt.total for pt in matching) / total_count, 4
                )
                b.avg_latency_ms = round(
                    sum(pt.avg_latency_ms * pt.total for pt in matching) / total_count, 2
                )
        return sorted(buckets.values(), key=lambda x: x.period)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    async def get_charts(
        self, user_id: Optional[str] = None, days: int = 30
    ) -> AnalyticsChartBundle:
        counts = await self._repo.sentiment_counts(user_id)
        trend_data = await self._repo.daily_trend(days=days, user_id=user_id)
        hourly = await self._repo.hourly_distribution(days=days, user_id=user_id)

        # Sentiment pie chart
        pie = AnalyticsChart(
            chart_type="pie",
            title="Sentiment Distribution",
            labels=["Positive", "Neutral", "Negative"],
            datasets=[
                ChartDataset(
                    label="Count",
                    data=[
                        float(counts["POSITIVE"]),
                        float(counts["NEUTRAL"]),
                        float(counts["NEGATIVE"]),
                    ],
                )
            ],
        )

        # Volume line chart
        volume_labels = [r["period"] for r in trend_data]
        volume = AnalyticsChart(
            chart_type="line",
            title="Prediction Volume",
            labels=volume_labels,
            datasets=[
                ChartDataset(
                    label="Total Predictions",
                    data=[float(r["total"]) for r in trend_data],
                )
            ],
        )

        # Confidence bar chart
        conf_bar = AnalyticsChart(
            chart_type="bar",
            title="Average Confidence by Day",
            labels=volume_labels,
            datasets=[
                ChartDataset(
                    label="Avg Confidence",
                    data=[r["avg_confidence"] for r in trend_data],
                )
            ],
        )

        # Latency area chart
        latency_area = AnalyticsChart(
            chart_type="area",
            title="Average Latency (ms)",
            labels=volume_labels,
            datasets=[
                ChartDataset(
                    label="Avg Latency",
                    data=[r["avg_latency_ms"] for r in trend_data],
                )
            ],
        )

        # Sentiment trend (stacked)
        sentiment_trend = AnalyticsChart(
            chart_type="trend",
            title="Sentiment Trend",
            labels=volume_labels,
            datasets=[
                ChartDataset(
                    label="Positive",
                    data=[float(r["positive"]) for r in trend_data],
                ),
                ChartDataset(
                    label="Neutral",
                    data=[float(r["neutral"]) for r in trend_data],
                ),
                ChartDataset(
                    label="Negative",
                    data=[float(r["negative"]) for r in trend_data],
                ),
            ],
        )

        # Heatmap
        day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        heatmap_datasets: List[ChartDataset] = []
        for d_idx, d_name in enumerate(day_names):
            row_data = [0.0] * 24
            for entry in hourly:
                if entry["day_of_week"] == d_idx:
                    row_data[entry["hour"]] = float(entry["count"])
            heatmap_datasets.append(ChartDataset(label=d_name, data=row_data))

        heatmap = AnalyticsChart(
            chart_type="heatmap",
            title="Hourly Activity Heatmap",
            labels=hour_labels,
            datasets=heatmap_datasets,
        )

        return AnalyticsChartBundle(
            sentiment_pie=pie,
            volume_line=volume,
            confidence_bar=conf_bar,
            latency_area=latency_area,
            sentiment_trend=sentiment_trend,
            hourly_heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    async def get_report(
        self,
        report_type: str = "predictions",
        fmt: str = "json",
        filters: Optional[AnalyticsFilter] = None,
    ) -> AnalyticsReport:
        f = filters or AnalyticsFilter()
        start, end = self._filter_dates(f)

        if report_type == "predictions":
            return await self._prediction_report(f, start, end)
        elif report_type == "users":
            return await self._user_report(f)
        elif report_type == "batch":
            return await self._batch_report(f)
        else:
            return await self._analytics_summary_report(f, start, end)

    async def _prediction_report(
        self,
        f: AnalyticsFilter,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> AnalyticsReport:
        rows, total = await self._repo.filtered_predictions(
            user_id=f.user_id,
            sentiment=f.sentiment,
            min_confidence=f.min_confidence,
            max_confidence=f.max_confidence,
            start=start,
            end=end,
            search=f.search,
            prediction_id=f.prediction_id,
            page=f.page,
            page_size=f.page_size,
        )

        data: List[Dict[str, Any]] = []
        for p in rows:
            data.append(
                {
                    "id": p.id,
                    "user_id": p.user_id,
                    "text_content": p.text_content,
                    "sentiment": p.sentiment_label,
                    "confidence": p.confidence_score,
                    "latency_ms": p.latency_ms,
                    "created_at": str(p.created_at) if p.created_at else None,
                }
            )

        # Summary
        counts = await self._repo.sentiment_counts(f.user_id, start, end)
        avg_conf = await self._repo.avg_confidence(f.user_id, start, end)

        return AnalyticsReport(
            report_type="predictions",
            generated_at=self._now().isoformat(),
            total_records=total,
            page=f.page,
            page_size=f.page_size,
            data=data,
            summary={
                "positive": counts["POSITIVE"],
                "neutral": counts["NEUTRAL"],
                "negative": counts["NEGATIVE"],
                "avg_confidence": avg_conf,
            },
        )

    async def _user_report(self, f: AnalyticsFilter) -> AnalyticsReport:
        user_data = await self._repo.per_user_analytics()
        return AnalyticsReport(
            report_type="users",
            generated_at=self._now().isoformat(),
            total_records=len(user_data),
            page=1,
            page_size=len(user_data) or 1,
            data=user_data,
            summary={"total_users": len(user_data)},
        )

    async def _batch_report(self, f: AnalyticsFilter) -> AnalyticsReport:
        batch = await self._repo.batch_job_counts(f.user_id)
        return AnalyticsReport(
            report_type="batch",
            generated_at=self._now().isoformat(),
            total_records=batch["total"],
            page=1,
            page_size=batch["total"] or 1,
            data=[batch],
            summary=batch,
        )

    async def _analytics_summary_report(
        self,
        f: AnalyticsFilter,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> AnalyticsReport:
        total = await self._repo.total_predictions(f.user_id)
        counts = await self._repo.sentiment_counts(f.user_id, start, end)
        avg_conf = await self._repo.avg_confidence(f.user_id, start, end)
        avg_lat = await self._repo.avg_latency(f.user_id, start, end)

        summary: Dict[str, Any] = {
            "total_predictions": total,
            **counts,
            "avg_confidence": avg_conf,
            "avg_latency_ms": avg_lat,
        }

        return AnalyticsReport(
            report_type="analytics",
            generated_at=self._now().isoformat(),
            total_records=total,
            page=1,
            page_size=total or 1,
            data=[summary],
            summary=summary,
        )

    # ------------------------------------------------------------------
    # CSV export helper
    # ------------------------------------------------------------------

    def report_to_csv(self, report: AnalyticsReport) -> str:
        output = io.StringIO()
        if not report.data:
            return ""
        writer = csv.DictWriter(output, fieldnames=list(report.data[0].keys()))
        writer.writeheader()
        for row in report.data:
            writer.writerow(row)
        return output.getvalue()

    # ------------------------------------------------------------------
    # User analytics
    # ------------------------------------------------------------------

    async def get_user_analytics(self) -> UserAnalyticsResponse:
        raw = await self._repo.per_user_analytics()
        users: List[UserAnalytics] = []
        for r in raw:
            users.append(
                UserAnalytics(
                    user_id=r["user_id"],
                    email=r["email"],
                    total_predictions=r["total_predictions"],
                    positive_count=r["positive_count"],
                    neutral_count=r["neutral_count"],
                    negative_count=r["negative_count"],
                    avg_confidence=r["avg_confidence"],
                    avg_latency_ms=r["avg_latency_ms"],
                    last_active=r["last_active"],
                )
            )
        return UserAnalyticsResponse(total_users=len(users), data=users)

    # ------------------------------------------------------------------
    # System analytics
    # ------------------------------------------------------------------

    async def get_system_analytics(self) -> SystemAnalytics:
        mem = psutil.virtual_memory()
        total_preds = await self._repo.total_predictions()
        total_users = await self._repo.total_user_count()
        batch = await self._repo.batch_job_counts()
        sessions = await self._repo.active_session_count()
        db_size = await self._repo.db_size_mb()

        return SystemAnalytics(
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_mb=round(mem.used / (1024 * 1024), 2),
            memory_percent=mem.percent,
            uptime_seconds=round(time.time() - _APP_START_TIME, 2),
            model_loaded=tf_engine.is_warm,
            total_predictions=total_preds,
            total_users=total_users,
            total_batch_jobs=batch["total"],
            active_sessions=sessions,
            db_size_mb=db_size,
        )

    # ------------------------------------------------------------------
    # Prediction statistics
    # ------------------------------------------------------------------

    async def get_prediction_statistics(
        self, user_id: Optional[str] = None
    ) -> PredictionStatistics:
        total = await self._repo.total_predictions(user_id)
        counts = await self._repo.sentiment_counts(user_id)
        avg_conf = await self._repo.avg_confidence(user_id)
        avg_lat = await self._repo.avg_latency(user_id)
        conf_range = await self._repo.confidence_range(user_id)

        pos = counts["POSITIVE"]
        neu = counts["NEUTRAL"]
        neg = counts["NEGATIVE"]

        return PredictionStatistics(
            total=total,
            positive_count=pos,
            neutral_count=neu,
            negative_count=neg,
            positive_pct=round(pos / total * 100, 2) if total else 0.0,
            neutral_pct=round(neu / total * 100, 2) if total else 0.0,
            negative_pct=round(neg / total * 100, 2) if total else 0.0,
            avg_confidence=avg_conf,
            avg_latency_ms=avg_lat,
            min_confidence=conf_range[0],
            max_confidence=conf_range[1],
            success_count=total,
            failure_count=0,
        )


# ---------------------------------------------------------------------------
# module-level helpers
# ---------------------------------------------------------------------------


def _date_in_week(period_str: str, week_start_str: str) -> bool:
    """Check if a date string falls in the ISO week starting on *week_start_str*."""
    try:
        d = date.fromisoformat(period_str)
        ws = date.fromisoformat(week_start_str)
        return ws <= d < ws + timedelta(days=7)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Backward-compatible wrapper used by the existing analytics endpoint
# ---------------------------------------------------------------------------


async def get_analytics_overview_data(user_id: str, db: AsyncSession) -> AnalyticsOverview:
    """Backward-compatible function called by the existing analytics route."""
    repo = AnalyticsRepository(db)
    service = EnterpriseAnalyticsService(repo)
    return await service.get_overview(user_id)
