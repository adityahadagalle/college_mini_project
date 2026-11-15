"""Enterprise Analytics API routes – Phase 7.

All business logic is delegated to :class:`EnterpriseAnalyticsService`. Routes
are thin wrappers that translate query parameters to filter objects and return
typed response schemas.
"""

from datetime import date
from typing import Any, Optional

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import User
from app.repositories.analytics_repository import AnalyticsRepository
from app.schemas.analytics import (
    AnalyticsChartBundle,
    AnalyticsFilter,
    AnalyticsOverview,
    AnalyticsTrend,
    PredictionStatistics,
    SystemAnalytics,
    UserAnalyticsResponse,
)
from app.services.analytics_service import EnterpriseAnalyticsService
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


def _build_service(db: AsyncSession) -> EnterpriseAnalyticsService:
    return EnterpriseAnalyticsService(AnalyticsRepository(db))


def _build_filter(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user_id: Optional[str] = None,
    sentiment: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    batch_job_id: Optional[str] = None,
    search: Optional[str] = None,
    prediction_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> AnalyticsFilter:
    return AnalyticsFilter(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        sentiment=sentiment,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        batch_job_id=batch_job_id,
        search=search,
        prediction_id=prediction_id,
        page=page,
        page_size=page_size,
    )


# ------------------------------------------------------------------
# GET /analytics/overview
# ------------------------------------------------------------------


@router.get("/overview", response_model=AnalyticsOverview)  # type: ignore[misc]
async def get_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsOverview:
    """Dashboard overview with counts, ratios, and system metrics."""
    svc = _build_service(db)
    return await svc.get_overview(current_user.id)


# ------------------------------------------------------------------
# GET /analytics/trends
# ------------------------------------------------------------------


@router.get("/trends", response_model=AnalyticsTrend)  # type: ignore[misc]
async def get_analytics_trends(
    granularity: str = Query("daily", pattern="^(daily|weekly|monthly|yearly)$"),
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsTrend:
    """Trend data for a given granularity over the last *days*."""
    svc = _build_service(db)
    return await svc.get_trends(granularity=granularity, days=days, user_id=current_user.id)


# ------------------------------------------------------------------
# GET /analytics/charts
# ------------------------------------------------------------------


@router.get("/charts", response_model=AnalyticsChartBundle)  # type: ignore[misc]
async def get_analytics_charts(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsChartBundle:
    """Chart-ready JSON bundle for frontend rendering."""
    svc = _build_service(db)
    return await svc.get_charts(user_id=current_user.id, days=days)


# ------------------------------------------------------------------
# GET /analytics/report
# ------------------------------------------------------------------


@router.get("/report")  # type: ignore[misc]
async def get_analytics_report(
    report_type: str = Query("predictions"),
    fmt: str = Query("json", pattern="^(json|csv)$"),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    sentiment: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    search: Optional[str] = Query(None),
    prediction_id: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Generate a report in JSON or CSV."""
    svc = _build_service(db)
    filters = _build_filter(
        start_date=start_date,
        end_date=end_date,
        user_id=current_user.id,
        sentiment=sentiment,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        search=search,
        prediction_id=prediction_id,
        page=page,
        page_size=page_size,
    )
    report = await svc.get_report(report_type=report_type, fmt=fmt, filters=filters)

    if fmt == "csv":
        csv_content = svc.report_to_csv(report)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={report_type}_report.csv"},
        )
    return report


# ------------------------------------------------------------------
# GET /analytics/users
# ------------------------------------------------------------------


@router.get("/users", response_model=UserAnalyticsResponse)  # type: ignore[misc]
async def get_analytics_users(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserAnalyticsResponse:
    """Per-user analytics summary."""
    svc = _build_service(db)
    return await svc.get_user_analytics()


# ------------------------------------------------------------------
# GET /analytics/system
# ------------------------------------------------------------------


@router.get("/system", response_model=SystemAnalytics)  # type: ignore[misc]
async def get_analytics_system(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SystemAnalytics:
    """System-level analytics: CPU, memory, DB size, model status."""
    svc = _build_service(db)
    return await svc.get_system_analytics()


# ------------------------------------------------------------------
# GET /analytics/statistics
# ------------------------------------------------------------------


@router.get("/statistics", response_model=PredictionStatistics)  # type: ignore[misc]
async def get_prediction_statistics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PredictionStatistics:
    """Extended prediction statistics for the current user."""
    svc = _build_service(db)
    return await svc.get_prediction_statistics(current_user.id)
