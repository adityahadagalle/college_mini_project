"""Analytics repository – all database queries for analytics.

Pure data access layer. No business logic. All queries use async SQLAlchemy
and return raw rows or scalar aggregates.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.models.auth import Session as SessionModel
from app.models.database import BatchJob, Prediction, User
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession


class AnalyticsRepository:
    """Data access for analytics aggregate queries."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Counts & aggregates
    # ------------------------------------------------------------------

    async def total_predictions(self, user_id: Optional[str] = None) -> int:
        stmt = select(func.count(Prediction.id))
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        result = await self._db.execute(stmt)
        val = result.scalar()
        return int(val) if val else 0

    async def predictions_in_range(
        self,
        start: datetime,
        end: datetime,
        user_id: Optional[str] = None,
    ) -> int:
        stmt = select(func.count(Prediction.id)).where(
            Prediction.created_at >= start,
            Prediction.created_at < end,
        )
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        result = await self._db.execute(stmt)
        val = result.scalar()
        return int(val) if val else 0

    async def sentiment_counts(
        self,
        user_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, int]:
        stmt = select(
            Prediction.sentiment_label,
            func.count(Prediction.id),
        ).group_by(Prediction.sentiment_label)
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        if start:
            stmt = stmt.where(Prediction.created_at >= start)
        if end:
            stmt = stmt.where(Prediction.created_at < end)
        result = await self._db.execute(stmt)
        rows = result.all()
        counts: Dict[str, int] = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        for label, cnt in rows:
            counts[str(label).upper()] = int(cnt)
        return counts

    async def avg_confidence(
        self,
        user_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> float:
        stmt = select(func.avg(Prediction.confidence_score))
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        if start:
            stmt = stmt.where(Prediction.created_at >= start)
        if end:
            stmt = stmt.where(Prediction.created_at < end)
        result = await self._db.execute(stmt)
        val = result.scalar()
        return round(float(val), 4) if val else 0.0

    async def avg_latency(
        self,
        user_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> float:
        stmt = select(func.avg(Prediction.latency_ms))
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        if start:
            stmt = stmt.where(Prediction.created_at >= start)
        if end:
            stmt = stmt.where(Prediction.created_at < end)
        result = await self._db.execute(stmt)
        val = result.scalar()
        return round(float(val), 2) if val else 0.0

    async def confidence_range(
        self,
        user_id: Optional[str] = None,
    ) -> Tuple[float, float]:
        stmt = select(
            func.min(Prediction.confidence_score),
            func.max(Prediction.confidence_score),
        )
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        result = await self._db.execute(stmt)
        row = result.one_or_none()
        if row and row[0] is not None:
            return (round(float(row[0]), 4), round(float(row[1]), 4))
        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # Active users / sessions
    # ------------------------------------------------------------------

    async def active_user_count(self) -> int:
        stmt = select(func.count(User.id)).where(User.is_active.is_(True))
        result = await self._db.execute(stmt)
        val = result.scalar()
        return int(val) if val else 0

    async def active_session_count(self) -> int:
        now = datetime.now(timezone.utc)
        stmt = select(func.count(SessionModel.id)).where(
            SessionModel.revoked.is_(False),
            SessionModel.expires_at > now,
        )
        result = await self._db.execute(stmt)
        val = result.scalar()
        return int(val) if val else 0

    # ------------------------------------------------------------------
    # Batch jobs
    # ------------------------------------------------------------------

    async def batch_job_counts(self, user_id: Optional[str] = None) -> Dict[str, int]:
        stmt = select(
            func.count(BatchJob.id),
            func.count(case((BatchJob.status == "FAILED", BatchJob.id))),
        )
        if user_id:
            stmt = stmt.where(BatchJob.user_id == user_id)
        result = await self._db.execute(stmt)
        row = result.one()
        return {
            "total": int(row[0]) if row[0] else 0,
            "failed": int(row[1]) if row[1] else 0,
        }

    # ------------------------------------------------------------------
    # Trends
    # ------------------------------------------------------------------

    async def daily_trend(
        self,
        days: int = 30,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        date_col = func.strftime("%Y-%m-%d", Prediction.created_at)
        stmt = (
            select(
                date_col.label("day"),
                func.count(Prediction.id).label("total"),
                func.count(case((Prediction.sentiment_label == "POSITIVE", Prediction.id))).label(
                    "positive"
                ),
                func.count(case((Prediction.sentiment_label == "NEUTRAL", Prediction.id))).label(
                    "neutral"
                ),
                func.count(case((Prediction.sentiment_label == "NEGATIVE", Prediction.id))).label(
                    "negative"
                ),
                func.avg(Prediction.confidence_score).label("avg_conf"),
                func.avg(Prediction.latency_ms).label("avg_lat"),
            )
            .where(Prediction.created_at >= cutoff)
            .group_by(date_col)
            .order_by(date_col)
        )
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        result = await self._db.execute(stmt)
        rows = result.all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "period": str(r[0]),
                    "total": int(r[1]),
                    "positive": int(r[2]),
                    "neutral": int(r[3]),
                    "negative": int(r[4]),
                    "avg_confidence": round(float(r[5]), 4) if r[5] else 0.0,
                    "avg_latency_ms": round(float(r[6]), 2) if r[6] else 0.0,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Filtered predictions (pagination)
    # ------------------------------------------------------------------

    async def filtered_predictions(
        self,
        user_id: Optional[str] = None,
        sentiment: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        search: Optional[str] = None,
        prediction_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Prediction], int]:
        base = select(Prediction)
        count_stmt = select(func.count(Prediction.id))

        conditions = []
        if user_id:
            conditions.append(Prediction.user_id == user_id)
        if sentiment:
            conditions.append(Prediction.sentiment_label == sentiment.upper())
        if min_confidence is not None:
            conditions.append(Prediction.confidence_score >= min_confidence)
        if max_confidence is not None:
            conditions.append(Prediction.confidence_score <= max_confidence)
        if start:
            conditions.append(Prediction.created_at >= start)
        if end:
            conditions.append(Prediction.created_at < end)
        if search:
            conditions.append(Prediction.text_content.ilike(f"%{search}%"))
        if prediction_id:
            conditions.append(Prediction.id == prediction_id)

        for cond in conditions:
            base = base.where(cond)
            count_stmt = count_stmt.where(cond)

        count_result = await self._db.execute(count_stmt)
        total = int(count_result.scalar() or 0)

        base = (
            base.order_by(Prediction.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self._db.execute(base)
        rows: List[Prediction] = list(result.scalars().all())
        return rows, total

    # ------------------------------------------------------------------
    # Per-user analytics
    # ------------------------------------------------------------------

    async def per_user_analytics(self) -> List[Dict[str, Any]]:
        stmt = (
            select(
                User.id,
                User.email,
                func.count(Prediction.id).label("total"),
                func.count(case((Prediction.sentiment_label == "POSITIVE", Prediction.id))).label(
                    "positive"
                ),
                func.count(case((Prediction.sentiment_label == "NEUTRAL", Prediction.id))).label(
                    "neutral"
                ),
                func.count(case((Prediction.sentiment_label == "NEGATIVE", Prediction.id))).label(
                    "negative"
                ),
                func.avg(Prediction.confidence_score).label("avg_conf"),
                func.avg(Prediction.latency_ms).label("avg_lat"),
                func.max(Prediction.created_at).label("last_active"),
            )
            .outerjoin(Prediction, User.id == Prediction.user_id)
            .group_by(User.id, User.email)
            .order_by(func.count(Prediction.id).desc())
        )
        result = await self._db.execute(stmt)
        rows = result.all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "user_id": str(r[0]),
                    "email": str(r[1]),
                    "total_predictions": int(r[2]),
                    "positive_count": int(r[3]),
                    "neutral_count": int(r[4]),
                    "negative_count": int(r[5]),
                    "avg_confidence": round(float(r[6]), 4) if r[6] else 0.0,
                    "avg_latency_ms": round(float(r[7]), 2) if r[7] else 0.0,
                    "last_active": str(r[8]) if r[8] else None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Hourly distribution (for heatmap)
    # ------------------------------------------------------------------

    async def hourly_distribution(
        self,
        days: int = 30,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        hour_col = func.strftime("%H", Prediction.created_at)
        dow_col = func.strftime("%w", Prediction.created_at)
        stmt = (
            select(
                dow_col.label("dow"),
                hour_col.label("hour"),
                func.count(Prediction.id).label("count"),
            )
            .where(Prediction.created_at >= cutoff)
            .group_by(dow_col, hour_col)
            .order_by(dow_col, hour_col)
        )
        if user_id:
            stmt = stmt.where(Prediction.user_id == user_id)
        result = await self._db.execute(stmt)
        rows = result.all()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "day_of_week": int(r[0]),
                    "hour": int(r[1]),
                    "count": int(r[2]),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Database size (SQLite)
    # ------------------------------------------------------------------

    async def db_size_mb(self) -> float:
        try:
            db_path = "./sentiment.db"
            if os.path.exists(db_path):
                return round(os.path.getsize(db_path) / (1024 * 1024), 2)
        except OSError:
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Total user count
    # ------------------------------------------------------------------

    async def total_user_count(self) -> int:
        stmt = select(func.count(User.id))
        result = await self._db.execute(stmt)
        val = result.scalar()
        return int(val) if val else 0
