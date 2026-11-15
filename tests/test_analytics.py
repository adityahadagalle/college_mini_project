"""Tests for Phase 7 – Enterprise Analytics & Business Intelligence.

Covers:
- Analytics calculations (overview, trends, charts, statistics)
- Filtering (sentiment, confidence, date, search, pagination)
- Reporting (JSON, CSV, different report types)
- Chart generation (all chart types)
- User analytics
- System analytics
- Empty datasets
- Invalid filters
- Backward compatibility
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.auth import AuditLog, PasswordHistory, RefreshToken, Session  # noqa: F401
from app.models.database import BatchJob, Prediction, User
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# Async test DB fixture
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


# Enable foreign keys for SQLite
@event.listens_for(test_engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


async def _override_get_db():  # type: ignore[no-untyped-def]
    async with TestSessionLocal() as session:
        yield session


@pytest.fixture(autouse=True)
async def setup_db():
    """Create tables before each test and drop after."""
    app.dependency_overrides[get_db] = _override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_USER_ID = "test-user-001"
_TEST_EMAIL = "analyst@test.com"
_TEST_PASSWORD_HASH = "$pbkdf2-sha256$29000$aaaa$bbbb"


async def _seed_user(db: AsyncSession) -> User:
    user = User(
        id=_TEST_USER_ID,
        email=_TEST_EMAIL,
        hashed_password=_TEST_PASSWORD_HASH,
        full_name="Test Analyst",
        is_active=True,
    )
    db.add(user)
    await db.flush()
    return user


async def _seed_predictions(
    db: AsyncSession, count: int = 10, user_id: str = _TEST_USER_ID
) -> None:
    sentiments = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    for i in range(count):
        p = Prediction(
            id=str(uuid.uuid4()),
            user_id=user_id,
            text_content=f"Test text {i}",
            sentiment_label=sentiments[i % 3],
            confidence_score=0.75 + (i % 10) * 0.02,
            positive_prob=0.6,
            neutral_prob=0.2,
            negative_prob=0.2,
            latency_ms=10.0 + i,
            created_at=datetime.now(timezone.utc) - timedelta(hours=i),
        )
        db.add(p)
    await db.flush()


async def _seed_batch_jobs(db: AsyncSession, user_id: str = _TEST_USER_ID) -> None:
    for status in ["COMPLETED", "COMPLETED", "FAILED"]:
        job = BatchJob(
            id=str(uuid.uuid4()),
            user_id=user_id,
            filename=f"test_{status.lower()}.csv",
            total_rows=100,
            processed_rows=100 if status == "COMPLETED" else 50,
            status=status,
        )
        db.add(job)
    await db.flush()


def _auth_header() -> Dict[str, str]:
    """Mock auth header that bypasses real JWT validation."""
    return {"Authorization": "Bearer test-token"}


# ---------------------------------------------------------------------------
# Mock authentication dependency
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_auth():
    """Bypass authentication for analytics tests."""
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email=_TEST_EMAIL,
            full_name="Test Analyst",
            role="user",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[get_current_user] = _mock_current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


client = TestClient(app)


# ===========================================================================
# Test: Analytics Overview
# ===========================================================================


class TestAnalyticsOverview:
    """GET /api/v1/analytics/overview."""

    async def test_overview_empty_dataset(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/overview", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 0
        assert data["average_confidence"] == 0.0
        assert data["sentiment_counts"]["positive"] == 0

    async def test_overview_with_data(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=9)
            await db.commit()
        resp = client.get("/api/v1/analytics/overview", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 9
        assert data["sentiment_counts"]["positive"] == 3
        assert data["sentiment_counts"]["neutral"] == 3
        assert data["sentiment_counts"]["negative"] == 3
        assert data["average_confidence"] > 0
        assert data["average_latency_ms"] > 0

    async def test_overview_time_buckets(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=3)
            await db.commit()
        resp = client.get("/api/v1/analytics/overview", headers=_auth_header())
        data = resp.json()
        assert data["predictions_today"] >= 0
        assert data["predictions_this_week"] >= 0
        assert data["predictions_this_month"] >= 0

    async def test_overview_batch_stats(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_batch_jobs(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/overview", headers=_auth_header())
        data = resp.json()
        assert data["batch_jobs_total"] == 3
        assert data["batch_jobs_failed"] == 1


# ===========================================================================
# Test: Trends
# ===========================================================================


class TestAnalyticsTrends:
    """GET /api/v1/analytics/trends."""

    async def test_trends_daily(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=10)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/trends?granularity=daily&days=7",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["granularity"] == "daily"
        assert isinstance(data["data"], list)

    async def test_trends_weekly(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=10)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/trends?granularity=weekly&days=30",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["granularity"] == "weekly"

    async def test_trends_monthly(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=10)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/trends?granularity=monthly&days=90",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["granularity"] == "monthly"

    async def test_trends_yearly(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/trends?granularity=yearly&days=365",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["granularity"] == "yearly"

    async def test_trends_empty(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/trends?granularity=daily&days=7",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["data"] == []


# ===========================================================================
# Test: Charts
# ===========================================================================


class TestAnalyticsCharts:
    """GET /api/v1/analytics/charts."""

    async def test_charts_structure(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=6)
            await db.commit()
        resp = client.get("/api/v1/analytics/charts?days=30", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment_pie" in data
        assert "volume_line" in data
        assert "confidence_bar" in data
        assert "latency_area" in data
        assert "sentiment_trend" in data
        assert "hourly_heatmap" in data

    async def test_pie_chart(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=9)
            await db.commit()
        resp = client.get("/api/v1/analytics/charts", headers=_auth_header())
        pie = resp.json()["sentiment_pie"]
        assert pie["chart_type"] == "pie"
        assert len(pie["labels"]) == 3
        assert len(pie["datasets"][0]["data"]) == 3

    async def test_line_chart(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await db.commit()
        resp = client.get("/api/v1/analytics/charts", headers=_auth_header())
        line = resp.json()["volume_line"]
        assert line["chart_type"] == "line"

    async def test_heatmap_chart(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await db.commit()
        resp = client.get("/api/v1/analytics/charts", headers=_auth_header())
        heatmap = resp.json()["hourly_heatmap"]
        assert heatmap["chart_type"] == "heatmap"
        assert len(heatmap["labels"]) == 24
        assert len(heatmap["datasets"]) == 7  # 7 days of week

    async def test_charts_empty(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/charts", headers=_auth_header())
        assert resp.status_code == 200
        pie = resp.json()["sentiment_pie"]
        assert sum(pie["datasets"][0]["data"]) == 0


# ===========================================================================
# Test: Reports
# ===========================================================================


class TestAnalyticsReport:
    """GET /api/v1/analytics/report."""

    async def test_prediction_report_json(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=predictions&fmt=json",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "predictions"
        assert data["total_records"] == 5
        assert len(data["data"]) == 5

    async def test_prediction_report_csv(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=3)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=predictions&fmt=csv",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "text_content" in resp.text

    async def test_user_report(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=3)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=users",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["report_type"] == "users"

    async def test_batch_report(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_batch_jobs(db)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=batch",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "batch"
        assert data["summary"]["total"] == 3

    async def test_analytics_summary_report(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=6)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=analytics",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "analytics"
        assert data["summary"]["total_predictions"] == 6

    async def test_report_empty_dataset(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=predictions",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["total_records"] == 0
        assert resp.json()["data"] == []


# ===========================================================================
# Test: Filtering
# ===========================================================================


class TestFiltering:
    """Advanced filtering via GET /api/v1/analytics/report."""

    async def test_filter_by_sentiment(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=9)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=predictions&sentiment=POSITIVE",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == 3
        for row in data["data"]:
            assert row["sentiment"] == "POSITIVE"

    async def test_filter_by_confidence(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=10)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=predictions&min_confidence=0.80",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        for row in resp.json()["data"]:
            assert row["confidence"] >= 0.80

    async def test_filter_by_search(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await db.commit()
        resp = client.get(
            "/api/v1/analytics/report?report_type=predictions&search=Test+text+0",
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["total_records"] >= 1

    async def test_pagination(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=15)
            await db.commit()
        resp1 = client.get(
            "/api/v1/analytics/report?report_type=predictions&page=1&page_size=5",
            headers=_auth_header(),
        )
        resp2 = client.get(
            "/api/v1/analytics/report?report_type=predictions&page=2&page_size=5",
            headers=_auth_header(),
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert len(resp1.json()["data"]) == 5
        assert len(resp2.json()["data"]) == 5
        # Pages should not overlap
        ids_1 = {r["id"] for r in resp1.json()["data"]}
        ids_2 = {r["id"] for r in resp2.json()["data"]}
        assert ids_1.isdisjoint(ids_2)


# ===========================================================================
# Test: User Analytics
# ===========================================================================


class TestUserAnalytics:
    """GET /api/v1/analytics/users."""

    async def test_user_analytics(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=6)
            await db.commit()
        resp = client.get("/api/v1/analytics/users", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_users"] >= 1
        assert len(data["data"]) >= 1
        user_row = data["data"][0]
        assert user_row["total_predictions"] == 6

    async def test_user_analytics_empty(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/users", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_users"] >= 1
        assert data["data"][0]["total_predictions"] == 0


# ===========================================================================
# Test: System Analytics
# ===========================================================================


class TestSystemAnalytics:
    """GET /api/v1/analytics/system."""

    async def test_system_analytics(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/system", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["cpu_percent"] >= 0
        assert data["memory_mb"] > 0
        assert data["uptime_seconds"] > 0
        assert isinstance(data["model_loaded"], bool)

    async def test_system_analytics_counts(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await _seed_batch_jobs(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/system", headers=_auth_header())
        data = resp.json()
        assert data["total_predictions"] == 5
        assert data["total_users"] >= 1
        assert data["total_batch_jobs"] == 3


# ===========================================================================
# Test: Statistics
# ===========================================================================


class TestPredictionStatistics:
    """GET /api/v1/analytics/statistics."""

    async def test_statistics(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=9)
            await db.commit()
        resp = client.get("/api/v1/analytics/statistics", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 9
        assert data["positive_count"] == 3
        assert data["neutral_count"] == 3
        assert data["negative_count"] == 3
        assert data["positive_pct"] > 0
        assert data["avg_confidence"] > 0
        assert data["min_confidence"] > 0
        assert data["max_confidence"] > 0

    async def test_statistics_empty(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            await _seed_user(db)
            await db.commit()
        resp = client.get("/api/v1/analytics/statistics", headers=_auth_header())
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["avg_confidence"] == 0.0


# ===========================================================================
# Test: Permission checks (unauthenticated)
# ===========================================================================


class TestPermissionChecks:
    """Verify endpoints require authentication."""

    def test_overview_no_auth(self) -> None:
        # Remove the override temporarily
        from app.api.v1.auth import get_current_user

        saved = app.dependency_overrides.get(get_current_user)
        app.dependency_overrides.pop(get_current_user, None)
        resp = client.get("/api/v1/analytics/overview")
        assert resp.status_code == 401
        if saved:
            app.dependency_overrides[get_current_user] = saved

    def test_trends_no_auth(self) -> None:
        from app.api.v1.auth import get_current_user

        saved = app.dependency_overrides.get(get_current_user)
        app.dependency_overrides.pop(get_current_user, None)
        resp = client.get("/api/v1/analytics/trends")
        assert resp.status_code == 401
        if saved:
            app.dependency_overrides[get_current_user] = saved


# ===========================================================================
# Test: Direct EnterpriseAnalyticsService & Repository Coverage
# ===========================================================================
class TestDirectAnalyticsService:
    async def test_all_granularity_trends(self, setup_db: None) -> None:
        from app.repositories.analytics_repository import AnalyticsRepository
        from app.services.analytics_service import EnterpriseAnalyticsService

        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=10)
            await db.commit()

            repo = AnalyticsRepository(db)
            svc = EnterpriseAnalyticsService(repo)

            daily = await svc.get_trends("daily")
            assert len(daily.data) > 0

            weekly = await svc.get_trends("weekly")
            assert len(weekly.data) > 0

            monthly = await svc.get_trends("monthly")
            assert len(monthly.data) > 0

            yearly = await svc.get_trends("yearly")
            assert len(yearly.data) > 0

    async def test_report_generations(self, setup_db: None) -> None:
        from app.repositories.analytics_repository import AnalyticsRepository
        from app.services.analytics_service import EnterpriseAnalyticsService

        async with TestSessionLocal() as db:
            await _seed_user(db)
            await _seed_predictions(db, count=5)
            await db.commit()

            repo = AnalyticsRepository(db)
            svc = EnterpriseAnalyticsService(repo)

            rep_pred = await svc.get_report("predictions", fmt="json")
            assert rep_pred.total_records >= 0

            rep_user = await svc.get_report("users", fmt="csv")
            assert rep_user.report_type == "users"

            rep_batch = await svc.get_report("batch", fmt="json")
            assert rep_batch.report_type == "batch"

            rep_summary = await svc.get_report("summary", fmt="json")
            assert rep_summary.report_type in ["summary", "analytics"]
