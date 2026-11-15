"""High-yielding coverage tests for API routes, Analytics Engine, Auth exception handlers, and Core Exceptions."""

import io
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.database import BatchJob, Prediction, User, UserSettings
from app.repositories.analytics_repository import AnalyticsRepository
from app.services.analytics_service import AnalyticsFilter, EnterpriseAnalyticsService
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def _override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


_TEST_USER_ID = "route-user-9999"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="route_user@test.com",
            full_name="Route User",
            role="admin",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[get_current_user] = _mock_current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
async def setup_db():
    app.dependency_overrides[get_db] = _override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with TestSessionLocal() as db:
        user = User(
            id=_TEST_USER_ID,
            email="route_user@test.com",
            hashed_password="hash",
            full_name="Route User",
            is_active=True,
        )
        db.add(user)
        await db.commit()
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.pop(get_db, None)


client = TestClient(app)


# ---------------------------------------------------------------------------
# API Routes Coverage
# ---------------------------------------------------------------------------


class TestApiRoutesFullCoverage:
    def test_settings_get_and_put_no_slash(self, setup_db: None) -> None:
        get_resp = client.get("/api/v1/settings/", follow_redirects=True)
        assert get_resp.status_code == 200

        put_resp = client.put(
            "/api/v1/settings/",
            json={"confidence_threshold": 0.90, "default_export_format": "pdf", "theme": "dark"},
            follow_redirects=True,
        )
        assert put_resp.status_code == 200

    def test_batch_download_and_status_routes(self, setup_db: None) -> None:
        # Create a completed batch job
        async def _seed_completed_job():
            async with TestSessionLocal() as db:
                job = BatchJob(
                    id="job-completed-77",
                    user_id=_TEST_USER_ID,
                    filename="sample.csv",
                    total_rows=2,
                    processed_rows=2,
                    status="COMPLETED",
                    result_file_path="tests/test_batch_and_history.py",
                )
                db.add(job)
                await db.commit()

        import asyncio

        asyncio.run(_seed_completed_job())

        # Check status route
        status_resp = client.get("/api/v1/batch/status/job-completed-77")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "COMPLETED"

        # Check download route
        dl_resp = client.get("/api/v1/batch/download/job-completed-77")
        assert dl_resp.status_code == 200

    def test_history_list_and_csv_export(self, setup_db: None) -> None:
        async def _seed_pred():
            async with TestSessionLocal() as db:
                p = Prediction(
                    id="pred-hist-77",
                    user_id=_TEST_USER_ID,
                    text_content="Sample text for history export",
                    sentiment_label="NEUTRAL",
                    confidence_score=0.75,
                    positive_prob=0.1,
                    neutral_prob=0.8,
                    negative_prob=0.1,
                    latency_ms=10.0,
                )
                db.add(p)
                await db.commit()

        import asyncio

        asyncio.run(_seed_pred())

        # History list
        h_resp = client.get("/api/v1/history?limit=10&offset=0")
        assert h_resp.status_code == 200

        # History export CSV
        exp_resp = client.get("/api/v1/history/export")
        assert exp_resp.status_code == 200

    def test_auth_api_exceptions(self, setup_db: None) -> None:
        # Invalid login password (HTTP 401)
        resp_login = client.post(
            "/api/v1/auth/login",
            data={"username": "nonexistent@test.com", "password": "WrongPassword!"},
        )
        assert resp_login.status_code == 401

        # Duplicate registration email (HTTP 400)
        resp_reg = client.post(
            "/api/v1/auth/register",
            json={
                "email": "route_user@test.com",
                "password": "SecurePassword123!",
                "full_name": "Duplicate User",
            },
        )
        assert resp_reg.status_code == 400


# ---------------------------------------------------------------------------
# Analytics Service & Repository Deep Coverage
# ---------------------------------------------------------------------------


class TestAnalyticsDeepCoverage:
    async def test_analytics_repository_and_service_branches(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            repo = AnalyticsRepository(db)
            svc = EnterpriseAnalyticsService(repo)

            # Analytics filter with dates
            f = AnalyticsFilter(
                start_date=date.today() - timedelta(days=7),
                end_date=date.today(),
                sentiment="POSITIVE",
                min_confidence=0.5,
                max_confidence=1.0,
                user_id=_TEST_USER_ID,
                search="sample",
            )

            # Filtered predictions
            preds, count = await repo.filtered_predictions(
                user_id=_TEST_USER_ID,
                sentiment="POSITIVE",
                min_confidence=0.5,
                max_confidence=1.0,
                start=datetime.now(timezone.utc) - timedelta(days=7),
                end=datetime.now(timezone.utc),
                search="sample",
                page=1,
                page_size=10,
            )
            assert isinstance(preds, list)

            # Reports (predictions, users, batch, summary in JSON, CSV, PDF)
            for r_type in ["predictions", "users", "batch", "summary"]:
                for fmt in ["json", "csv", "pdf"]:
                    rep = await svc.get_report(report_type=r_type, fmt=fmt, filters=f)
                    assert rep is not None

            # Analytics user analytics & system analytics & statistics
            user_ana = await svc.get_user_analytics()
            assert user_ana is not None

            sys_ana = await svc.get_system_analytics()
            assert sys_ana is not None

            overview = await svc.get_overview()
            assert overview is not None


# ---------------------------------------------------------------------------
# Core Exception Handlers Coverage
# ---------------------------------------------------------------------------


class TestCoreExceptionsCoverage:
    def test_register_exception_handlers(self) -> None:
        from app.core.exceptions import register_exception_handlers
        from fastapi.exceptions import RequestValidationError

        test_app = FastAPI()
        register_exception_handlers(test_app)

        @test_app.get("/err-val")
        def raise_val():
            raise RequestValidationError(
                [{"loc": ["query", "page"], "msg": "invalid int", "type": "type_error"}]
            )

        @test_app.get("/err-500")
        def raise_500():
            raise RuntimeError("Unexpected error")

        c = TestClient(test_app, raise_server_exceptions=False)
        assert c.get("/err-val").status_code == 422
        assert c.get("/err-500").status_code == 500
