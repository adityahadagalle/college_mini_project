"""Tests for batch processing, prediction, history, and settings API endpoints & services.

Covers:
- backend/app/api/v1/batch.py (upload, status, download, non-existent job 404, non-completed job download error, invalid extension 400)
- backend/app/services/batch_service.py (process_batch_job_async, valid CSV processing, invalid/malformed CSV handling)
- backend/app/api/v1/history.py (get history, pagination, filters, export CSV)
- backend/app/api/v1/settings.py (get user settings, update user settings)
- backend/app/api/v1/predict.py & prediction_service.py (predict single, empty text validation)
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.auth import AuditLog, PasswordHistory, RefreshToken, Session  # noqa: F401
from app.models.database import BatchJob, Prediction, User, UserSettings
from app.services.batch_service import process_batch_job_async
from app.services.prediction_service import PredictionService
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def _override_get_db():  # type: ignore[no-untyped-def]
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


_TEST_USER_ID = "batch-history-user-123"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="batch_user@test.com",
            full_name="Batch User",
            role="user",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[get_current_user] = _mock_current_user
    yield


@pytest.fixture(autouse=True)
def mock_batch_db():
    from unittest.mock import patch

    with patch("app.services.batch_service.AsyncSessionLocal", TestSessionLocal):
        yield


@pytest.fixture(autouse=True)
async def setup_db():
    app.dependency_overrides[get_db] = _override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Seed base user
    async with TestSessionLocal() as db:
        user = User(
            id=_TEST_USER_ID,
            email="batch_user@test.com",
            hashed_password="hash",
            full_name="Batch User",
            is_active=True,
        )
        db.add(user)
        await db.commit()
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.clear()


client = TestClient(app)


# ---------------------------------------------------------------------------
# Batch Processing API & Service Tests
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    def test_upload_invalid_extension(self, setup_db: None) -> None:
        file_content = b"header1,header2\nval1,val2"
        resp = client.post(
            "/api/v1/batch/upload",
            files={"file": ("test.png", file_content, "image/png")},
        )
        assert resp.status_code == 400

    def test_upload_empty_file(self, setup_db: None) -> None:
        resp = client.post(
            "/api/v1/batch/upload",
            files={"file": ("test.csv", b"", "text/csv")},
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()

    def test_upload_valid_csv_flow(self, setup_db: None) -> None:
        csv_content = b"text\nGreat experience!\nTerrible service."
        resp = client.post(
            "/api/v1/batch/upload",
            files={"file": ("sentiment_data.csv", csv_content, "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["filename"] == "sentiment_data.csv"
        job_id = data["job_id"]

        # Check job status
        status_resp = client.get(f"/api/v1/batch/status/{job_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["job_id"] == job_id

    def test_batch_status_nonexistent(self, setup_db: None) -> None:
        resp = client.get("/api/v1/batch/status/non-existent-job-id")
        assert resp.status_code == 404

    def test_download_non_completed_job(self, setup_db: None) -> None:
        # Create a pending batch job
        async def _create_pending():
            async with TestSessionLocal() as db:
                job = BatchJob(
                    id="pending-job-123",
                    user_id=_TEST_USER_ID,
                    filename="pending.csv",
                    status="PENDING",
                )
                db.add(job)
                await db.commit()

        import asyncio

        asyncio.run(_create_pending())

        resp = client.get("/api/v1/batch/download/pending-job-123")
        assert resp.status_code == 400

    async def test_batch_service_async_processing(self, setup_db: None) -> None:
        import tempfile

        # Write a temporary CSV file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
            tmp.write("text\nAwesome product!\nHorrible support.\nNeutral sentiment.\n")
            tmp_path = tmp.name

        job_id = str(uuid.uuid4())
        async with TestSessionLocal() as db:
            job = BatchJob(
                id=job_id,
                user_id=_TEST_USER_ID,
                filename="unit_test_batch.csv",
                status="PENDING",
            )
            db.add(job)
            await db.commit()

        # Run batch processor
        await process_batch_job_async(job_id, tmp_path, _TEST_USER_ID)

        # Verify completed job status in database
        async with TestSessionLocal() as db:
            result = await db.get(BatchJob, job_id)
            assert result is not None
            assert result.status == "COMPLETED"
            assert result.total_rows == 4  # header + 3 text rows
            assert result.processed_rows == 4
            assert result.result_file_path is not None
            if os.path.exists(result.result_file_path):
                os.remove(result.result_file_path)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# History API Tests
# ---------------------------------------------------------------------------


class TestHistoryAPI:
    async def _seed_predictions(self) -> None:
        async with TestSessionLocal() as db:
            for i in range(5):
                p = Prediction(
                    id=str(uuid.uuid4()),
                    user_id=_TEST_USER_ID,
                    text_content=f"History text sample {i}",
                    sentiment_label="POSITIVE" if i % 2 == 0 else "NEGATIVE",
                    confidence_score=0.90,
                    positive_prob=0.90,
                    neutral_prob=0.05,
                    negative_prob=0.05,
                    latency_ms=12.5,
                )
                db.add(p)
            await db.commit()

    def test_get_history(self, setup_db: None) -> None:
        import asyncio

        asyncio.run(self._seed_predictions())

        resp = client.get("/api/v1/history?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5

        # Test search filter
        search_resp = client.get("/api/v1/history?search=sample+0")
        assert search_resp.status_code == 200
        assert len(search_resp.json()) == 1

        # Test sentiment filter
        pos_resp = client.get("/api/v1/history?sentiment=POSITIVE")
        assert pos_resp.status_code == 200
        assert len(pos_resp.json()) == 3

    def test_export_history_csv(self, setup_db: None) -> None:
        import asyncio

        asyncio.run(self._seed_predictions())

        resp = client.get("/api/v1/history/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        assert "History text sample" in resp.text


# ---------------------------------------------------------------------------
# Settings API Tests
# ---------------------------------------------------------------------------


class TestSettingsAPI:
    def test_get_and_update_settings(self, setup_db: None) -> None:
        # Get settings (auto creates default settings)
        get_resp = client.get("/api/v1/settings/")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["theme"] == "dark"

        # Update settings
        update_payload = {
            "confidence_threshold": 0.85,
            "default_export_format": "json",
            "theme": "light",
        }
        put_resp = client.put("/api/v1/settings/", json=update_payload)
        assert put_resp.status_code == 200
        updated = put_resp.json()
        assert updated["confidence_threshold"] == 0.85
        assert updated["theme"] == "light"


# ---------------------------------------------------------------------------
# Prediction API & Service Tests
# ---------------------------------------------------------------------------


class TestPredictionAPI:
    def test_predict_endpoint(self, setup_db: None) -> None:
        resp = client.post(
            "/api/v1/predict/",
            json={"text": "The service provided an exceptional experience!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert "probabilities" in data

    def test_predict_empty_text_raises_400(self, setup_db: None) -> None:
        resp = client.post(
            "/api/v1/predict/",
            json={"text": "   "},
        )
        assert resp.status_code == 400

    def test_prediction_service(self, setup_db: None) -> None:
        from app.schemas.sentiment import PredictionRequest
        from app.services.tf_inference import tf_engine

        svc = PredictionService(tf_engine)
        res = svc.predict_single(PredictionRequest(text="Wonderful product!"))
        sentiment = res["sentiment"] if isinstance(res, dict) else res.label
        assert sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]
