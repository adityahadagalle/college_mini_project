"""Targeted tests for PredictionService file predictions, Batch API routes, History CSV export, Settings API, and Database core utilities."""

import io
from datetime import datetime, timezone
from typing import Any

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.database import BatchJob, Prediction, User, UserSettings
from app.schemas.sentiment import BatchPredictionRequest, PredictionRequest
from app.services.prediction_service import PredictionService
from app.services.tf_inference import tf_engine
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def _override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


_TEST_USER_ID = "pred-core-user-999"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="pred_core@test.com",
            full_name="Pred Core User",
            role="user",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[get_current_user] = _mock_current_user
    yield


@pytest.fixture(autouse=True)
async def setup_db():
    app.dependency_overrides[get_db] = _override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with TestSessionLocal() as db:
        user = User(
            id=_TEST_USER_ID,
            email="pred_core@test.com",
            hashed_password="hash",
            full_name="Pred Core User",
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
# PredictionService Unit Tests
# ---------------------------------------------------------------------------


class TestPredictionServiceDetails:
    def test_predict_batch(self) -> None:
        svc = PredictionService(tf_engine)

        # Single prediction
        single_res = svc.predict_single(PredictionRequest(text="Excellent service!"))
        sentiment = single_res["sentiment"] if isinstance(single_res, dict) else single_res.label
        assert sentiment in ["POSITIVE", "NEUTRAL", "NEGATIVE"]

        # Batch prediction
        batch_req = BatchPredictionRequest(texts=["Great product!", "Terrible service."])
        batch_res = svc.predict_batch(batch_req)
        assert len(batch_res) == 2

    async def test_predict_file_csv_and_json(self) -> None:
        svc = PredictionService(tf_engine)

        # CSV upload file
        csv_data = b"text\nWonderful app!\nWorst performance."
        upload_csv = UploadFile(io.BytesIO(csv_data), filename="sample.csv")
        res_csv = await svc.predict_file(upload_csv, text_column="text")
        len_csv = len(res_csv.predictions) if hasattr(res_csv, "predictions") else len(res_csv)
        assert len_csv == 2

        # JSON upload file
        json_data = b'[{"text": "Love it!"}, {"text": "Hate it!"}]'
        upload_json = UploadFile(io.BytesIO(json_data), filename="sample.json")
        res_json = await svc.predict_file(upload_json, text_column="text")
        len_json = len(res_json.predictions) if hasattr(res_json, "predictions") else len(res_json)
        assert len_json == 2

        # Empty file raises BatchPredictionError
        upload_empty = UploadFile(io.BytesIO(b""), filename="empty.pdf")
        from app.core.inference_exceptions import BatchPredictionError

        with pytest.raises(BatchPredictionError):
            await svc.predict_file(upload_empty)


# ---------------------------------------------------------------------------
# Batch & History Route Edge Cases
# ---------------------------------------------------------------------------


class TestBatchAndHistoryRoutesEdgeCases:
    def test_batch_status_404(self) -> None:
        resp = client.get("/api/v1/batch/status/nonexistent-id")
        assert resp.status_code == 404

    def test_batch_download_404_and_400(self, setup_db: None) -> None:
        # Nonexistent job 404
        resp404 = client.get("/api/v1/batch/download/nonexistent-id")
        assert resp404.status_code == 404

        # Job not completed 400
        async def _seed_job():
            async with TestSessionLocal() as db:
                job = BatchJob(
                    id="job-pending-1",
                    user_id=_TEST_USER_ID,
                    filename="test.csv",
                    total_rows=5,
                    processed_rows=0,
                    status="PENDING",
                )
                db.add(job)
                await db.commit()

        import asyncio

        asyncio.run(_seed_job())

        resp400 = client.get("/api/v1/batch/download/job-pending-1")
        assert resp400.status_code == 400

    def test_history_export_csv_and_get(self, setup_db: None) -> None:
        async def _seed_history():
            async with TestSessionLocal() as db:
                p = Prediction(
                    id="p-100",
                    user_id=_TEST_USER_ID,
                    text_content="Export text sample",
                    sentiment_label="POSITIVE",
                    confidence_score=0.95,
                    positive_prob=0.95,
                    neutral_prob=0.03,
                    negative_prob=0.02,
                    latency_ms=12.5,
                )
                db.add(p)
                await db.commit()

        import asyncio

        asyncio.run(_seed_history())

        # GET history with search & filter
        hist_resp = client.get("/api/v1/history/?search=Export&sentiment=POSITIVE")
        assert hist_resp.status_code == 200
        assert len(hist_resp.json()) == 1

        # CSV Export
        resp = client.get("/api/v1/history/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "Export text sample" in resp.text


# ---------------------------------------------------------------------------
# Settings API & Database Utility Tests
# ---------------------------------------------------------------------------


class TestSettingsAndDatabaseCore:
    def test_settings_get_and_put(self, setup_db: None) -> None:
        get_resp = client.get("/api/v1/settings/")
        assert get_resp.status_code == 200
        assert get_resp.json()["theme"] == "dark"

        put_resp = client.put(
            "/api/v1/settings/",
            json={"confidence_threshold": 0.88, "default_export_format": "csv", "theme": "light"},
        )
        assert put_resp.status_code == 200
        assert put_resp.json()["theme"] == "light"
        assert put_resp.json()["confidence_threshold"] == 0.88

    async def test_get_db_exception_rollback(self) -> None:
        from app.core.database import get_db

        db_gen = get_db()
        session = await anext(db_gen)
        assert session is not None

        with pytest.raises(RuntimeError):
            await db_gen.athrow(RuntimeError("Test DB rollback"))
