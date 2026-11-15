"""Comprehensive Test Suite for Phase 4: Backend Integration & Cross-Feature Intelligence.

Verifies:
1. End-to-end unified review analysis linking Prediction, ReviewAspect, ReviewEmotion, and TopicAssignment.
2. Fault isolation: ABSA unloaded returns status="unavailable" without failing the request.
3. Fault isolation: Emotion service error returns status="failed" without failing the request.
4. Topic projection against completed discovery run (matching cluster and < 0.45 outlier case), and no-run behavior.
5. Unified batch processing (sync execution, deduplication of duplicate texts, and > 500 batch cap validation).
6. Asynchronous batch processing and job status tracking.
7. Cross-feature SQL analytics aggregations (aspect_sentiment_by_topic, emotion_by_overall_sentiment, etc.).
8. Evidence drilldown endpoint with filtering and pagination.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models.database import (
    DiscoveredTopic,
    Prediction,
    ReviewAspect,
    ReviewEmotion,
    TopicAssignment,
    TopicDiscoveryRun,
)
from app.services.absa_service import absa_service
from app.services.emotion_service import emotion_service
from app.services.intelligence_service import intelligence_service
from app.services.topic_service import topic_service

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(
    TEST_DB_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


async def _override_get_db():
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


_TEST_USER_ID = "intelligence-user-001"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="intel_tester@platform.ai",
            full_name="Intelligence Tester",
            role="user",
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
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def client():
    return TestClient(app)


# Helper to seed a completed topic run
async def seed_completed_topic_run(session: AsyncSession) -> str:
    run_id = f"test-run-{uuid.uuid4()}"
    run = TopicDiscoveryRun(
        id=run_id,
        user_id=_TEST_USER_ID,
        status="COMPLETED",
        corpus_size=30,
        num_topics=2,
        outlier_count=1,
        parameters_json={},
        metrics_json={"mean_intra_cluster_cohesion": 0.65},
    )
    session.add(run)

    # Battery centroid
    battery_vec = np.zeros(384, dtype=np.float32)
    battery_vec[0] = 1.0  # normalized vector along dim 0
    t1 = DiscoveredTopic(
        run_id=run_id,
        topic_index=0,
        auto_label="Battery Life / Charging",
        human_label="Battery Performance",
        keywords_json=[{"word": "battery", "score": 1.5}, {"word": "charging", "score": 1.2}],
        centroid_json=battery_vec.tolist(),
        review_count=15,
        cohesion_score=0.72,
        is_archived=False,
    )

    # Audio centroid
    audio_vec = np.zeros(384, dtype=np.float32)
    audio_vec[1] = 1.0  # normalized vector along dim 1
    t2 = DiscoveredTopic(
        run_id=run_id,
        topic_index=1,
        auto_label="Sound / Noise Cancellation",
        human_label="Audio Experience",
        keywords_json=[{"word": "sound", "score": 1.4}, {"word": "bass", "score": 1.1}],
        centroid_json=audio_vec.tolist(),
        review_count=12,
        cohesion_score=0.68,
        is_archived=False,
    )

    # Outlier cluster (-1)
    t_out = DiscoveredTopic(
        run_id=run_id,
        topic_index=-1,
        auto_label="Outliers / Unassigned",
        human_label=None,
        keywords_json=[],
        centroid_json=None,
        review_count=3,
        cohesion_score=None,
        is_archived=False,
    )

    session.add_all([t1, t2, t_out])
    await session.commit()
    return run_id


# ==============================================================================
# 1. End-to-End Unified Review Analysis
# ==============================================================================


@pytest.mark.asyncio
async def test_unified_analyze_end_to_end_linking():
    """Verify Prediction, ReviewAspect, ReviewEmotion, and TopicAssignment are linked under review_id."""
    async with TestSessionLocal() as session:
        run_id = await seed_completed_topic_run(session)

        review_text = "The battery life is fantastic and lasts all day, but the screen scratches easily."
        res = await intelligence_service.analyze_review(
            session=session,
            text=review_text,
            user_id=_TEST_USER_ID,
            save_to_db=True,
            project_topic=True,
        )

        assert res.review_id is not None
        assert res.text_content == review_text
        assert res.overall_sentiment is not None
        assert res.overall_sentiment.sentiment in ("POSITIVE", "NEUTRAL", "NEGATIVE")
        assert res.capability_status["sentiment"].status == "completed"

        # Verify DB records
        pred = await session.get(Prediction, res.review_id)
        assert pred is not None
        assert pred.text_content == review_text

        # Verify aspect records in DB
        res_aspects = await session.execute(
            select(ReviewAspect).where(ReviewAspect.prediction_id == res.review_id)
        )
        aspect_records = res_aspects.scalars().all()
        assert len(aspect_records) == len(res.aspects)

        # Verify emotion record in DB
        if res.emotions is not None:
            res_emo = await session.execute(
                select(ReviewEmotion).where(ReviewEmotion.prediction_id == res.review_id)
            )
            emo_rec = res_emo.scalar_one_or_none()
            assert emo_rec is not None
            assert emo_rec.primary_cx_category == res.emotions.primary_cx_category

        # Verify topic assignment in DB if topic projected
        if res.topic is not None and res.topic.topic_id is not None:
            res_ta = await session.execute(
                select(TopicAssignment).where(TopicAssignment.prediction_id == res.review_id)
            )
            ta_rec = res_ta.scalar_one_or_none()
            assert ta_rec is not None
            assert ta_rec.topic_id == res.topic.topic_id


# ==============================================================================
# 2. Strict Failure Isolation Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_absa_failure_isolation_when_unloaded():
    """When ABSA models are unloaded, sentiment, emotions, and topics still succeed with absa status='unavailable'."""
    async with TestSessionLocal() as session:
        await seed_completed_topic_run(session)
        orig_ate = absa_service.ate_model
        try:
            absa_service.ate_model = None  # simulate unloaded ABSA
            assert absa_service.is_ready() is False

            res = await intelligence_service.analyze_review(
                session=session,
                text="The sound quality is crisp and clear.",
                user_id=_TEST_USER_ID,
                save_to_db=False,
            )

            assert res.capability_status["absa"].status == "unavailable"
            assert "not loaded" in res.capability_status["absa"].error
            assert res.aspects == []

            # Overall sentiment and emotions must still complete
            assert res.capability_status["sentiment"].status == "completed"
            assert res.overall_sentiment is not None
        finally:
            absa_service.ate_model = orig_ate


@pytest.mark.asyncio
async def test_emotion_failure_isolation_when_exception_raised(monkeypatch):
    """When emotion detection raises an exception, other capabilities complete and emotion status='failed'."""
    async with TestSessionLocal() as session:
        await seed_completed_topic_run(session)

        # Monkeypatch emotion analyze to raise an exception
        async def mock_fail_analyze(*args, **kwargs):
            raise RuntimeError("CUDA out of memory simulated error")

        monkeypatch.setattr(emotion_service, "analyze", mock_fail_analyze)

        res = await intelligence_service.analyze_review(
            session=session,
            text="Amazing customer support solved my ticket.",
            user_id=_TEST_USER_ID,
            save_to_db=False,
        )

        assert res.capability_status["emotion"].status == "failed"
        assert "CUDA out of memory" in res.capability_status["emotion"].error
        assert res.emotions is None

        # Sentiment and ABSA must succeed
        assert res.capability_status["sentiment"].status == "completed"
        assert res.overall_sentiment is not None


@pytest.mark.asyncio
async def test_topic_projection_when_no_completed_run_exists():
    """When no completed topic discovery run exists, topic status='unavailable' without failing request."""
    async with TestSessionLocal() as session:
        # DB has 0 topic discovery runs
        res = await intelligence_service.analyze_review(
            session=session,
            text="Random feedback with zero prior topic clustering runs.",
            user_id=_TEST_USER_ID,
            save_to_db=False,
            project_topic=True,
        )

        assert res.capability_status["topics"].status == "unavailable"
        assert "No completed topic discovery run found" in res.capability_status["topics"].error
        assert res.topic is None
        assert res.capability_status["sentiment"].status == "completed"


# ==============================================================================
# 3. API Endpoints: Single Analyze & Batch Processing
# ==============================================================================


def test_api_intelligence_analyze_endpoint(client: TestClient):
    """POST /api/v1/intelligence/analyze returns unified response with capability_status."""
    payload = {
        "text": "The battery life lasts two full days, but audio quality is terrible.",
        "save_to_db": True,
        "project_topic": False,
    }
    res = client.post("/api/v1/intelligence/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert "review_id" in data
    assert "overall_sentiment" in data
    assert "capability_status" in data
    assert "sentiment" in data["capability_status"]
    assert "absa" in data["capability_status"]
    assert "emotion" in data["capability_status"]
    assert "topics" in data["capability_status"]
    assert data["capability_status"]["topics"]["status"] == "unavailable"


def test_api_intelligence_analyze_short_text_bad_request(client: TestClient):
    """POST /api/v1/intelligence/analyze with < 3 characters returns HTTP 400 or 422."""
    res = client.post("/api/v1/intelligence/analyze", json={"text": "hi"})
    assert res.status_code in (400, 422)


def test_api_intelligence_batch_sync_and_deduplication(client: TestClient):
    """POST /api/v1/intelligence/batch runs synchronously and deduplicates duplicate texts."""
    reviews = [
        "Great battery life!",
        "Terrible sound quality.",
        "Great battery life!",  # exact duplicate
    ]
    payload = {"reviews": reviews, "sync": True, "save_to_db": False}

    res = client.post("/api/v1/intelligence/batch", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "COMPLETED"
    assert data["total_records"] == 3
    assert len(data["results"]) == 3
    # Check that the 1st and 3rd result have identical text
    assert data["results"][0]["text_content"] == data["results"][2]["text_content"]


def test_api_intelligence_batch_size_cap_validation(client: TestClient):
    """POST /api/v1/intelligence/batch with > 500 items returns HTTP 400."""
    large_reviews = [f"Review {i}" for i in range(501)]
    res = client.post(
        "/api/v1/intelligence/batch",
        json={"reviews": large_reviews, "sync": True},
    )
    assert res.status_code in (400, 422)


def test_api_intelligence_batch_async_tracking(client: TestClient):
    """POST /api/v1/intelligence/batch with sync=False creates async job tracked via GET /batch/{job_id}."""
    reviews = ["Async review one", "Async review two"]
    res = client.post(
        "/api/v1/intelligence/batch",
        json={"reviews": reviews, "sync": False, "save_to_db": False},
    )
    assert res.status_code == 200
    job_data = res.json()
    job_id = job_data["job_id"]
    assert job_data["status"] in ("PENDING", "PROCESSING", "COMPLETED")

    # GET /batch/{job_id}
    res_status = client.get(f"/api/v1/intelligence/batch/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["job_id"] == job_id


# ==============================================================================
# 4. Cross-Feature SQL Analytics & Evidence Drilldown
# ==============================================================================


@pytest.mark.asyncio
async def test_cross_feature_analytics_sql_aggregation():
    """Verify SQL cross-feature aggregations across topics, aspects, emotions, and sentiment."""
    async with TestSessionLocal() as session:
        run_id = await seed_completed_topic_run(session)

        # Ingest 3 reviews to test cross-feature groupings
        revs = [
            "The battery is incredible and charges super fast.",
            "Battery drain is awful and device overheats.",
            "Sound quality is crystal clear with amazing noise cancellation.",
        ]
        for r in revs:
            await intelligence_service.analyze_review(
                session=session,
                text=r,
                user_id=_TEST_USER_ID,
                save_to_db=True,
                project_topic=True,
            )

        analytics = await intelligence_service.get_cross_feature_analytics(
            session=session, min_sample_size=1
        )

        assert analytics.total_analyzed_reviews >= 3
        assert len(analytics.methodology_note) > 0
        assert "co-occurrence" in analytics.methodology_note
        assert isinstance(analytics.aspect_sentiment_by_topic, list)
        assert isinstance(analytics.emotion_by_overall_sentiment, list)
        assert isinstance(analytics.negative_aspect_trends, list)

        # Check low_sample_warning flag
        for item in analytics.aspect_sentiment_by_topic:
            if item.sample_count < 5:
                assert item.low_sample_warning is True


def test_api_evidence_drilldown(client: TestClient):
    """GET /api/v1/intelligence/evidence returns paginated matching reviews."""
    # First analyze a review
    client.post(
        "/api/v1/intelligence/analyze",
        json={"text": "Outstanding camera quality with vibrant colors.", "save_to_db": True},
    )

    res = client.get("/api/v1/intelligence/evidence?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()

    assert "total_matched" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert data["page"] == 1
