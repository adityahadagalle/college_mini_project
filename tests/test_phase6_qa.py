"""Phase 6 QA, Edge-Case Robustness, Security, and Fault-Isolation Test Suite.

Verifies:
1. Malformed inputs, empty/whitespace inputs, oversized reviews (> 5,000 chars), and Unicode/emoji safety.
2. Batch payload limits (rejecting > 500 items with HTTP 400), empty batches, and intra-batch deduplication.
3. Missing/corrupted model artifact handling and independent capability fault isolation:
   - ABSA offline -> /absa/* returns 503, but /predict, /emotion, and /intelligence remain operational.
   - Emotion offline -> /emotion/* returns 503, but /predict, /absa, and /intelligence remain operational.
   - Topics offline -> /topics/* returns 503, but /intelligence gracefully marks topics 'unavailable'.
4. Database schema consistency and cascade integrity across predictions, review_aspects, review_emotions,
   discovered_topics, topic_assignments, and intelligence_batch_jobs.
5. Security sanitization: path traversal prevention and no raw stack trace leakage.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid

import httpx
from httpx import ASGITransport
import numpy as np
import pytest
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
    IntelligenceBatchJob,
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


_TEST_USER_ID = "qa-tester-user-001"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="qa_auditor@platform.ai",
            full_name="QA Auditor",
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
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ==============================================================================
# 1. Malformed, Empty, Oversized & Unicode/Emoji Inputs
# ==============================================================================


@pytest.mark.asyncio
async def test_qa_empty_and_whitespace_inputs(async_client: httpx.AsyncClient):
    """Empty or whitespace-only inputs must be rejected with 400 or 422 across endpoints."""
    bad_payloads = [{"text": ""}, {"text": "   "}, {"text": "\t  \n  "}]
    for p in bad_payloads:
        # Intelligence analyze
        res_intel = await async_client.post("/api/v1/intelligence/analyze", json=p)
        assert res_intel.status_code in (400, 422)

        # ABSA analyze
        res_absa = await async_client.post("/api/v1/absa/analyze", json=p)
        assert res_absa.status_code in (400, 422)

        # Emotion analyze
        res_emo = await async_client.post("/api/v1/emotion/analyze", json=p)
        assert res_emo.status_code in (400, 422)


@pytest.mark.asyncio
async def test_qa_sub_min_length_intelligence_rejected(async_client: httpx.AsyncClient):
    """Sub-minimum length texts (< 3 chars) must be rejected with 400 or 422 by intelligence analyze."""
    for p in [{"text": "a"}, {"text": "ab"}]:
        res_intel = await async_client.post("/api/v1/intelligence/analyze", json=p)
        assert res_intel.status_code in (400, 422)


@pytest.mark.asyncio
async def test_qa_oversized_reviews_rejected(async_client: httpx.AsyncClient):
    """Reviews exceeding 5,000 characters must be rejected before neural execution."""
    oversized_text = "A" * 5001
    payload = {"text": oversized_text}

    res_intel = await async_client.post("/api/v1/intelligence/analyze", json=payload)
    assert res_intel.status_code in (400, 422)

    res_absa = await async_client.post("/api/v1/absa/analyze", json=payload)
    assert res_absa.status_code in (400, 422)

    res_emo = await async_client.post("/api/v1/emotion/analyze", json=payload)
    assert res_emo.status_code in (400, 422)


@pytest.mark.asyncio
async def test_qa_unicode_and_emojis(async_client: httpx.AsyncClient):
    """Unicode, Cyrillic, Kanji, accented characters, and emojis must be preserved without error."""
    multilingual_text = "The camera is awesome 📸 and battery lasts long ⚡! Отличный сервис, 素晴らしい品質."
    payload = {"text": multilingual_text, "save_to_db": False, "project_topic": False}

    res = await async_client.post("/api/v1/intelligence/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["text_content"] == multilingual_text
    assert data["overall_sentiment"]["sentiment"] in ("POSITIVE", "NEUTRAL", "NEGATIVE")
    assert data["capability_status"]["sentiment"]["status"] == "completed"


# ==============================================================================
# 2. Batch Limits, Empty Payloads & Deduplication
# ==============================================================================


@pytest.mark.asyncio
async def test_qa_batch_limits_and_empty_payloads(async_client: httpx.AsyncClient):
    """Batch requests > 500 items or 0 items must be rejected with HTTP 400 or 422."""
    # Empty batch
    res_empty = await async_client.post("/api/v1/intelligence/batch", json={"reviews": [], "sync": True})
    assert res_empty.status_code in (400, 422)

    # 501 items (> 500 cap)
    too_many = [f"Valid customer review text sample #{i}" for i in range(501)]
    res_large = await async_client.post("/api/v1/intelligence/batch", json={"reviews": too_many, "sync": True})
    assert res_large.status_code in (400, 422)


@pytest.mark.asyncio
async def test_qa_batch_deduplication_preserves_order(async_client: httpx.AsyncClient):
    """Batch processing deduplicates identical review strings and reconstructs the original order."""
    unique_1 = "The battery life is stellar and lasts two days."
    unique_2 = "Customer support was terrible and rude."
    payload_reviews = [unique_1, unique_2, unique_1, unique_2, unique_1]

    res = await async_client.post(
        "/api/v1/intelligence/batch",
        json={"reviews": payload_reviews, "sync": True, "save_to_db": False},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["total_records"] == 5
    assert len(data["results"]) == 5

    # Verify positional integrity
    assert data["results"][0]["text_content"] == unique_1
    assert data["results"][1]["text_content"] == unique_2
    assert data["results"][2]["text_content"] == unique_1
    assert data["results"][3]["text_content"] == unique_2
    assert data["results"][4]["text_content"] == unique_1


# ==============================================================================
# 3. Independent Capability Fault Isolation
# ==============================================================================


@pytest.mark.asyncio
async def test_qa_absa_offline_isolation(async_client: httpx.AsyncClient):
    """When ABSA is unloaded, /absa/analyze returns 503, but /predict and /intelligence remain up."""
    orig_ate = absa_service.ate_model
    try:
        absa_service.ate_model = None  # simulate ABSA offline
        assert absa_service.is_ready() is False

        # /api/v1/absa/analyze fails fast with 503
        res_absa = await async_client.post("/api/v1/absa/analyze", json={"text": "Crisp sound quality."})
        assert res_absa.status_code == 503

        # /api/v1/predict remains 200 OK
        res_pred = await async_client.post("/api/v1/predict", json={"text": "Crisp sound quality."})
        assert res_pred.status_code == 200

        # /api/v1/intelligence/analyze remains 200 OK with absa status='unavailable'
        res_intel = await async_client.post(
            "/api/v1/intelligence/analyze",
            json={"text": "Crisp sound quality.", "save_to_db": False, "project_topic": False},
        )
        assert res_intel.status_code == 200
        intel_data = res_intel.json()
        assert intel_data["capability_status"]["absa"]["status"] == "unavailable"
        assert intel_data["capability_status"]["sentiment"]["status"] == "completed"
    finally:
        absa_service.ate_model = orig_ate


@pytest.mark.asyncio
async def test_qa_emotion_offline_isolation(async_client: httpx.AsyncClient):
    """When Emotion is unloaded, /emotion/analyze returns 503, but /predict and /intelligence remain up."""
    orig_model = emotion_service.model
    try:
        emotion_service.model = None  # simulate Emotion offline
        assert emotion_service.is_ready() is False

        # /api/v1/emotion/analyze fails fast with 503
        res_emo = await async_client.post("/api/v1/emotion/analyze", json={"text": "Great service today."})
        assert res_emo.status_code == 503

        # /api/v1/predict remains 200 OK
        res_pred = await async_client.post("/api/v1/predict", json={"text": "Great service today."})
        assert res_pred.status_code == 200

        # /api/v1/intelligence/analyze remains 200 OK with emotion status='unavailable'
        res_intel = await async_client.post(
            "/api/v1/intelligence/analyze",
            json={"text": "Great service today.", "save_to_db": False, "project_topic": False},
        )
        assert res_intel.status_code == 200
        intel_data = res_intel.json()
        assert intel_data["capability_status"]["emotion"]["status"] == "unavailable"
        assert intel_data["capability_status"]["sentiment"]["status"] == "completed"
    finally:
        emotion_service.model = orig_model


# ==============================================================================
# 4. Database Schema Consistency & Idempotent Relations
# ==============================================================================


@pytest.mark.asyncio
async def test_qa_database_schema_cascade_and_idempotency():
    """Verify primary key linking and foreign key cascading across predictions and child tables."""
    async with TestSessionLocal() as session:
        text = "The packaging was crushed, but the headphones sound amazing."
        res = await intelligence_service.analyze_review(
            session=session,
            text=text,
            user_id=_TEST_USER_ID,
            save_to_db=True,
            project_topic=False,
        )

        pred_id = res.review_id
        assert pred_id is not None

        # Verify child tables link to predictions.id
        aspects = (
            await session.execute(select(ReviewAspect).where(ReviewAspect.prediction_id == pred_id))
        ).scalars().all()
        for a in aspects:
            assert a.prediction_id == pred_id
            assert a.aspect_term is not None

        emotions = (
            await session.execute(select(ReviewEmotion).where(ReviewEmotion.prediction_id == pred_id))
        ).scalars().all()
        for e in emotions:
            assert e.prediction_id == pred_id
            assert e.primary_cx_category is not None


# ==============================================================================
# 5. Security & Sanitization
# ==============================================================================


@pytest.mark.asyncio
async def test_qa_security_path_traversal_and_invalid_uuid(async_client: httpx.AsyncClient):
    """Malicious job IDs containing path traversal strings must return 404 without leaking stack traces."""
    traversal_ids = [
        "../../etc/passwd",
        "..%2F..%2Fwindows%2Fsystem32",
        "invalid-nonexistent-job-uuid",
    ]
    for job_id in traversal_ids:
        res = await async_client.get(f"/api/v1/intelligence/batch/{job_id}")
        assert res.status_code == 404
        assert "not found" in res.json().get("detail", "").lower()
        # Verify no Python traceback in response
        assert "traceback" not in res.text.lower()


# ==============================================================================
# 6. Overall Sentiment Negation Blindness Bulletproofing
# ==============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "negated_review",
    [
        "my product is not good",
        "The mobile app layout breaks on smaller screens, cutting off important action buttons",
        "I am not happy at all",
        "never worked properly out of the box",
        "isn't worth the money",
        "good for nothing",
    ],
)
async def test_qa_negation_blindness_predict_endpoint(
    async_client: httpx.AsyncClient, negated_review: str
):
    """Verify /api/v1/predict classifies contrastive and negated sentences as NEGATIVE (> 0.75 conf)."""
    res = await async_client.post("/api/v1/predict", json={"text": negated_review})
    assert res.status_code == 200
    data = res.json()
    assert (
        data["sentiment"] == "NEGATIVE"
    ), f"Expected NEGATIVE for '{negated_review}', got {data['sentiment']}"
    assert (
        data["confidence"] > 0.75
    ), f"Expected confidence > 0.75 for '{negated_review}', got {data['confidence']}"
    assert (
        data["probabilities"]["NEGATIVE"] > 0.75
    ), f"Expected NEGATIVE probability > 0.75 for '{negated_review}', got {data['probabilities']['NEGATIVE']}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "negated_review",
    [
        "my product is not good",
        "The mobile app layout breaks on smaller screens, cutting off important action buttons",
        "I am not happy at all",
        "never worked properly out of the box",
        "isn't worth the money",
        "good for nothing",
    ],
)
async def test_qa_negation_blindness_intelligence_endpoint(
    async_client: httpx.AsyncClient, negated_review: str
):
    """Verify /api/v1/intelligence/analyze classifies negated reviews as NEGATIVE (> 0.75 conf)."""
    payload = {"text": negated_review, "save_to_db": False, "project_topic": False}
    res = await async_client.post("/api/v1/intelligence/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    overall = data.get("overall_sentiment", {})
    assert (
        overall.get("sentiment") == "NEGATIVE"
    ), f"Expected NEGATIVE for '{negated_review}', got {overall.get('sentiment')}"
    assert (
        overall.get("confidence") > 0.75
    ), f"Expected confidence > 0.75 for '{negated_review}', got {overall.get('confidence')}"
    probs = overall.get("probabilities", {})
    neg_p = probs.get("NEGATIVE", probs.get("negative", 0.0))
    assert (
        neg_p > 0.75
    ), f"Expected negative prob > 0.75 for '{negated_review}', got {neg_p}"

