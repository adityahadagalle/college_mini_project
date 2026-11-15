"""Comprehensive Test Suite for Phase 2: Emotion Detection & CX Taxonomy.

Verifies:
1. Single clear emotion (anger, joy, gratitude).
2. Multiple simultaneous emotions (e.g. anger + gratitude in one review).
3. Neutral / informative low-emotion text.
4. Negation handling ("not happy at all with this purchase").
5. Mixed sentiment vs emotion independence.
6. Empty and whitespace input handling without crashing.
7. Batch inference API endpoint (`POST /api/v1/emotion/batch`).
8. Fail-fast HTTP 503 when models are unloaded without affecting `/predict` or `/absa/analyze`.
9. Database persistence in `review_emotions` table and analytics aggregation.
"""

from datetime import datetime, timezone
from typing import Any

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.database import Prediction, ReviewEmotion, User
from app.services.absa_service import absa_service
from app.services.emotion_service import emotion_service
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


_TEST_USER_ID = "emotion-test-user-001"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="emotion_test@platform.ai",
            full_name="Emotion Tester",
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


# ==============================================================================
# Service Unit Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_emotion_single_clear_delight():
    """Verify single dominant positive emotion (e.g. joy / excitement)."""
    text = "I am absolutely thrilled and overjoyed with this amazing phone!"
    res = await emotion_service.analyze(text)

    assert res.primary_emotion == "Delight / Excitement"
    assert res.confidence > 0.4
    labels = [e.label for e in res.detected_emotions]
    assert any(l in ["joy", "excitement", "admiration"] for l in labels)
    assert len(res.raw_scores) == 28


@pytest.mark.asyncio
async def test_emotion_single_clear_frustration():
    """Verify single dominant negative emotion (anger / annoyance)."""
    text = "I am extremely angry and furious at this terrible, defective device!"
    res = await emotion_service.analyze(text)

    assert res.primary_emotion == "Frustration / Anger"
    assert res.confidence > 0.4
    labels = [e.label for e in res.detected_emotions]
    assert any(l in ["anger", "annoyance", "disgust"] for l in labels)


@pytest.mark.asyncio
async def test_emotion_concurrent_multiple_emotions():
    """Verify simultaneous multi-label detection (e.g. annoyance + gratitude in one review)."""
    text = "This device makes me furious and annoyed, but I am very grateful for the quick refund."
    res = await emotion_service.analyze(text)

    # Must detect both Frustration / Anger and Satisfaction / Gratitude
    detected_cats = [c.category for c in res.detected_cx_categories]
    detected_raw = [e.label for e in res.detected_emotions]

    assert any("Anger" in c or "Frustration" in c for c in detected_cats)
    assert any("Gratitude" in c or "Satisfaction" in c for c in detected_cats)
    assert "gratitude" in detected_raw
    assert any(l in ["annoyance", "anger"] for l in detected_raw)


@pytest.mark.asyncio
async def test_emotion_neutral_informative():
    """Verify neutral technical/spec text maps to Neutral / Informative."""
    text = "The package measures 15 by 20 centimeters and arrived on Tuesday."
    res = await emotion_service.analyze(text)

    assert res.primary_emotion == "Neutral / Informative"
    # Should not trigger high anger or joy
    assert res.raw_scores["anger"] < 0.35
    assert res.raw_scores["joy"] < 0.35


@pytest.mark.asyncio
async def test_emotion_negation_handling():
    """Verify negation: 'not happy at all' expresses disappointment / frustration, not joy."""
    text = "I am not happy at all with this purchase."
    res = await emotion_service.analyze(text)

    assert res.primary_emotion in ["Disappointment", "Frustration / Anger"]
    # Joy and excitement must remain very low
    assert res.raw_scores["joy"] < 0.25
    assert res.raw_scores["excitement"] < 0.25


@pytest.mark.asyncio
async def test_emotion_sentiment_vs_emotion_distinction():
    """Verify emotion captures granular state (e.g., sadness/disappointment vs pure negative)."""
    text = "I really wanted to love this laptop, but I feel so let down and sad that it keeps crashing."
    res = await emotion_service.analyze(text)

    assert res.primary_emotion in ["Disappointment", "Frustration / Anger"]
    detected_raw = [e.label for e in res.detected_emotions]
    assert any(l in ["disappointment", "sadness", "disapproval", "annoyance"] for l in detected_raw)


@pytest.mark.asyncio
async def test_emotion_empty_and_whitespace_input():
    """Verify empty or whitespace strings return gracefully without raising."""
    res_empty = await emotion_service.analyze("")
    assert res_empty.primary_emotion == "Neutral / Informative"
    assert res_empty.detected_emotions == []

    res_ws = await emotion_service.analyze("    \n\t   ")
    assert res_ws.primary_emotion == "Neutral / Informative"
    assert res_ws.detected_emotions == []


# ==============================================================================
# API Endpoint Integration Tests
# ==============================================================================


def test_api_emotion_analyze_endpoint(client):
    """Test POST /api/v1/emotion/analyze contract and payload."""
    payload = {
        "text": "The camera quality is magnificent, thank you so much!",
        "save_to_db": True,
    }
    resp = client.post("/api/v1/emotion/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["text"] == payload["text"]
    assert "primary_emotion" in data
    assert "confidence" in data
    assert "detected_cx_categories" in data
    assert "detected_emotions" in data
    assert "raw_scores" in data
    assert len(data["raw_scores"]) == 28
    assert "latency_ms" in data
    assert data["model_name"] == "RoBERTa-GoEmotions-MultiLabel"


def test_api_emotion_batch_endpoint(client):
    """Test POST /api/v1/emotion/batch micro-batching."""
    payload = {
        "texts": [
            "Super excited about this new feature!",
            "Terrible customer service, very disappointed.",
            "Order arrived in standard cardboard packaging.",
        ],
        "save_to_db": True,
    }
    resp = client.post("/api/v1/emotion/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_samples"] == 3
    assert len(data["results"]) == 3
    assert data["results"][0]["primary_emotion"] == "Delight / Excitement"
    assert data["results"][1]["primary_emotion"] in ["Disappointment", "Frustration / Anger"]


def test_api_emotion_analytics_endpoint(client):
    """Test GET /api/v1/emotion/analytics with DB persistence."""
    # Seed records
    client.post(
        "/api/v1/emotion/analyze",
        json={"text": "I am so happy and thankful for this!", "save_to_db": True},
    )
    client.post(
        "/api/v1/emotion/analyze",
        json={"text": "This is completely broken and makes me so angry.", "save_to_db": True},
    )

    resp = client.get("/api/v1/emotion/analytics")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_records" in data
    assert data["total_records"] >= 2
    assert "cx_distribution" in data
    assert "cx_ratios" in data
    assert "top_raw_emotions" in data
    assert "recent_emotion_records" in data


def test_api_emotion_metadata_endpoint(client):
    """Test GET /api/v1/emotion/metadata returns real evaluated metrics."""
    resp = client.get("/api/v1/emotion/metadata")
    assert resp.status_code == 200
    data = resp.json()

    assert data["model_name"] == "RoBERTa-GoEmotions-MultiLabel"
    assert "metrics" in data
    assert "micro_f1" in data["metrics"]
    assert "macro_f1" in data["metrics"]
    assert "calibrated_thresholds" in data
    assert len(data["calibrated_thresholds"]) == 28


def test_emotion_503_fail_fast_when_models_unloaded(client, monkeypatch):
    """Verify HTTP 503 fail-fast when emotion model is unloaded, without affecting /predict or /absa."""
    monkeypatch.setattr(emotion_service, "model", None)
    assert not emotion_service.is_ready()

    # /api/v1/emotion/analyze must fail-fast with 503
    resp_analyze = client.post(
        "/api/v1/emotion/analyze",
        json={"text": "I am thrilled with this!"},
    )
    assert resp_analyze.status_code == 503
    assert "Emotion neural models are unavailable" in resp_analyze.json()["detail"]

    # /api/v1/emotion/batch must fail-fast with 503
    resp_batch = client.post(
        "/api/v1/emotion/batch",
        json={"texts": ["Great product!"]},
    )
    assert resp_batch.status_code == 503
    assert "Emotion neural models are unavailable" in resp_batch.json()["detail"]

    # Existing /api/v1/predict must remain 200 OK
    resp_predict = client.post(
        "/api/v1/predict",
        json={"text": "Outstanding platform with fast inference."},
    )
    assert resp_predict.status_code == 200
    assert resp_predict.json()["sentiment"] == "POSITIVE"

    # Phase 1 /api/v1/absa/analyze must remain 200 OK
    resp_absa = client.post(
        "/api/v1/absa/analyze",
        json={"text": "The camera is excellent, but the battery is terrible."},
    )
    assert resp_absa.status_code == 200
    assert resp_absa.json()["aspect_count"] >= 2
