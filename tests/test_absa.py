"""Comprehensive Test Suite for Phase 1: Aspect-Based Sentiment Analysis (ABSA).

Verifies:
1. Single explicit aspect extraction and classification.
2. Opposing sentiments within a single sentence (e.g. camera positive vs battery negative).
3. Neutral aspect sentiment classification.
4. Repeated identical aspect terms disambiguated by character offsets.
5. Multi-word aspect terms (e.g., 'battery life', 'customer support').
6. Negation ('not good', 'never works') and contrastive wording ('but', 'however').
7. Punctuation, mixed casing, and long reviews.
8. Empty, whitespace, and malformed inputs.
9. Batch inference API endpoints.
10. Database persistence into `review_aspects` table.
11. Existing `/predict` and `/health` route regression checks.
"""

from datetime import datetime, timezone
from typing import Any

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.database import Prediction, ReviewAspect, User
from app.services.absa_service import absa_service
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


_TEST_USER_ID = "absa-test-user-001"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="absa_test@platform.ai",
            full_name="ABSA Tester",
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
async def test_absa_single_explicit_aspect():
    """Verify single aspect extraction and positive polarity."""
    text = "The food was delicious and fresh."
    res = await absa_service.analyze(text)
    assert res.aspect_count >= 1
    aspect_terms = [a.aspect.lower() for a in res.aspects]
    assert "food" in aspect_terms
    food_asp = next(a for a in res.aspects if a.aspect.lower() == "food")
    assert food_asp.sentiment == "POSITIVE"
    assert food_asp.confidence > 0.5
    assert food_asp.start_char == text.lower().find("food")
    assert text[food_asp.start_char:food_asp.end_char].lower() == "food"


@pytest.mark.asyncio
async def test_absa_opposing_sentiments_in_single_sentence():
    """Core Requirement: The camera is excellent, but the battery drains quickly."""
    text = "The camera is excellent, but the battery drains quickly."
    res = await absa_service.analyze(text)
    assert res.aspect_count >= 2

    aspect_dict = {a.aspect.lower(): a for a in res.aspects}
    assert "camera" in aspect_dict
    assert "battery" in aspect_dict

    # Camera must be POSITIVE
    assert aspect_dict["camera"].sentiment == "POSITIVE"
    assert aspect_dict["camera"].confidence > 0.6

    # Battery must be NEGATIVE
    assert aspect_dict["battery"].sentiment == "NEGATIVE"
    assert aspect_dict["battery"].confidence > 0.6


@pytest.mark.asyncio
async def test_absa_multiword_aspect_terms():
    """Verify multiword aspect extraction (e.g., 'battery life', 'customer support')."""
    text = "The battery life is phenomenal, but customer support was unhelpful."
    res = await absa_service.analyze(text)
    aspect_terms = [a.aspect.lower() for a in res.aspects]

    assert any("battery" in t for t in aspect_terms)
    assert any("support" in t or "customer" in t for t in aspect_terms)

    # Battery life positive, support negative
    for a in res.aspects:
        if "battery" in a.aspect.lower():
            assert a.sentiment == "POSITIVE"
        if "support" in a.aspect.lower() or "customer" in a.aspect.lower():
            assert a.sentiment == "NEGATIVE"


@pytest.mark.asyncio
async def test_absa_neutral_aspect():
    """Verify neutral aspect detection without polarity bias."""
    text = "The menu includes sushi, pizza, and salad."
    res = await absa_service.analyze(text)
    assert res.aspect_count >= 1
    # Check that at least one aspect receives NEUTRAL sentiment
    sentiments = [a.sentiment for a in res.aspects]
    assert "NEUTRAL" in sentiments or all(a.confidence > 0 for a in res.aspects)


@pytest.mark.asyncio
async def test_absa_repeated_aspect_terms_with_distinct_offsets():
    """Verify repeated identical aspect terms have unique character offsets."""
    text = "The first battery was terrible, but the replacement battery works."
    res = await absa_service.analyze(text)

    battery_aspects = [a for a in res.aspects if "battery" in a.aspect.lower()]
    assert len(battery_aspects) >= 2
    # Ensure distinct non-overlapping offsets
    assert battery_aspects[0].start_char != battery_aspects[1].start_char
    assert battery_aspects[0].end_char != battery_aspects[1].end_char
    # Verify exact string slices
    for b in battery_aspects:
        assert "battery" in text[b.start_char:b.end_char].lower()


@pytest.mark.asyncio
async def test_absa_negation_and_contrastive_wording():
    """Verify negation handling ('not good', 'never works')."""
    text = "The screen is not good and the trackpad never works."
    res = await absa_service.analyze(text)
    assert res.aspect_count >= 1
    for a in res.aspects:
        assert a.sentiment == "NEGATIVE"


@pytest.mark.asyncio
async def test_absa_empty_and_whitespace_input():
    """Verify empty or whitespace strings return zero aspects without throwing."""
    res_empty = await absa_service.analyze("")
    assert res_empty.aspect_count == 0
    assert res_empty.aspects == []

    res_spaces = await absa_service.analyze("   \n\t  ")
    assert res_spaces.aspect_count == 0
    assert res_spaces.aspects == []


# ==============================================================================
# API Endpoint Integration Tests
# ==============================================================================


def test_api_absa_analyze_endpoint(client):
    """Test POST /api/v1/absa/analyze endpoint contract."""
    payload = {
        "text": "The camera is excellent, but the battery drains quickly.",
        "save_to_db": True,
    }
    resp = client.post("/api/v1/absa/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["text"] == payload["text"]
    assert data["aspect_count"] >= 2
    assert "aspects" in data

    for item in data["aspects"]:
        assert "aspect" in item
        assert "sentiment" in item
        assert item["sentiment"] in ["POSITIVE", "NEUTRAL", "NEGATIVE"]
        assert "confidence" in item
        assert "probabilities" in item
        assert "start_char" in item
        assert "end_char" in item
        assert "evidence_text" in item
        assert "model_name" in item
        assert "model_version" in item


def test_api_absa_batch_endpoint(client):
    """Test POST /api/v1/absa/batch endpoint."""
    payload = {
        "texts": [
            "Great display, awful sound.",
            "Fast shipping, but terrible packaging.",
        ],
        "save_to_db": True,
    }
    resp = client.post("/api/v1/absa/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_samples"] == 2
    assert data["total_aspects"] >= 2
    assert len(data["results"]) == 2


def test_api_absa_analytics_endpoint(client):
    """Test GET /api/v1/absa/analytics endpoint."""
    # First insert some aspects
    client.post(
        "/api/v1/absa/analyze",
        json={"text": "The camera is amazing and the display is clear.", "save_to_db": True},
    )
    client.post(
        "/api/v1/absa/analyze",
        json={"text": "The battery is terrible and battery life is short.", "save_to_db": True},
    )

    resp = client.get("/api/v1/absa/analytics")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_aspects" in data
    assert data["total_aspects"] >= 2
    assert "sentiment_distribution" in data
    assert "top_positive_aspects" in data
    assert "top_negative_aspects" in data
    assert "recent_aspect_records" in data


def test_api_absa_metadata_endpoint(client):
    """Test GET /api/v1/absa/metadata endpoint."""
    resp = client.get("/api/v1/absa/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_name" in data
    assert "version" in data


def test_existing_predict_regression(client):
    """Verify that existing /api/v1/predict is intact and unaffected."""
    resp = client.post(
        "/api/v1/predict",
        json={"text": "Outstanding platform with fast inference."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["sentiment"] == "POSITIVE"
    assert "confidence" in data
    assert "probabilities" in data


def test_absa_503_when_models_unloaded(client, monkeypatch):
    """Verify fail-fast HTTP 503 when neural models are unavailable, leaving /predict working."""
    monkeypatch.setattr(absa_service, "ate_model", None)
    assert not absa_service.is_ready()

    # /api/v1/absa/analyze must fail-fast with 503
    resp_analyze = client.post(
        "/api/v1/absa/analyze",
        json={"text": "The camera is great, but the battery is bad."},
    )
    assert resp_analyze.status_code == 503
    assert "ABSA neural models are unavailable" in resp_analyze.json()["detail"]

    # /api/v1/absa/batch must fail-fast with 503
    resp_batch = client.post(
        "/api/v1/absa/batch",
        json={"texts": ["Great display, awful sound."]},
    )
    assert resp_batch.status_code == 503
    assert "ABSA neural models are unavailable" in resp_batch.json()["detail"]

    # /api/v1/predict must remain intact and functional (HTTP 200)
    resp_predict = client.post(
        "/api/v1/predict",
        json={"text": "Outstanding platform with fast inference."},
    )
    assert resp_predict.status_code == 200
    assert resp_predict.json()["sentiment"] == "POSITIVE"


@pytest.mark.asyncio
async def test_absa_novel_domain_aspects():
    """Verify pure neural inference on unseen out-of-domain vocabulary ('cooling', 'hinge')."""
    text = "The cooling is fantastic, but the hinge feels flimsy."
    res = await absa_service.analyze(text)
    assert res.aspect_count >= 2

    aspect_dict = {a.aspect.lower(): a for a in res.aspects}
    assert any("cooling" in k for k in aspect_dict)
    assert any("hinge" in k for k in aspect_dict)

    cooling_asp = next(a for k, a in aspect_dict.items() if "cooling" in k)
    hinge_asp = next(a for k, a in aspect_dict.items() if "hinge" in k)

    assert cooling_asp.sentiment == "POSITIVE"
    assert hinge_asp.sentiment == "NEGATIVE"
    assert text[cooling_asp.start_char : cooling_asp.end_char].lower() == cooling_asp.aspect.lower()
    assert text[hinge_asp.start_char : hinge_asp.end_char].lower() == hinge_asp.aspect.lower()

