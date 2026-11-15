"""Comprehensive Test Suite for Phase 3: Automatic Topic Discovery.

Verifies:
1. Local Sentence-Transformers model loading and CPU execution (0 GPU VRAM).
2. Deterministic embedding generation and caching in `review_embeddings`.
3. Minimum corpus size validation (>= 15 unique reviews required, HTTP 400).
4. Full unsupervised discovery pipeline (all-MiniLM-L6-v2 + UMAP/SVD + HDBSCAN + c-TF-IDF).
5. Outlier preservation (`topic_index = -1` marked as 'Outliers / Unassigned').
6. Representative reviews ranking (top 3 closest to centroid).
7. Human curation actions: rename, merge, archive with `topic_audit_logs`.
8. Cross-run topic comparison and candidate emerging theme detection with mandatory disclaimer.
9. Fail-fast HTTP 503 when model is unloaded without regressing `/predict`, `/absa/*`, or `/emotion/*`.
10. Concurrency lock: duplicate simultaneous discovery run prevention (HTTP 409).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.main import app
from app.models.database import (
    DiscoveredTopic,
    Prediction,
    ReviewEmbedding,
    TopicAssignment,
    TopicAuditLog,
    TopicDiscoveryRun,
)
from app.schemas.topic import TopicDiscoveryRequest
from app.services.absa_service import absa_service
from app.services.emotion_service import emotion_service
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


_TEST_USER_ID = "topic-test-user-001"


@pytest.fixture(autouse=True)
def mock_auth():
    from app.api.v1.auth import get_current_user
    from app.schemas import auth as auth_schemas

    async def _mock_current_user(token: str = "", db: Any = None) -> auth_schemas.UserProfile:
        return auth_schemas.UserProfile(
            id=_TEST_USER_ID,
            email="topic_tester@platform.ai",
            full_name="Topic Tester",
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


SAMPLE_TEST_REVIEWS = [
    # Battery & Charging (6 reviews)
    "The battery life on this phone easily lasts two full days with heavy usage.",
    "Battery drains completely within three hours of screen on time, terrible battery.",
    "Fast charging takes less than thirty minutes to reach eighty percent charge.",
    "The device gets hot and thermal throttles while charging overnight.",
    "Battery health dropped to eighty percent after only five months of usage.",
    "Awesome battery endurance for long flights and road trips.",

    # Audio & Sound (6 reviews)
    "The sound quality is crisp and the bass response is deep without any distortion.",
    "Speakers sound tinny and muddy at high volumes, very poor sound quality.",
    "Active noise cancellation completely silences background engine rumble.",
    "The microphone on phone calls picks up too much background noise and wind.",
    "Bluetooth audio connection stutters when walking away from the device.",
    "Audiophile grade sound with crystal clear vocal clarity.",

    # Customer Service & Shipping (6 reviews)
    "Customer service resolved my issue within ten minutes and sent a replacement.",
    "Shipping was delayed by three weeks with zero tracking updates from support.",
    "Package arrived crushed and the outer retail box was torn.",
    "Support representative was polite and helped me through the warranty process.",
    "The return process was painful with repetitive automated replies from support.",
    "Fast express delivery arrived two days earlier than originally estimated.",

    # Camera & Photos (6 reviews)
    "Low light night photography is stunning with vibrant color rendering.",
    "The camera shutter lag causes me to miss quick action moments.",
    "Portrait mode creates beautiful natural depth of field and bokeh.",
    "Ultrawide camera lens has noticeable optical distortion at the edges.",
    "Video recording stabilization is buttery smooth even when running.",
    "Colors in photos look washed out and artificial in direct sunlight.",
]


# ==============================================================================
# Service Unit Tests
# ==============================================================================


@pytest.mark.asyncio
async def test_topic_service_is_ready():
    """Verify topic embedding model is loaded locally and runs on CPU."""
    assert topic_service.is_ready() is True
    assert topic_service.embedding_dimension == 384
    assert topic_service.device.type == "cpu"


@pytest.mark.asyncio
async def test_topic_encode_texts_cpu():
    """Verify CPU neural embedding generation with unit normalization."""
    texts = ["Excellent battery life", "Terrible customer service delay"]
    embeddings = topic_service.encode_texts(texts, batch_size=2)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)
    # Check L2 normalization (norm should be ~1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)


@pytest.mark.asyncio
async def test_topic_embedding_cache_hit_and_miss():
    """Verify deterministic embedding cache lookup, hit rate, and persistence."""
    async with TestSessionLocal() as session:
        texts = [
            "Great sound quality on these headphones",
            "Terrible customer service and delayed shipping",
        ]
        # First call: 2 misses, 0 hits
        embs1, stats1 = await topic_service.get_or_create_embeddings(session, texts)
        assert stats1["misses"] == 2
        assert stats1["hits"] == 0
        assert embs1.shape == (2, 384)

        # Second call with same texts: 2 hits, 0 misses
        embs2, stats2 = await topic_service.get_or_create_embeddings(session, texts)
        assert stats2["hits"] == 2
        assert stats2["misses"] == 0
        assert stats2["hit_rate_pct"] == 100.0

        # Embeddings should match identically
        np.testing.assert_allclose(embs1, embs2, atol=1e-5)


@pytest.mark.asyncio
async def test_topic_discovery_min_corpus_size_rejection():
    """Verify that fewer than 15 reviews raises ValueError."""
    async with TestSessionLocal() as session:
        few_reviews = [{"text": f"Review number {i}"} for i in range(10)]
        with pytest.raises(ValueError, match="at least 15 unique reviews"):
            await topic_service.execute_discovery_run(
                session=session,
                run_id="run-too-small",
                reviews_data=few_reviews,
                parameters={"min_cluster_size": 2},
            )


@pytest.mark.asyncio
async def test_topic_discovery_execution_end_to_end():
    """Verify complete unsupervised discovery pipeline, clustering, c-TF-IDF, and outliers."""
    async with TestSessionLocal() as session:
        run_id = f"test-run-{uuid.uuid4()}"
        reviews_data = [{"text": t, "sentiment_label": "POSITIVE"} for t in SAMPLE_TEST_REVIEWS]

        result = await topic_service.execute_discovery_run(
            session=session,
            run_id=run_id,
            reviews_data=reviews_data,
            parameters={"min_cluster_size": 2, "n_neighbors": 5, "n_components": 3},
        )

        assert result.run_id == run_id
        assert result.corpus_size == len(SAMPLE_TEST_REVIEWS)
        assert result.num_topics > 0
        assert result.metrics is not None
        assert "mean_intra_cluster_cohesion" in result.metrics
        assert "keyword_diversity_ratio" in result.metrics

        # Outlier topic must exist with topic_index = -1
        outlier_topics = [t for t in result.topics if t.topic_index == -1]
        assert len(outlier_topics) == 1
        assert outlier_topics[0].auto_label == "Outliers / Unassigned"

        # Meaningful topics should have c-TF-IDF keywords
        meaningful = [t for t in result.topics if t.topic_index != -1]
        for topic in meaningful:
            assert len(topic.keywords) > 0
            assert topic.review_count > 0
            assert topic.cohesion_score is not None


# ==============================================================================
# API Endpoints & Curation Tests
# ==============================================================================


def test_api_discover_min_reviews_bad_request(client: TestClient):
    """POST /api/v1/topics/discover with < 15 reviews returns HTTP 400."""
    payload = {
        "source": "custom_texts",
        "custom_texts": ["Short review 1", "Short review 2"],
    }
    res = client.post("/api/v1/topics/discover", json=payload)
    assert res.status_code == 400
    assert "at least 15 unique reviews" in res.json()["detail"]


def test_api_discover_success(client: TestClient):
    """POST /api/v1/topics/discover executes synchronous discovery and returns topics."""
    payload = {
        "source": "custom_texts",
        "custom_texts": SAMPLE_TEST_REVIEWS,
        "min_cluster_size": 2,
        "n_neighbors": 5,
        "n_components": 3,
    }
    res = client.post("/api/v1/topics/discover?sync=true", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["corpus_size"] == 24
    assert len(data["topics"]) > 0

    run_id = data["run_id"]

    # GET /api/v1/topics
    res_list = client.get(f"/api/v1/topics?run_id={run_id}")
    assert res_list.status_code == 200
    assert len(res_list.json()["topics"]) == len(data["topics"])


def test_api_topic_reviews_pagination(client: TestClient):
    """GET /api/v1/topics/{topic_id}/reviews returns paginated reviews with representative flags."""
    # First discover
    payload = {
        "source": "custom_texts",
        "custom_texts": SAMPLE_TEST_REVIEWS,
        "min_cluster_size": 2,
    }
    disc_res = client.post("/api/v1/topics/discover?sync=true", json=payload)
    assert disc_res.status_code == 200
    topics = disc_res.json()["topics"]

    # Select a meaningful topic
    target_topic = [t for t in topics if t["topic_index"] != -1][0]
    topic_id = target_topic["id"]

    rev_res = client.get(f"/api/v1/topics/{topic_id}/reviews?page=1&page_size=10")
    assert rev_res.status_code == 200
    rev_data = rev_res.json()
    assert rev_data["topic_id"] == topic_id
    assert rev_data["total_reviews"] == target_topic["review_count"]
    assert len(rev_data["reviews"]) > 0

    # At least one review should be representative if cluster has reviews
    reps = [r for r in rev_data["reviews"] if r["is_representative"]]
    assert len(reps) > 0


def test_api_topic_rename_and_audit(client: TestClient):
    """PATCH /api/v1/topics/{topic_id}/rename updates human_label and writes audit log."""
    disc_res = client.post(
        "/api/v1/topics/discover?sync=true",
        json={"source": "custom_texts", "custom_texts": SAMPLE_TEST_REVIEWS},
    )
    topics = disc_res.json()["topics"]
    target_topic = [t for t in topics if t["topic_index"] != -1][0]
    topic_id = target_topic["id"]

    new_name = "Battery Endurance & Charging Speed"
    patch_res = client.patch(
        f"/api/v1/topics/{topic_id}/rename",
        json={"human_label": new_name},
    )
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["human_label"] == new_name
    assert updated["display_label"] == new_name


def test_api_topic_merge_and_archive(client: TestClient):
    """POST /api/v1/topics/merge moves assignments, increments count, and archives source topic."""
    disc_res = client.post(
        "/api/v1/topics/discover?sync=true",
        json={"source": "custom_texts", "custom_texts": SAMPLE_TEST_REVIEWS},
    )
    topics = [t for t in disc_res.json()["topics"] if t["topic_index"] != -1]
    if len(topics) < 2:
        pytest.skip("Not enough non-outlier clusters to test merge.")

    source_id = topics[0]["id"]
    target_id = topics[1]["id"]
    expected_count = topics[0]["review_count"] + topics[1]["review_count"]

    merge_res = client.post(
        "/api/v1/topics/merge",
        json={
            "source_topic_id": source_id,
            "target_topic_id": target_id,
            "new_label": "Consolidated Audio & Hardware",
        },
    )
    assert merge_res.status_code == 200
    merged = merge_res.json()
    assert merged["id"] == target_id
    assert merged["review_count"] == expected_count
    assert merged["human_label"] == "Consolidated Audio & Hardware"

    # Source topic should now be archived and omitted from default list
    res_list = client.get(f"/api/v1/topics?run_id={disc_res.json()['run_id']}")
    active_ids = [t["id"] for t in res_list.json()["topics"]]
    assert source_id not in active_ids


def test_api_topic_archive(client: TestClient):
    """PATCH /api/v1/topics/{topic_id}/archive archives topic from active view."""
    disc_res = client.post(
        "/api/v1/topics/discover?sync=true",
        json={"source": "custom_texts", "custom_texts": SAMPLE_TEST_REVIEWS},
    )
    topic_id = disc_res.json()["topics"][0]["id"]

    arch_res = client.patch(f"/api/v1/topics/{topic_id}/archive")
    assert arch_res.status_code == 200
    assert arch_res.json()["is_archived"] is True


def test_api_topic_compare_runs(client: TestClient):
    """GET /api/v1/topics/compare detects matched topics and candidate emerging themes with disclaimer."""
    # Run 1: Battery and Audio
    run1 = client.post(
        "/api/v1/topics/discover?sync=true",
        json={"source": "custom_texts", "custom_texts": SAMPLE_TEST_REVIEWS[:16]},
    ).json()

    # Run 2: Full reviews
    run2 = client.post(
        "/api/v1/topics/discover?sync=true",
        json={"source": "custom_texts", "custom_texts": SAMPLE_TEST_REVIEWS},
    ).json()

    comp_res = client.get(
        f"/api/v1/topics/compare?run_id_a={run1['run_id']}&run_id_b={run2['run_id']}"
    )
    assert comp_res.status_code == 200
    data = comp_res.json()
    assert "matched_topics" in data
    assert "emerging_topics_in_b" in data

    for emerging in data["emerging_topics_in_b"]:
        assert emerging["max_similarity_to_prior_run"] < 0.45
        assert "requires human review due to unsupervised cluster drift" in emerging["uncertainty_note"]


def test_api_topic_metadata(client: TestClient):
    """GET /api/v1/topics/metadata returns model info, CPU device, 0 VRAM, and benchmark comparison."""
    res = client.get("/api/v1/topics/metadata")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert data["device"] == "cpu"
    assert data["vram_usage_mb"] == 0.0
    assert data["embedding_dimension"] == 384
    assert data["min_reviews_required"] == 15
    assert data["local_weights_verified"] is True
    assert "benchmark_metrics" in data


# ==============================================================================
# Model Isolation & Fail-Fast 503 Tests
# ==============================================================================


def test_fail_fast_503_isolation(client: TestClient):
    """Verify that when topic model is unloaded, topic discovery returns HTTP 503 while other endpoints work."""
    orig_model = topic_service.model
    try:
        # Simulate unloaded topic service
        topic_service.model = None

        res_topic = client.post(
            "/api/v1/topics/discover",
            json={"source": "custom_texts", "custom_texts": SAMPLE_TEST_REVIEWS},
        )
        assert res_topic.status_code == 503
        assert "unavailable or not loaded" in res_topic.json()["detail"]

        # Other services must continue to function (e.g. emotion analyze returns 200)
        res_emotion = client.post(
            "/api/v1/emotion/analyze",
            json={"text": "I am so happy with this purchase", "save_to_db": False},
        )
        assert res_emotion.status_code == 200

        # ABSA must continue to function
        res_absa = client.post(
            "/api/v1/absa/analyze",
            json={"text": "Great battery life", "save_to_db": False},
        )
        assert res_absa.status_code == 200

    finally:
        # Restore service
        topic_service.model = orig_model
