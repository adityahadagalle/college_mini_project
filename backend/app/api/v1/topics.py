"""Automatic Topic Discovery REST API Endpoints."""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import AsyncSessionLocal, get_db
from app.models.database import Prediction, User
from app.schemas.topic import (
    DiscoveredTopicItem,
    TopicComparisonResponse,
    TopicDiscoveryJobStatus,
    TopicDiscoveryRequest,
    TopicListResponse,
    TopicMergeRequest,
    TopicMetadataResponse,
    TopicRenameRequest,
    TopicReviewsResponse,
)
from app.services.topic_service import (
    MIN_REVIEWS_THRESHOLD,
    TOPIC_METADATA_PATH,
    normalize_text,
    topic_service,
)

router = APIRouter()


async def _run_discovery_background(
    run_id: str,
    reviews_data: List[Dict[str, Any]],
    parameters: Dict[str, Any],
    user_id: Optional[str] = None,
):
    """Background task runner for asynchronous topic discovery."""
    async with AsyncSessionLocal() as session:
        try:
            await topic_service.execute_discovery_run(
                session=session,
                run_id=run_id,
                reviews_data=reviews_data,
                parameters=parameters,
                user_id=user_id,
            )
        except Exception as e:
            # Error is logged and tracked in topic_service._jobs
            pass


@router.post(
    "/discover",
    response_model=Union[TopicListResponse, TopicDiscoveryJobStatus],
)
async def discover_topics(
    payload: TopicDiscoveryRequest,
    background_tasks: BackgroundTasks,
    sync: bool = Query(
        default=True,
        description="Whether to run synchronously (default True for immediate feedback)",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Initiate a collection-level topic discovery run over reviews."""
    if not topic_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topic discovery embedding model is unavailable or not loaded",
        )

    if topic_service.is_job_running():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another topic discovery run is currently in progress. Please wait for it to complete.",
        )

    # Ingest reviews based on payload source
    reviews_data: List[Dict[str, Any]] = []

    if payload.source == "custom_texts":
        if not payload.custom_texts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Custom texts list cannot be empty when source='custom_texts'.",
            )
        for text in payload.custom_texts:
            if text and text.strip():
                reviews_data.append({"text": text.strip()})
    else:
        # Ingest from Prediction table
        stmt = select(Prediction).order_by(Prediction.created_at.desc())
        if payload.sentiment_filter:
            stmt = stmt.where(Prediction.sentiment_label == payload.sentiment_filter.upper())
        if payload.start_date:
            try:
                dt_start = datetime.fromisoformat(payload.start_date)
                stmt = stmt.where(Prediction.created_at >= dt_start)
            except ValueError:
                pass
        if payload.end_date:
            try:
                dt_end = datetime.fromisoformat(payload.end_date)
                stmt = stmt.where(Prediction.created_at <= dt_end)
            except ValueError:
                pass

        res = await db.execute(stmt)
        predictions = res.scalars().all()
        for p in predictions:
            reviews_data.append(
                {
                    "prediction_id": p.id,
                    "text": p.text_content,
                    "sentiment_label": p.sentiment_label,
                    "confidence_score": p.confidence_score,
                }
            )

    # Validate unique reviews count
    unique_texts = set(normalize_text(r["text"]) for r in reviews_data if len(normalize_text(r["text"])) > 3)
    if len(unique_texts) < MIN_REVIEWS_THRESHOLD:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Topic discovery requires at least {MIN_REVIEWS_THRESHOLD} unique reviews for "
                f"statistically valid clustering. Provided corpus size: {len(unique_texts)}"
            ),
        )

    run_id = str(uuid.uuid4())
    params = payload.model_dump()

    if sync:
        try:
            result = await topic_service.execute_discovery_run(
                session=db,
                run_id=run_id,
                reviews_data=reviews_data,
                parameters=params,
                user_id=current_user.id,
            )
            return result
        except ValueError as val_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err)
            ) from val_err
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Topic discovery execution failed: {str(exc)}",
            ) from exc
    else:
        # Run asynchronously in background task
        now_iso = datetime.utcnow().isoformat()
        topic_service._jobs[run_id] = {
            "job_id": run_id,
            "status": "PENDING",
            "progress": 0.0,
            "stage": "Job queued for background processing",
            "corpus_size": len(unique_texts),
            "num_topics": 0,
            "outlier_count": 0,
            "metrics": None,
            "error_message": None,
            "created_at": now_iso,
            "completed_at": None,
        }
        background_tasks.add_task(
            _run_discovery_background,
            run_id=run_id,
            reviews_data=reviews_data,
            parameters=params,
            user_id=current_user.id,
        )
        return TopicDiscoveryJobStatus(**topic_service._jobs[run_id])


@router.get("/jobs/{job_id}", response_model=TopicDiscoveryJobStatus)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> TopicDiscoveryJobStatus:
    """Poll progress status of an asynchronous topic discovery job."""
    status_obj = topic_service.get_job_status(job_id)
    if not status_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic discovery job '{job_id}' not found.",
        )
    return status_obj


@router.get("", response_model=TopicListResponse)
async def list_topics(
    run_id: Optional[str] = Query(None, description="Topic discovery run UUID (defaults to latest)"),
    include_archived: bool = Query(False, description="Whether to include human-archived topics"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TopicListResponse:
    """List discovered topics for a specific discovery run (or most recent run)."""
    try:
        response = await topic_service.list_topics(
            session=db, run_id=run_id, include_archived=include_archived
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err)
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch topics: {str(exc)}",
        ) from exc


@router.get("/{topic_id}/reviews", response_model=TopicReviewsResponse)
async def get_topic_reviews(
    topic_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TopicReviewsResponse:
    """Fetch paginated review assignments for a topic, highlighting top 3 representatives."""
    try:
        return await topic_service.get_topic_reviews(
            session=db, topic_id=topic_id, page=page, page_size=page_size
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err)
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve topic reviews: {str(exc)}",
        ) from exc


@router.patch("/{topic_id}/rename", response_model=DiscoveredTopicItem)
async def rename_topic(
    topic_id: str,
    payload: TopicRenameRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiscoveredTopicItem:
    """Set a human-curated title for a discovered topic and record an audit log."""
    try:
        return await topic_service.rename_topic(
            session=db,
            topic_id=topic_id,
            human_label=payload.human_label,
            user_id=current_user.id,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err)
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rename topic: {str(exc)}",
        ) from exc


@router.post("/merge", response_model=DiscoveredTopicItem)
async def merge_topics(
    payload: TopicMergeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiscoveredTopicItem:
    """Merge source topic into target topic, archive source, and record audit log."""
    try:
        return await topic_service.merge_topics(
            session=db,
            source_topic_id=payload.source_topic_id,
            target_topic_id=payload.target_topic_id,
            new_label=payload.new_label,
            user_id=current_user.id,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err)
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to merge topics: {str(exc)}",
        ) from exc


@router.patch("/{topic_id}/archive", response_model=DiscoveredTopicItem)
async def archive_topic(
    topic_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DiscoveredTopicItem:
    """Archive a discovered topic from the active view and record an audit log."""
    try:
        return await topic_service.archive_topic(
            session=db,
            topic_id=topic_id,
            user_id=current_user.id,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err)
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive topic: {str(exc)}",
        ) from exc


@router.get("/compare", response_model=TopicComparisonResponse)
async def compare_topic_runs(
    run_id_a: str = Query(..., description="First topic discovery run UUID (baseline)"),
    run_id_b: str = Query(..., description="Second topic discovery run UUID (comparison)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TopicComparisonResponse:
    """Compare topic centroids across two discovery runs to detect matched and candidate emerging themes."""
    try:
        return await topic_service.compare_runs(
            session=db, run_id_a=run_id_a, run_id_b=run_id_b
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(val_err)
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare topic discovery runs: {str(exc)}",
        ) from exc


@router.get("/metadata", response_model=TopicMetadataResponse)
async def get_topic_metadata() -> TopicMetadataResponse:
    """Expose benchmark evaluation, embedding model specs, and architecture parameters."""
    benchmark_metrics: Dict[str, Any] = {}
    if os.path.isfile(TOPIC_METADATA_PATH):
        try:
            with open(TOPIC_METADATA_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                benchmark_metrics = loaded.get("benchmark_comparison", {})
        except Exception:
            pass

    return TopicMetadataResponse(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        architecture="CPU-Optimized Embeddings + UMAP + HDBSCAN + c-TF-IDF",
        device="cpu",
        vram_usage_mb=0.0,
        embedding_dimension=384,
        min_reviews_required=15,
        local_weights_verified=topic_service.is_ready(),
        benchmark_metrics=benchmark_metrics,
    )
