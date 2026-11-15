"""Unified Multi-Capability Intelligence and Cross-Feature Analytics REST APIs."""

import asyncio
from datetime import datetime
import os
from typing import Any, Dict, List, Optional
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.auth import get_current_user
from app.core.database import AsyncSessionLocal, get_db
from app.models.database import User
from app.schemas.intelligence import (
    EvidenceDrilldownResponse,
    IntelligenceAnalyticsResponse,
    IntelligenceAnalyzeRequest,
    IntelligenceBatchJobStatus,
    IntelligenceBatchRequest,
    UnifiedReviewResponse,
)
from app.services.intelligence_service import intelligence_service

router = APIRouter()

MAX_BATCH_SIZE = 500


async def _run_batch_background(
    job_id: str,
    reviews: List[str],
    user_id: Optional[str] = None,
    save_to_db: bool = True,
):
    """Background task runner for asynchronous batch multi-task intelligence."""
    async with AsyncSessionLocal() as session:
        try:
            intelligence_service.update_job_progress(
                job_id=job_id,
                processed_records=0,
                progress_pct=0.0,
                status="PROCESSING",
            )
            results = await intelligence_service.process_batch(
                session=session,
                reviews=reviews,
                user_id=user_id,
                save_to_db=save_to_db,
                job_id=job_id,
            )

            # Analyze counts
            succeeded = 0
            failed = 0
            partial = 0
            for r in results:
                statuses = [v.status for v in r.capability_status.values()]
                if all(s == "completed" for s in statuses):
                    succeeded += 1
                elif any(s == "completed" for s in statuses):
                    partial += 1
                else:
                    failed += 1

            intelligence_service.update_job_progress(
                job_id=job_id,
                processed_records=len(reviews),
                progress_pct=100.0,
                status="COMPLETED",
                succeeded=succeeded,
                failed=failed,
                partial=partial,
            )
        except Exception as e:
            intelligence_service.update_job_progress(
                job_id=job_id,
                processed_records=0,
                progress_pct=0.0,
                status="FAILED",
                error_message=str(e),
            )


@router.post("/analyze", response_model=UnifiedReviewResponse)
async def analyze_unified_review(
    payload: IntelligenceAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UnifiedReviewResponse:
    """Analyze a single review across Overall Sentiment, ABSA, Emotion Detection, and Topic Projection."""
    cleaned = payload.text.strip()
    if len(cleaned) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Review text must be at least 3 characters long.",
        )

    try:
        response = await intelligence_service.analyze_review(
            session=db,
            text=cleaned,
            user_id=current_user.id,
            save_to_db=payload.save_to_db,
            project_topic=payload.project_topic,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unified analysis failed: {str(exc)}",
        ) from exc


@router.post("/batch", response_model=IntelligenceBatchJobStatus)
async def process_batch_reviews(
    payload: IntelligenceBatchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IntelligenceBatchJobStatus:
    """Execute unified multi-task intelligence over a batch of reviews."""
    if not payload.reviews:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch reviews list cannot be empty.",
        )

    if len(payload.reviews) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(payload.reviews)} exceeds maximum allowed limit of {MAX_BATCH_SIZE}.",
        )

    job_id = str(uuid.uuid4())
    now_iso = datetime.utcnow().isoformat()

    # If small batch (<= 25) and sync=True, execute synchronously
    if payload.sync and len(payload.reviews) <= 25:
        results = await intelligence_service.process_batch(
            session=db,
            reviews=payload.reviews,
            user_id=current_user.id,
            save_to_db=payload.save_to_db,
        )

        succeeded = 0
        failed = 0
        partial = 0
        for r in results:
            statuses = [v.status for v in r.capability_status.values()]
            if all(s == "completed" for s in statuses):
                succeeded += 1
            elif any(s == "completed" for s in statuses):
                partial += 1
            else:
                failed += 1

        status_obj = IntelligenceBatchJobStatus(
            job_id=job_id,
            status="COMPLETED",
            progress_pct=100.0,
            total_records=len(payload.reviews),
            processed_records=len(payload.reviews),
            succeeded_records=succeeded,
            failed_records=failed,
            partial_records=partial,
            created_at=now_iso,
            completed_at=datetime.utcnow().isoformat(),
            results=results,
        )
        intelligence_service._in_memory_jobs[job_id] = status_obj.model_dump()
        return status_obj

    # Asynchronous background execution
    status_obj = IntelligenceBatchJobStatus(
        job_id=job_id,
        status="PENDING",
        progress_pct=0.0,
        total_records=len(payload.reviews),
        processed_records=0,
        succeeded_records=0,
        failed_records=0,
        partial_records=0,
        created_at=now_iso,
        completed_at=None,
        results=None,
    )
    intelligence_service._in_memory_jobs[job_id] = status_obj.model_dump()

    background_tasks.add_task(
        _run_batch_background,
        job_id=job_id,
        reviews=payload.reviews,
        user_id=current_user.id,
        save_to_db=payload.save_to_db,
    )

    return status_obj


@router.get("/batch/{job_id}", response_model=IntelligenceBatchJobStatus)
async def get_batch_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> IntelligenceBatchJobStatus:
    """Fetch real-time progress and completion status of an asynchronous batch job."""
    status_obj = intelligence_service.get_batch_job(job_id)
    if not status_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Intelligence batch job '{job_id}' not found.",
        )
    return status_obj


@router.get("/analytics", response_model=IntelligenceAnalyticsResponse)
async def get_cross_feature_analytics(
    min_sample_size: int = Query(
        1, ge=1, le=1000, description="Minimum co-occurrence sample count threshold"
    ),
    start_date: Optional[str] = Query(None, description="Start date ISO filter"),
    end_date: Optional[str] = Query(None, description="End date ISO filter"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IntelligenceAnalyticsResponse:
    """Retrieve SQL-aggregated cross-feature analytics across topics, aspects, emotions, and sentiment."""
    try:
        return await intelligence_service.get_cross_feature_analytics(
            session=db,
            min_sample_size=min_sample_size,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute cross-feature analytics: {str(exc)}",
        ) from exc


@router.get("/evidence", response_model=EvidenceDrilldownResponse)
async def get_evidence_drilldown(
    topic_id: Optional[str] = Query(None, description="Filter by discovered topic ID"),
    aspect_term: Optional[str] = Query(None, description="Filter by aspect term substring"),
    emotion_category: Optional[str] = Query(None, description="Filter by CX emotion category"),
    sentiment: Optional[str] = Query(None, description="Filter by overall sentiment (POSITIVE, NEGATIVE, NEUTRAL)"),
    page: int = Query(1, ge=1, description="Page index (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EvidenceDrilldownResponse:
    """Retrieve actual review evidence and model outputs matching drilldown criteria."""
    try:
        return await intelligence_service.get_evidence_drilldown(
            session=db,
            topic_id=topic_id,
            aspect_term=aspect_term,
            emotion_category=emotion_category,
            sentiment=sentiment,
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query evidence drilldown: {str(exc)}",
        ) from exc
