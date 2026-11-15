"""Aspect-Based Sentiment Analysis (ABSA) REST API Endpoints."""

import json
import os
from typing import Any, Dict

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import User
from app.schemas.absa import (
    ABSABatchRequest,
    ABSABatchResponse,
    ABSARequest,
    ABSAResponse,
    AspectAnalyticsResponse,
)
from app.services.absa_service import absa_service
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/analyze", response_model=ABSAResponse)
async def analyze_aspects(
    payload: ABSARequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ABSAResponse:
    """Analyze single review text for explicit aspect terms and individual aspect sentiments."""
    if not absa_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ABSA neural models are unavailable or not loaded",
        )

    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input review text cannot be empty or whitespace.",
        )

    try:
        response = await absa_service.analyze(
            text=payload.text,
            prediction_id=payload.prediction_id,
            save_to_db=payload.save_to_db,
            db=db,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ABSA extraction failed: {str(exc)}",
        ) from exc


@router.post("/batch", response_model=ABSABatchResponse)
async def analyze_aspects_batch(
    payload: ABSABatchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ABSABatchResponse:
    """Analyze multiple reviews for aspects with micro-batching."""
    if not absa_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ABSA neural models are unavailable or not loaded",
        )

    if not payload.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input batch list cannot be empty.",
        )

    try:
        response = await absa_service.analyze_batch(
            texts=payload.texts,
            save_to_db=payload.save_to_db,
            db=db,
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ABSA processing error: {str(exc)}",
        ) from exc


@router.get("/analytics", response_model=AspectAnalyticsResponse)
async def get_aspect_analytics(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AspectAnalyticsResponse:
    """Fetch aggregated aspect frequency and polarity analytics across recorded predictions."""
    try:
        analytics = await absa_service.get_aspect_analytics(db=db, limit=limit)
        return analytics
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate aspect analytics: {str(exc)}",
        ) from exc


@router.get("/metadata")
async def get_absa_metadata() -> Dict[str, Any]:
    """Expose evaluated benchmarks, hyperparameters, and versioning for ABSA models."""
    meta_path = "models/absa/absa_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "model_name": "DeBERTa-v3-ABSA-Modular",
        "version": "v1.0.0",
        "status": "ACTIVE",
        "evaluation_benchmark": "MAMS-ATSA",
    }
