"""Emotion Detection REST API Endpoints."""

import json
import os
from typing import Any, Dict

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import User
from app.schemas.emotion import (
    EmotionAnalyticsResponse,
    EmotionBatchRequest,
    EmotionBatchResponse,
    EmotionRequest,
    EmotionResponse,
)
from app.services.emotion_service import emotion_service
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/analyze", response_model=EmotionResponse)
async def analyze_emotion(
    payload: EmotionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EmotionResponse:
    """Analyze single review text for 28 GoEmotions labels and 6-Category CX Taxonomy rollup."""
    if not emotion_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion neural models are unavailable or not loaded",
        )

    if not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input review text cannot be empty or whitespace.",
        )

    try:
        response = await emotion_service.analyze(
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
            detail=f"Emotion analysis failed: {str(exc)}",
        ) from exc


@router.post("/batch", response_model=EmotionBatchResponse)
async def analyze_emotion_batch(
    payload: EmotionBatchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EmotionBatchResponse:
    """Analyze multiple review texts for emotions with micro-batching."""
    if not emotion_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion neural models are unavailable or not loaded",
        )

    if not payload.texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input batch list cannot be empty.",
        )

    try:
        response = await emotion_service.analyze_batch(
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
            detail=f"Batch emotion analysis error: {str(exc)}",
        ) from exc


@router.get("/analytics", response_model=EmotionAnalyticsResponse)
async def get_emotion_analytics(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> EmotionAnalyticsResponse:
    """Fetch aggregated CX emotion distribution, ratios, and top raw GoEmotions labels."""
    try:
        analytics = await emotion_service.get_analytics(db=db, limit=limit)
        return analytics
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate emotion analytics: {str(exc)}",
        ) from exc


@router.get("/metadata")
async def get_emotion_metadata() -> Dict[str, Any]:
    """Expose evaluated benchmarks, calibrated thresholds, and versioning for Emotion Detection."""
    meta_path = "models/emotion/emotion_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "model_name": "RoBERTa-GoEmotions-MultiLabel",
        "version": "v1.0.0",
        "status": "ACTIVE",
        "benchmark_dataset": "Google Research GoEmotions Test Split",
    }
