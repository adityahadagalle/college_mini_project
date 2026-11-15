import csv
import io
from datetime import datetime
from typing import List, Optional

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import Prediction, User
from app.schemas.sentiment import PredictResponse
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.get("", response_model=List[PredictResponse])  # type: ignore[misc]
async def get_history(
    search: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    sentiment_filter: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[PredictResponse]:
    query = select(Prediction).where(Prediction.user_id == current_user.id)

    if search:
        query = query.where(Prediction.text_content.ilike(f"%{search}%"))

    filter_val = sentiment or sentiment_filter
    if filter_val:
        query = query.where(Prediction.sentiment_label == filter_val.upper())

    query = query.order_by(Prediction.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    predictions = result.scalars().all()

    return [
        PredictResponse(
            id=p.id,
            text_content=p.text_content,
            sentiment=p.sentiment_label,
            confidence=p.confidence_score,
            probabilities={
                "POSITIVE": p.positive_prob,
                "NEUTRAL": p.neutral_prob,
                "NEGATIVE": p.negative_prob,
            },
            latency_ms=p.latency_ms,
            created_at=str(p.created_at) if p.created_at else None,
        )
        for p in predictions
    ]


@router.get("/export")  # type: ignore[misc]
async def export_history_csv(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    result = await db.execute(
        select(Prediction)
        .where(Prediction.user_id == current_user.id)
        .order_by(Prediction.created_at.desc())
    )
    predictions = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "timestamp", "text_content", "sentiment", "confidence"])

    for p in predictions:
        writer.writerow(
            [
                p.id,
                p.created_at.isoformat() if p.created_at else "",
                p.text_content,
                p.sentiment_label,
                p.confidence_score,
            ]
        )

    filename = f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return Response(content=output.getvalue(), media_type="text/csv", headers=headers)
