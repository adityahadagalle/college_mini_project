import uuid

from app.api.v1.auth import get_current_user
from app.core.database import get_db
from app.models.database import Prediction, User
from app.schemas.sentiment import ExplainabilityToken, PredictRequest, PredictResponse
from app.services.tf_inference import tf_engine
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("", response_model=PredictResponse)  # type: ignore[misc]
async def predict_sentiment(
    payload: PredictRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    if not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input text cannot be empty or whitespace only.",
        )

    res = tf_engine.predict_single(payload.text)

    probs = res.get("probabilities", {})
    pred_entity = Prediction(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        text_content=payload.text[:5000],
        sentiment_label=res["sentiment"],
        confidence_score=res["confidence"],
        positive_prob=probs.get("POSITIVE", probs.get("positive", 0.0)),
        neutral_prob=probs.get("NEUTRAL", probs.get("neutral", 0.0)),
        negative_prob=probs.get("NEGATIVE", probs.get("negative", 0.0)),
        latency_ms=res["latency_ms"],
        explainability_json=res.get("explainability", []),
    )

    db.add(pred_entity)
    await db.commit()
    await db.refresh(pred_entity)

    exp_tokens = [ExplainabilityToken(**item) for item in res.get("explainability", [])]

    return PredictResponse(
        id=pred_entity.id,
        text_content=pred_entity.text_content,
        sentiment=pred_entity.sentiment_label,
        confidence=pred_entity.confidence_score,
        probabilities={
            "POSITIVE": pred_entity.positive_prob,
            "NEUTRAL": pred_entity.neutral_prob,
            "NEGATIVE": pred_entity.negative_prob,
        },
        latency_ms=pred_entity.latency_ms,
        explainability=exp_tokens,
        created_at=str(pred_entity.created_at) if pred_entity.created_at else None,
    )
