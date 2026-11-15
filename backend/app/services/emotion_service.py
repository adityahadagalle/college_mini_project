"""Emotion Detection Service.

Provides:
1. Singleton neural inference engine backed by local RoBERTa weights in `models/emotion/`.
2. Multi-label Sigmoid prediction over 28 GoEmotions classes with per-class calibrated thresholds.
3. 6-Category Customer Experience (CX) Taxonomy aggregation.
4. Database persistence and real SQL analytics queries.
5. Strict fail-fast readiness checking (HTTP 503 if weights are unavailable).
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.datasets.emotion_dataset import (
    CX_CATEGORIES,
    CX_TAXONOMY_MAP,
    DEFAULT_CALIBRATED_THRESHOLDS,
    GO_EMOTIONS_LABELS,
    map_emotions_to_cx_scores,
)
from app.models.database import ReviewEmotion
from app.schemas.emotion import (
    DetectedCXCategory,
    DetectedEmotionItem,
    EmotionAnalyticsResponse,
    EmotionBatchResponse,
    EmotionFrequencyItem,
    EmotionResponse,
)

logger = logging.getLogger(__name__)

MODEL_LOCAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/emotion"))
METADATA_LOCAL_PATH = os.path.join(MODEL_LOCAL_DIR, "emotion_metadata.json")


class EmotionDetectionService:
    """Production-grade multi-label emotion detection service."""

    def __init__(self, model_dir: str = MODEL_LOCAL_DIR):
        self.model_dir = model_dir
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = torch.float16 if self.device.type == "cuda" else torch.float32
        self.thresholds: Dict[str, float] = DEFAULT_CALIBRATED_THRESHOLDS.copy()
        self.metadata: Dict[str, Any] = {}
        self.model_name = "RoBERTa-GoEmotions-MultiLabel"
        self.model_version = "v1.0.0"

        self._load_metadata()
        self._load_local_weights()

    def _load_metadata(self) -> None:
        """Load versioned calibration metadata from local JSON file if available."""
        if os.path.isfile(METADATA_LOCAL_PATH):
            try:
                with open(METADATA_LOCAL_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    cal_th = self.metadata.get("calibrated_thresholds")
                    if isinstance(cal_th, dict):
                        self.thresholds.update(cal_th)
                    self.model_name = self.metadata.get("model_name", self.model_name)
                    self.model_version = self.metadata.get("version", self.model_version)
                    logger.info("Loaded emotion calibration thresholds from emotion_metadata.json")
            except Exception as e:
                logger.warning(f"Could not read emotion metadata at {METADATA_LOCAL_PATH}: {e}")

    def _load_local_weights(self) -> None:
        """Load transformer weights and tokenizer from local directory with local_files_only=True."""
        if not os.path.isdir(self.model_dir):
            logger.warning(f"Emotion model directory {self.model_dir} not found. Service offline.")
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading local emotion model from {self.model_dir} (device: {self.device}, precision: {self.precision})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                local_files_only=True,
                torch_dtype=self.precision,
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Emotion Detection model loaded successfully into memory.")
        except Exception as e:
            logger.error(f"Failed to load emotion model weights from {self.model_dir}: {e}")
            self.model = None
            self.tokenizer = None

    def is_ready(self) -> bool:
        """Return True only if local model and tokenizer are initialized and ready."""
        return self.model is not None and self.tokenizer is not None

    async def analyze(
        self,
        text: str,
        prediction_id: Optional[str] = None,
        save_to_db: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> EmotionResponse:
        """Run single-review neural multi-label emotion classification."""
        start_t = time.perf_counter()

        if not self.is_ready():
            raise RuntimeError("Emotion neural models are unavailable or not loaded.")

        clean_text = text.strip()
        if not clean_text:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            return EmotionResponse(
                text=text,
                primary_emotion="Neutral / Informative",
                confidence=0.0,
                detected_cx_categories=[],
                detected_emotions=[],
                raw_scores={label: 0.0 for label in GO_EMOTIONS_LABELS},
                model_name=self.model_name,
                model_version=self.model_version,
                latency_ms=round(latency_ms, 2),
            )

        # Tokenize and run forward pass
        enc = self.tokenizer(
            clean_text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        raw_scores: Dict[str, float] = {
            GO_EMOTIONS_LABELS[i]: round(float(probs[i]), 4)
            for i in range(len(GO_EMOTIONS_LABELS))
        }

        # Apply per-class calibrated thresholds
        detected_emotions: List[DetectedEmotionItem] = []
        for i, label in enumerate(GO_EMOTIONS_LABELS):
            prob = float(probs[i])
            th = self.thresholds.get(label, 0.35)
            if prob >= th:
                detected_emotions.append(
                    DetectedEmotionItem(
                        label=label,
                        probability=round(prob, 4),
                        threshold=round(th, 4),
                        cx_category=CX_TAXONOMY_MAP.get(label, "Neutral / Informative"),
                    )
                )

        # Sort active emotions by probability descending
        detected_emotions.sort(key=lambda x: x.probability, reverse=True)

        # Map to 6-Category CX Taxonomy
        cx_breakdown = map_emotions_to_cx_scores(raw_scores, self.thresholds)
        detected_cx: List[DetectedCXCategory] = []
        for cat_name, cat_data in cx_breakdown.items():
            if cat_data["is_active"]:
                th = min([e["threshold"] for e in cat_data["active_emotions"]], default=0.35)
                detected_cx.append(
                    DetectedCXCategory(
                        category=cat_name,
                        score=cat_data["score"],
                        threshold=round(th, 4),
                    )
                )

        detected_cx.sort(key=lambda x: x.score, reverse=True)

        # Determine primary emotion and confidence
        if detected_cx:
            primary_emotion = detected_cx[0].category
            confidence = detected_cx[0].score
        elif detected_emotions:
            primary_emotion = detected_emotions[0].cx_category
            confidence = detected_emotions[0].probability
        else:
            # Fallback to Neutral / Informative or top score
            top_prob = max(raw_scores.values())
            top_label = max(raw_scores, key=raw_scores.get)  # type: ignore[arg-type]
            if raw_scores.get("neutral", 0.0) >= 0.25:
                primary_emotion = "Neutral / Informative"
                confidence = raw_scores.get("neutral", 0.25)
            else:
                primary_emotion = CX_TAXONOMY_MAP.get(top_label, "Neutral / Informative")
                confidence = top_prob

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        resp = EmotionResponse(
            text=text,
            primary_emotion=primary_emotion,
            confidence=round(float(confidence), 4),
            detected_cx_categories=detected_cx,
            detected_emotions=detected_emotions,
            raw_scores=raw_scores,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=round(latency_ms, 2),
        )

        if save_to_db and db:
            await self.persist_emotion(db, resp, prediction_id)

        return resp

    async def analyze_batch(
        self,
        texts: List[str],
        save_to_db: bool = False,
        db: Optional[AsyncSession] = None,
        batch_size: int = 16,
    ) -> EmotionBatchResponse:
        """Run batched neural inference with chunking <= 16."""
        start_t = time.perf_counter()
        if not self.is_ready():
            raise RuntimeError("Emotion neural models are unavailable or not loaded.")

        results: List[EmotionResponse] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for t in chunk:
                res = await self.analyze(t, save_to_db=save_to_db, db=db)
                results.append(res)

        total_latency = (time.perf_counter() - start_t) * 1000.0
        return EmotionBatchResponse(
            results=results,
            total_samples=len(texts),
            latency_ms=round(total_latency, 2),
        )

    async def persist_emotion(
        self,
        db: AsyncSession,
        res: EmotionResponse,
        prediction_id: Optional[str] = None,
    ) -> ReviewEmotion:
        """Persist emotion detection result into database."""
        record = ReviewEmotion(
            prediction_id=prediction_id,
            primary_cx_category=res.primary_emotion,
            confidence_score=res.confidence,
            detected_emotions_json=[d.model_dump() for d in res.detected_emotions],
            raw_emotions_json=res.raw_scores,
            model_name=res.model_name,
            model_version=res.model_version,
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record

    async def get_analytics(self, db: AsyncSession, limit: int = 50) -> EmotionAnalyticsResponse:
        """Calculate real CX category distributions and raw emotion frequencies from DB."""
        # 1. Total records
        total_stmt = select(func.count(ReviewEmotion.id))
        total_records = (await db.execute(total_stmt)).scalar() or 0

        # 2. CX Category distribution
        cx_counts: Dict[str, int] = {cat: 0 for cat in CX_CATEGORIES}
        if total_records > 0:
            cx_stmt = (
                select(ReviewEmotion.primary_cx_category, func.count(ReviewEmotion.id))
                .group_by(ReviewEmotion.primary_cx_category)
            )
            for cat, count in (await db.execute(cx_stmt)).all():
                if cat in cx_counts:
                    cx_counts[cat] = count
                else:
                    cx_counts[cat] = count

        cx_ratios: Dict[str, float] = {
            cat: round(count / total_records, 4) if total_records > 0 else 0.0
            for cat, count in cx_counts.items()
        }

        # 3. Top raw emotions across recent records
        recent_stmt = (
            select(ReviewEmotion)
            .order_by(desc(ReviewEmotion.created_at))
            .limit(limit)
        )
        recent_rows = (await db.execute(recent_stmt)).scalars().all()

        raw_counter: Dict[str, int] = {l: 0 for l in GO_EMOTIONS_LABELS}
        recent_records_list: List[Dict[str, Any]] = []

        for r in recent_rows:
            raw_data = r.raw_emotions_json if isinstance(r.raw_emotions_json, dict) else {}
            detected_list = r.detected_emotions_json if isinstance(r.detected_emotions_json, list) else []

            for item in detected_list:
                lbl = item.get("label") if isinstance(item, dict) else None
                if lbl and lbl in raw_counter:
                    raw_counter[lbl] += 1

            recent_records_list.append({
                "id": r.id,
                "prediction_id": r.prediction_id,
                "primary_cx_category": r.primary_cx_category,
                "confidence_score": round(r.confidence_score, 4),
                "detected_emotions": detected_list,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })

        top_raw_emotions: List[EmotionFrequencyItem] = []
        for lbl, count in sorted(raw_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_raw_emotions.append(
                EmotionFrequencyItem(
                    label=lbl,
                    count=count,
                    ratio=round(count / max(total_records, 1), 4),
                    cx_category=CX_TAXONOMY_MAP.get(lbl, "Neutral / Informative"),
                )
            )

        return EmotionAnalyticsResponse(
            total_records=total_records,
            cx_distribution=cx_counts,
            cx_ratios=cx_ratios,
            top_raw_emotions=top_raw_emotions,
            recent_emotion_records=recent_records_list,
        )


# Singleton instance
emotion_service = EmotionDetectionService()
