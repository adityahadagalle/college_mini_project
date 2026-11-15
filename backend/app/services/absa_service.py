"""Production Aspect-Based Sentiment Analysis (ABSA) Service.

Combines Stage 1 (Aspect Term Extraction) with Stage 2 (Aspect-Conditioned Cross-Encoder Sentiment).
Thread-safe singleton with FP16/CPU device management and database persistence.
"""

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from app.models.database import ReviewAspect
from app.schemas.absa import (
    ABSABatchResponse,
    ABSAResponse,
    AspectAnalyticsResponse,
    AspectFrequencyItem,
    AspectResult,
)
from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

logger = logging.getLogger("absa_service")
logger.setLevel(logging.INFO)


class ABSAService:
    """Thread-safe singleton service for two-stage Aspect-Based Sentiment Analysis."""

    _instance: Optional["ABSAService"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ABSAService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = self.device == "cuda"
        precision_str = "FP16 (torch.float16)" if self.use_fp16 else "FP32 (torch.float32)"
        logger.info(f"ABSA Service configured on device: {self.device} with precision: {precision_str}")

        self.model_name = "DeBERTa-v3-ABSA-Modular"
        self.model_version = "v1.0.0"

        # Model storage paths: prioritize local disk artifacts
        self.local_ate_dir = "models/absa/ate"
        self.local_alsc_dir = "models/absa/alsc"
        self.remote_ate_id = "yangheng/deberta-v3-base-end2end-absa"
        self.remote_alsc_id = "yangheng/deberta-v3-base-absa-v1.1"

        self.ate_tokenizer = None
        self.ate_model = None
        self.alsc_tokenizer = None
        self.alsc_model = None

        self._load_lock = threading.Lock()
        self._load_models()
        self._initialized = True

    def is_ready(self) -> bool:
        """Check if neural ATE and ALSC models and tokenizers are loaded and ready."""
        return (
            self.ate_model is not None
            and self.ate_tokenizer is not None
            and self.alsc_model is not None
            and self.alsc_tokenizer is not None
        )

    def _load_models(self) -> None:
        """Load ATE and ALSC models from local disk with VRAM safeguards, falling back to HF Hub if missing."""
        with self._load_lock:
            try:
                # 1. Load ATE Token Classifier (Local first)
                if os.path.isdir(self.local_ate_dir) and os.path.exists(os.path.join(self.local_ate_dir, "config.json")):
                    logger.info(f"Loading ATE model from local disk: {self.local_ate_dir}")
                    self.ate_tokenizer = AutoTokenizer.from_pretrained(self.local_ate_dir, local_files_only=True)
                    self.ate_model = AutoModelForTokenClassification.from_pretrained(self.local_ate_dir, local_files_only=True)
                else:
                    logger.info(f"Local ATE directory not found. Fetching from remote HF Hub: {self.remote_ate_id}")
                    self.ate_tokenizer = AutoTokenizer.from_pretrained(self.remote_ate_id)
                    self.ate_model = AutoModelForTokenClassification.from_pretrained(self.remote_ate_id)

                self.ate_model.to(self.device)
                if self.use_fp16:
                    self.ate_model.half()
                self.ate_model.eval()

                # 2. Load ALSC Aspect-Conditioned Cross-Encoder (Local first)
                if os.path.isdir(self.local_alsc_dir) and os.path.exists(os.path.join(self.local_alsc_dir, "config.json")):
                    logger.info(f"Loading ALSC model from local disk: {self.local_alsc_dir}")
                    self.alsc_tokenizer = AutoTokenizer.from_pretrained(self.local_alsc_dir, local_files_only=True)
                    self.alsc_model = AutoModelForSequenceClassification.from_pretrained(self.local_alsc_dir, local_files_only=True)
                else:
                    logger.info(f"Local ALSC directory not found. Fetching from remote HF Hub: {self.remote_alsc_id}")
                    self.alsc_tokenizer = AutoTokenizer.from_pretrained(self.remote_alsc_id)
                    self.alsc_model = AutoModelForSequenceClassification.from_pretrained(self.remote_alsc_id)

                self.alsc_model.to(self.device)
                if self.use_fp16:
                    self.alsc_model.half()
                self.alsc_model.eval()

                logger.info("ABSA two-stage neural pipeline loaded successfully from local artifacts.")
            except Exception as exc:
                logger.error(f"Failed to load ABSA neural models: {exc}")
                self.ate_model = None
                self.ate_tokenizer = None
                self.alsc_model = None
                self.alsc_tokenizer = None

    def _extract_aspect_spans(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract explicit aspect terms with start and end character offsets via neural ATE token classification.

        Handles single words, multiword phrases, repeated aspect terms, and contractions.
        Returns empty list if no explicit aspects are predicted by the model.
        """
        if not text or not text.strip():
            return []

        if not self.is_ready():
            raise RuntimeError("ABSA ATE neural model is not loaded or ready.")

        encoding = self.ate_tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        offsets = encoding["offset_mapping"][0].numpy()
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.ate_model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=2)[0].cpu().numpy()

        raw_spans: List[Tuple[int, int]] = []
        curr_start: Optional[int] = None
        curr_end: Optional[int] = None

        for p, (start, end) in zip(preds, offsets):
            if start == end:
                continue
            label = self.ate_model.config.id2label[p]
            if label.startswith("B-ASP"):
                if curr_start is not None and curr_end is not None:
                    raw_spans.append((curr_start, curr_end))
                curr_start = int(start)
                curr_end = int(end)
            elif label.startswith("I-ASP") and curr_start is not None:
                curr_end = int(end)
            else:
                if curr_start is not None and curr_end is not None:
                    raw_spans.append((curr_start, curr_end))
                    curr_start = None
                    curr_end = None

        if curr_start is not None and curr_end is not None:
            raw_spans.append((curr_start, curr_end))

        extracted_spans: List[Tuple[str, int, int]] = []
        for s, e in raw_spans:
            raw_sub = text[s:e]
            clean_sub = raw_sub.strip(" \t\n\r,.-!?:;\"'()[]{}")
            if clean_sub and len(clean_sub) > 1:
                lead = raw_sub.find(clean_sub)
                real_s = s + lead
                real_e = real_s + len(clean_sub)
                extracted_spans.append((clean_sub, real_s, real_e))

        # Deduplicate identical spans (preserving separate occurrences at different offsets)
        deduped = []
        seen = set()
        for clean_term, s, e in extracted_spans:
            if clean_term and (clean_term, s, e) not in seen:
                seen.add((clean_term, s, e))
                deduped.append((clean_term, s, e))

        return deduped

    def _classify_aspect_sentiment(
        self, aspect: str, text: str
    ) -> Tuple[str, float, Dict[str, float]]:
        """Classify aspect sentiment conditioned on sentence context [CLS] aspect [SEP] text [SEP] via neural ALSC cross-encoder."""
        if not self.is_ready():
            raise RuntimeError("ABSA ALSC neural model is not loaded or ready.")

        inputs = self.alsc_tokenizer(
            aspect,
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.alsc_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Model config id2label: {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        prob_neg = float(probs[0])
        prob_neu = float(probs[1])
        prob_pos = float(probs[2])

        probs_dict = {
            "POSITIVE": round(prob_pos, 4),
            "NEUTRAL": round(prob_neu, 4),
            "NEGATIVE": round(prob_neg, 4),
        }

        pred_idx = int(np.argmax(probs))
        sentiment_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
        sentiment = sentiment_map.get(pred_idx, "NEUTRAL")
        confidence = float(np.max(probs))

        return sentiment, round(confidence, 4), probs_dict

    async def analyze(
        self,
        text: str,
        prediction_id: Optional[str] = None,
        save_to_db: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> ABSAResponse:
        """Run complete two-stage ABSA analysis on a single text string."""
        start_time = time.perf_counter()

        if not self.is_ready():
            raise RuntimeError("ABSA neural models are unavailable or not loaded.")

        if not text or not text.strip():
            return ABSAResponse(
                text=text,
                aspects=[],
                aspect_count=0,
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        # 1. Extract Spans
        spans = self._extract_aspect_spans(text)

        aspect_results: List[AspectResult] = []
        db_entities: List[ReviewAspect] = []

        # 2. Classify each aspect independently
        for term, start_char, end_char in spans:
            sentiment, confidence, probs = self._classify_aspect_sentiment(term, text)
            evidence_text = text[start_char:end_char]

            res = AspectResult(
                aspect=term,
                aspect_category=self._categorize_aspect(term),
                is_implicit=False,
                start_char=start_char,
                end_char=end_char,
                sentiment=sentiment,
                confidence=confidence,
                probabilities=probs,
                evidence_text=evidence_text,
                model_name=self.model_name,
                model_version=self.model_version,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            aspect_results.append(res)

            if save_to_db and db is not None:
                entity = ReviewAspect(
                    prediction_id=prediction_id,
                    aspect_term=term,
                    aspect_category=res.aspect_category,
                    is_implicit=False,
                    start_char=start_char,
                    end_char=end_char,
                    sentiment_label=sentiment,
                    confidence_score=confidence,
                    positive_prob=probs["POSITIVE"],
                    neutral_prob=probs["NEUTRAL"],
                    negative_prob=probs["NEGATIVE"],
                    evidence_text=evidence_text,
                    model_name=self.model_name,
                    model_version=self.model_version,
                )
                db_entities.append(entity)

        if save_to_db and db is not None and db_entities:
            for entity in db_entities:
                db.add(entity)
            await db.commit()

        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        return ABSAResponse(
            text=text,
            aspects=aspect_results,
            aspect_count=len(aspect_results),
            latency_ms=latency_ms,
        )

    async def analyze_batch(
        self,
        texts: List[str],
        save_to_db: bool = False,
        db: Optional[AsyncSession] = None,
    ) -> ABSABatchResponse:
        """Process batch of texts with micro-batching."""
        start_time = time.perf_counter()
        results: List[ABSAResponse] = []
        total_aspects = 0

        for t in texts:
            resp = await self.analyze(t, save_to_db=save_to_db, db=db)
            results.append(resp)
            total_aspects += resp.aspect_count

        total_latency = round((time.perf_counter() - start_time) * 1000.0, 2)
        return ABSABatchResponse(
            results=results,
            total_samples=len(texts),
            total_aspects=total_aspects,
            latency_ms=total_latency,
        )

    @staticmethod
    def _categorize_aspect(aspect: str) -> Optional[str]:
        """Deterministic rule-based category normalization tag (e.g. CAMERA, BATTERY, DISPLAY).

        Note: Heuristic dictionary mapping used for grouping analytics; not a neural classification output.
        """
        asp = aspect.lower()
        if any(w in asp for w in ["battery", "charging", "charge", "power"]):
            return "BATTERY"
        elif any(w in asp for w in ["screen", "display", "monitor", "resolution"]):
            return "DISPLAY"
        elif any(w in asp for w in ["camera", "lens", "photo", "video"]):
            return "CAMERA"
        elif any(w in asp for w in ["support", "service", "staff", "rep", "agent"]):
            return "CUSTOMER_SERVICE"
        elif any(w in asp for w in ["price", "cost", "pricing", "expensive", "cheap", "value"]):
            return "PRICING"
        elif any(w in asp for w in ["speed", "performance", "fast", "slow", "processor", "ram", "cpu"]):
            return "PERFORMANCE"
        elif any(w in asp for w in ["build", "quality", "chassis", "design", "keyboard", "trackpad", "touchpad"]):
            return "DESIGN_BUILD"
        elif any(w in asp for w in ["food", "dish", "meal", "taste", "menu", "drink"]):
            return "FOOD_BEVERAGE"
        return "GENERAL"

    async def get_aspect_analytics(
        self, db: AsyncSession, limit: int = 50
    ) -> AspectAnalyticsResponse:
        """Fetch aggregated aspect frequency and polarity analytics from review_aspects table."""
        # 1. Total counts
        total_stmt = select(func.count(ReviewAspect.id))
        total_res = await db.execute(total_stmt)
        total_aspects = total_res.scalar() or 0

        # 2. Polarity distribution
        pol_stmt = select(
            ReviewAspect.sentiment_label, func.count(ReviewAspect.id)
        ).group_by(ReviewAspect.sentiment_label)
        pol_res = await db.execute(pol_stmt)
        pol_map = {row[0]: row[1] for row in pol_res.all()}

        pos_count = pol_map.get("POSITIVE", 0)
        neu_count = pol_map.get("NEUTRAL", 0)
        neg_count = pol_map.get("NEGATIVE", 0)

        sentiment_dist = {
            "POSITIVE": pos_count,
            "NEUTRAL": neu_count,
            "NEGATIVE": neg_count,
        }
        total_counted = pos_count + neu_count + neg_count
        sentiment_ratios = {
            "POSITIVE": round(pos_count / max(total_counted, 1), 4),
            "NEUTRAL": round(neu_count / max(total_counted, 1), 4),
            "NEGATIVE": round(neg_count / max(total_counted, 1), 4),
        }

        # 3. Frequency per aspect term
        term_stmt = (
            select(
                ReviewAspect.aspect_term,
                func.count(ReviewAspect.id).label("total_cnt"),
                func.sum(case((ReviewAspect.sentiment_label == "POSITIVE", 1), else_=0)).label("pos_cnt"),
                func.sum(case((ReviewAspect.sentiment_label == "NEUTRAL", 1), else_=0)).label("neu_cnt"),
                func.sum(case((ReviewAspect.sentiment_label == "NEGATIVE", 1), else_=0)).label("neg_cnt"),
            )
            .group_by(ReviewAspect.aspect_term)
            .order_by(desc("total_cnt"))
            .limit(limit)
        )

        term_res = await db.execute(term_stmt)
        rows = term_res.all()
        unique_aspects = len(rows)

        items: List[AspectFrequencyItem] = []
        for term, total_cnt, p_cnt, n_cnt, neg_cnt in rows:
            p_val = int(p_cnt or 0)
            neu_val = int(n_cnt or 0)
            neg_val = int(neg_cnt or 0)
            t_val = int(total_cnt or 0)
            net_score = round((p_val - neg_val) / max(t_val, 1), 3)

            items.append(
                AspectFrequencyItem(
                    aspect=term,
                    count=t_val,
                    positive_count=p_val,
                    neutral_count=neu_val,
                    negative_count=neg_val,
                    net_sentiment_score=net_score,
                )
            )

        # Sort top positive and top negative
        top_positive = sorted(
            [item for item in items if item.positive_count > 0],
            key=lambda x: (x.positive_count, x.net_sentiment_score),
            reverse=True,
        )[:10]

        top_negative = sorted(
            [item for item in items if item.negative_count > 0],
            key=lambda x: (x.negative_count, -x.net_sentiment_score),
            reverse=True,
        )[:10]

        top_neutral = sorted(
            [item for item in items if item.neutral_count > 0],
            key=lambda x: x.neutral_count,
            reverse=True,
        )[:10]

        # 4. Recent aspect extractions
        recent_stmt = (
            select(ReviewAspect)
            .order_by(desc(ReviewAspect.created_at))
            .limit(20)
        )
        recent_res = await db.execute(recent_stmt)
        recent_entities = recent_res.scalars().all()

        recent_records = [
            {
                "id": e.id,
                "prediction_id": e.prediction_id,
                "aspect": e.aspect_term,
                "category": e.aspect_category,
                "sentiment": e.sentiment_label,
                "confidence": e.confidence_score,
                "start_char": e.start_char,
                "end_char": e.end_char,
                "evidence_text": e.evidence_text,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in recent_entities
        ]

        return AspectAnalyticsResponse(
            total_aspects=total_aspects,
            unique_aspects_count=unique_aspects,
            sentiment_distribution=sentiment_dist,
            sentiment_ratios=sentiment_ratios,
            top_positive_aspects=top_positive,
            top_negative_aspects=top_negative,
            top_neutral_aspects=top_neutral,
            recent_aspect_records=recent_records,
        )


# Global singleton instance
absa_service = ABSAService()
