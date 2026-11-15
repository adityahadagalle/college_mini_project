"""Unified Multi-Capability Orchestration Service.

Integrates:
1. Overall Sentiment (TensorFlow BiLSTM-SelfAttention)
2. Aspect-Based Sentiment Analysis (DeBERTa-v3 ATE + ALSC)
3. Emotion Detection (RoBERTa-GoEmotions Multi-Label)
4. Automatic Topic Discovery Projection (frozen-centroid cosine similarity)
5. Cross-Feature SQL Analytics and Evidence Drilldown
"""

import asyncio
from datetime import datetime
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import numpy as np
from sqlalchemy import case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    DiscoveredTopic,
    IntelligenceBatchJob,
    Prediction,
    ReviewAspect,
    ReviewEmotion,
    TopicAssignment,
    TopicDiscoveryRun,
)
from app.schemas.intelligence import (
    AspectSentimentByTopicItem,
    CapabilityStatusItem,
    EmotionByAspectCategoryItem,
    EmotionBySentimentItem,
    EvidenceDrilldownResponse,
    EvidenceItem,
    IntelligenceAnalyticsResponse,
    IntelligenceBatchJobStatus,
    NegativeAspectTrendItem,
    OverallSentimentResult,
    TopicSummaryItem,
    UnifiedAspectItem,
    UnifiedEmotionResult,
    UnifiedReviewResponse,
    UnifiedTopicProjection,
)
from app.services.absa_service import absa_service
from app.services.emotion_service import emotion_service
from app.services.tf_inference import tf_engine
from app.services.topic_service import topic_service

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.45


class IntelligenceService:
    """Singleton orchestration service for unified multi-task intelligence."""

    _instance: Optional["IntelligenceService"] = None

    def __init__(self):
        self._in_memory_jobs: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> "IntelligenceService":
        if cls._instance is None:
            cls._instance = IntelligenceService()
        return cls._instance

    async def _project_onto_latest_run(
        self, session: AsyncSession, text: str
    ) -> Tuple[Optional[UnifiedTopicProjection], Dict[str, Any]]:
        """Project review embedding against latest completed TopicDiscoveryRun centroids.

        Never triggers HDBSCAN. Projects using cosine similarity in 384-d semantic space.
        """
        # 1. Fetch latest completed discovery run
        stmt_run = (
            select(TopicDiscoveryRun)
            .where(TopicDiscoveryRun.status == "COMPLETED")
            .order_by(desc(TopicDiscoveryRun.created_at))
            .limit(1)
        )
        res_run = await session.execute(stmt_run)
        run_rec = res_run.scalar_one_or_none()
        if not run_rec:
            return None, {
                "reason": "No completed topic discovery run found. Run topic discovery on corpus first."
            }

        # 2. Fetch topics belonging to this run
        stmt_topics = (
            select(DiscoveredTopic)
            .where(DiscoveredTopic.run_id == run_rec.id)
            .order_by(DiscoveredTopic.topic_index)
        )
        res_topics = await session.execute(stmt_topics)
        topic_records = res_topics.scalars().all()
        if not topic_records:
            return None, {"reason": "No topics found in latest discovery run."}

        meaningful_topics = [
            t
            for t in topic_records
            if t.topic_index != -1 and not t.is_archived and t.centroid_json
        ]
        outlier_topic = next((t for t in topic_records if t.topic_index == -1), None)

        if not meaningful_topics:
            return (
                UnifiedTopicProjection(
                    topic_id=outlier_topic.id if outlier_topic else None,
                    topic_index=-1,
                    topic_label="Outliers / Unassigned",
                    similarity_score=0.0,
                    topic_run_version=run_rec.id,
                    is_outlier=True,
                ),
                {"run_id": run_rec.id},
            )

        # 3. Generate or retrieve embedding
        embs, _ = await topic_service.get_or_create_embeddings(session, [text])
        rev_emb = embs[0]  # (384,)

        # 4. Dot product with normalized centroids
        centroids = np.array(
            [t.centroid_json for t in meaningful_topics], dtype=np.float32
        )
        sims = np.dot(centroids, rev_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= DRIFT_THRESHOLD:
            matched_t = meaningful_topics[best_idx]
            return (
                UnifiedTopicProjection(
                    topic_id=matched_t.id,
                    topic_index=matched_t.topic_index,
                    topic_label=matched_t.human_label or matched_t.auto_label,
                    similarity_score=round(best_sim, 4),
                    topic_run_version=run_rec.id,
                    is_outlier=False,
                ),
                {"run_id": run_rec.id},
            )
        else:
            return (
                UnifiedTopicProjection(
                    topic_id=outlier_topic.id if outlier_topic else None,
                    topic_index=-1,
                    topic_label="Outliers / Unassigned",
                    similarity_score=round(best_sim, 4),
                    topic_run_version=run_rec.id,
                    is_outlier=True,
                ),
                {"run_id": run_rec.id},
            )

    async def analyze_review(
        self,
        session: AsyncSession,
        text: str,
        user_id: Optional[str] = None,
        save_to_db: bool = True,
        project_topic: bool = True,
    ) -> UnifiedReviewResponse:
        """Unified multi-task inference for a single review with strict fault isolation."""
        t0 = time.time()
        review_id = str(uuid.uuid4())

        capability_status: Dict[str, CapabilityStatusItem] = {
            "sentiment": CapabilityStatusItem(status="pending"),
            "absa": CapabilityStatusItem(status="pending"),
            "emotion": CapabilityStatusItem(status="pending"),
            "topics": CapabilityStatusItem(status="pending"),
        }

        # 1. Overall Sentiment (TensorFlow Engine)
        overall_sentiment: Optional[OverallSentimentResult] = None
        t_sent = time.time()
        try:
            sent_res = tf_engine.predict_single(text)
            sent_lat = (time.time() - t_sent) * 1000
            overall_sentiment = OverallSentimentResult(
                sentiment=sent_res["sentiment"],
                confidence=round(sent_res["confidence"], 4),
                probabilities={
                    k: round(v, 4) for k, v in sent_res["probabilities"].items()
                },
                latency_ms=round(sent_lat, 2),
                explainability=sent_res.get("explainability"),
            )
            capability_status["sentiment"] = CapabilityStatusItem(
                status="completed",
                model_name="TensorFlow-BiLSTM-SelfAttention",
                model_version="v1.0.0",
                latency_ms=round(sent_lat, 2),
            )
        except Exception as e:
            logger.error(f"Overall sentiment engine error: {e}", exc_info=True)
            capability_status["sentiment"] = CapabilityStatusItem(
                status="failed",
                error=f"Sentiment engine failure: {str(e)}",
            )

        # 2. Aspect-Based Sentiment Analysis (DeBERTa-v3 ABSA Service)
        aspects: List[UnifiedAspectItem] = []
        t_absa = time.time()
        if absa_service.is_ready():
            try:
                absa_resp = await absa_service.analyze(
                    text=text, save_to_db=False
                )
                absa_lat = (time.time() - t_absa) * 1000
                for a in absa_resp.aspects:
                    aspects.append(
                        UnifiedAspectItem(
                            aspect_term=a.aspect,
                            aspect_category=a.aspect_category,
                            sentiment=a.sentiment,
                            confidence=round(a.confidence, 4),
                            start_char=a.start_char,
                            end_char=a.end_char,
                            evidence_text=a.evidence_text,
                        )
                    )
                capability_status["absa"] = CapabilityStatusItem(
                    status="completed",
                    model_name="DeBERTa-v3-ABSA",
                    model_version="v1.0.0",
                    latency_ms=round(absa_lat, 2),
                )
            except Exception as e:
                logger.error(f"ABSA extraction error: {e}", exc_info=True)
                capability_status["absa"] = CapabilityStatusItem(
                    status="failed",
                    error=f"ABSA extraction failed: {str(e)}",
                )
        else:
            capability_status["absa"] = CapabilityStatusItem(
                status="unavailable",
                error="ABSA DeBERTa models are not loaded or local weights missing",
            )

        # 3. Emotion Detection (RoBERTa-GoEmotions Service)
        emotions: Optional[UnifiedEmotionResult] = None
        t_emo = time.time()
        if emotion_service.is_ready():
            try:
                emo_resp = await emotion_service.analyze(
                    text=text, save_to_db=False
                )
                emo_lat = (time.time() - t_emo) * 1000
                emotions = UnifiedEmotionResult(
                    primary_cx_category=emo_resp.primary_emotion,
                    confidence_score=round(emo_resp.confidence, 4),
                    detected_emotions=[
                        {"emotion": d.label, "score": round(d.probability, 4)}
                        for d in emo_resp.detected_emotions
                    ],
                    all_probabilities={
                        k: round(v, 4) for k, v in emo_resp.raw_scores.items()
                    },
                )
                capability_status["emotion"] = CapabilityStatusItem(
                    status="completed",
                    model_name="RoBERTa-GoEmotions-MultiLabel",
                    model_version="v1.0.0",
                    latency_ms=round(emo_lat, 2),
                )
            except Exception as e:
                logger.error(f"Emotion detection error: {e}", exc_info=True)
                capability_status["emotion"] = CapabilityStatusItem(
                    status="failed",
                    error=f"Emotion detection failed: {str(e)}",
                )
        else:
            capability_status["emotion"] = CapabilityStatusItem(
                status="unavailable",
                error="Emotion RoBERTa model is not loaded or local weights missing",
            )

        # 4. Lightweight Topic Projection
        topic_proj: Optional[UnifiedTopicProjection] = None
        t_top = time.time()
        if not project_topic:
            capability_status["topics"] = CapabilityStatusItem(
                status="unavailable",
                error="Topic projection disabled in request",
            )
        elif not topic_service.is_ready():
            capability_status["topics"] = CapabilityStatusItem(
                status="unavailable",
                error="Topic embedding model is not loaded or local weights missing",
            )
        else:
            try:
                topic_proj, meta = await self._project_onto_latest_run(session, text)
                top_lat = (time.time() - t_top) * 1000
                if topic_proj is not None:
                    capability_status["topics"] = CapabilityStatusItem(
                        status="completed",
                        model_name="all-MiniLM-L6-v2 + CentroidProjection",
                        model_version=meta.get("run_id", "v1.0.0"),
                        latency_ms=round(top_lat, 2),
                    )
                else:
                    capability_status["topics"] = CapabilityStatusItem(
                        status="unavailable",
                        error=meta.get(
                            "reason", "No completed topic discovery run found"
                        ),
                    )
            except Exception as e:
                logger.error(f"Topic projection error: {e}", exc_info=True)
                capability_status["topics"] = CapabilityStatusItem(
                    status="failed",
                    error=f"Topic projection failed: {str(e)}",
                )

        # 5. Database Persistence (Linking Prediction, ReviewAspects, ReviewEmotions, TopicAssignment)
        if save_to_db and overall_sentiment is not None:
            try:
                pred_record = Prediction(
                    id=review_id,
                    user_id=user_id or "anonymous-user",
                    text_content=text,
                    sentiment_label=overall_sentiment.sentiment,
                    confidence_score=overall_sentiment.confidence,
                    positive_prob=overall_sentiment.probabilities.get("positive", 0.0),
                    neutral_prob=overall_sentiment.probabilities.get("neutral", 0.0),
                    negative_prob=overall_sentiment.probabilities.get("negative", 0.0),
                    latency_ms=overall_sentiment.latency_ms or 0.0,
                    explainability_json=overall_sentiment.explainability,
                )
                session.add(pred_record)
                await session.flush()

                # Aspect records
                aspect_records = []
                for a in aspects:
                    pos = 1.0 if a.sentiment == "POSITIVE" else 0.0
                    neu = 1.0 if a.sentiment == "NEUTRAL" else 0.0
                    neg = 1.0 if a.sentiment == "NEGATIVE" else 0.0
                    aspect_records.append(
                        ReviewAspect(
                            prediction_id=review_id,
                            aspect_term=a.aspect_term,
                            aspect_category=a.aspect_category,
                            start_char=a.start_char,
                            end_char=a.end_char,
                            sentiment_label=a.sentiment,
                            confidence_score=a.confidence,
                            positive_prob=pos,
                            neutral_prob=neu,
                            negative_prob=neg,
                            evidence_text=a.evidence_text,
                            model_name="DeBERTa-v3-ABSA",
                            model_version="v1.0.0",
                        )
                    )
                if aspect_records:
                    session.add_all(aspect_records)

                # Emotion record
                if emotions is not None:
                    emo_rec = ReviewEmotion(
                        prediction_id=review_id,
                        primary_cx_category=emotions.primary_cx_category,
                        confidence_score=emotions.confidence_score,
                        detected_emotions_json=emotions.detected_emotions,
                        raw_emotions_json=emotions.all_probabilities,
                        model_name="RoBERTa-GoEmotions-MultiLabel",
                        model_version="v1.0.0",
                    )
                    session.add(emo_rec)

                # Topic Assignment record
                if topic_proj is not None and topic_proj.topic_id is not None:
                    sim = topic_proj.similarity_score or 0.0
                    ta_rec = TopicAssignment(
                        run_id=topic_proj.topic_run_version,
                        topic_id=topic_proj.topic_id,
                        prediction_id=review_id,
                        review_text=text,
                        distance_to_centroid=round(float(1.0 - sim), 4),
                        is_outlier=topic_proj.is_outlier,
                        sentiment_label=overall_sentiment.sentiment,
                        confidence_score=overall_sentiment.confidence,
                    )
                    session.add(ta_rec)

                await session.commit()
            except Exception as e:
                logger.error(f"Error persisting unified prediction: {e}", exc_info=True)
                await session.rollback()

        total_lat = (time.time() - t0) * 1000

        return UnifiedReviewResponse(
            review_id=review_id,
            text_content=text,
            overall_sentiment=overall_sentiment,
            aspects=aspects,
            emotions=emotions,
            topic=topic_proj,
            capability_status=capability_status,
            total_latency_ms=round(total_lat, 2),
            created_at=datetime.utcnow().isoformat(),
        )

    async def process_batch(
        self,
        session: AsyncSession,
        reviews: List[str],
        user_id: Optional[str] = None,
        save_to_db: bool = True,
        job_id: Optional[str] = None,
    ) -> List[UnifiedReviewResponse]:
        """Unified multi-task inference over batches with memory-safe micro-chunking and deduplication."""
        # Deduplicate identical review texts within batch to avoid redundant neural passes
        unique_texts: List[str] = []
        text_to_orig_indices: Dict[str, List[int]] = {}
        for idx, text in enumerate(reviews):
            cleaned = text.strip()
            if cleaned not in text_to_orig_indices:
                text_to_orig_indices[cleaned] = []
                unique_texts.append(cleaned)
            text_to_orig_indices[cleaned].append(idx)

        total_unique = len(unique_texts)
        total_orig = len(reviews)
        results_by_cleaned_text: Dict[str, UnifiedReviewResponse] = {}

        chunk_size = 16  # Memory safe chunking for 4 GB VRAM
        processed = 0

        for i in range(0, total_unique, chunk_size):
            chunk = unique_texts[i : i + chunk_size]
            for text in chunk:
                res = await self.analyze_review(
                    session=session,
                    text=text,
                    user_id=user_id,
                    save_to_db=save_to_db,
                    project_topic=True,
                )
                results_by_cleaned_text[text] = res
                processed += 1

            # Update job progress if job_id provided
            if job_id:
                pct = round((processed / total_unique) * 100.0, 1)
                self.update_job_progress(
                    job_id=job_id,
                    processed_records=processed,
                    progress_pct=pct,
                    status="PROCESSING",
                )

        # Reconstruct ordered results matching original review list
        ordered_results: List[UnifiedReviewResponse] = []
        for text in reviews:
            cleaned = text.strip()
            ordered_results.append(results_by_cleaned_text[cleaned])

        return ordered_results

    def update_job_progress(
        self,
        job_id: str,
        processed_records: int,
        progress_pct: float,
        status: str = "PROCESSING",
        succeeded: int = 0,
        failed: int = 0,
        partial: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Update in-memory job status tracker."""
        if job_id in self._in_memory_jobs:
            self._in_memory_jobs[job_id]["processed_records"] = processed_records
            self._in_memory_jobs[job_id]["progress_pct"] = progress_pct
            self._in_memory_jobs[job_id]["status"] = status
            if succeeded:
                self._in_memory_jobs[job_id]["succeeded_records"] = succeeded
            if failed:
                self._in_memory_jobs[job_id]["failed_records"] = failed
            if partial:
                self._in_memory_jobs[job_id]["partial_records"] = partial
            if error_message:
                self._in_memory_jobs[job_id]["error_message"] = error_message
            if status in ("COMPLETED", "FAILED"):
                self._in_memory_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    def get_batch_job(self, job_id: str) -> Optional[IntelligenceBatchJobStatus]:
        """Fetch status of an asynchronous batch job."""
        data = self._in_memory_jobs.get(job_id)
        if not data:
            return None
        return IntelligenceBatchJobStatus(**data)

    # ==============================================================================
    # Cross-Feature SQL Analytics
    # ==============================================================================

    async def get_cross_feature_analytics(
        self,
        session: AsyncSession,
        min_sample_size: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> IntelligenceAnalyticsResponse:
        """Execute real SQL-backed cross-feature aggregation queries across models."""
        # 1. Total analyzed reviews
        stmt_tot = select(func.count(Prediction.id))
        total_reviews = (await session.execute(stmt_tot)).scalar() or 0

        # 2. Aspect Sentiment by Topic
        # Join review_aspects -> predictions -> topic_assignments -> discovered_topics
        stmt_aspect_topic = (
            select(
                DiscoveredTopic.id.label("topic_id"),
                func.coalesce(DiscoveredTopic.human_label, DiscoveredTopic.auto_label).label("topic_name"),
                DiscoveredTopic.topic_index,
                ReviewAspect.aspect_term,
                ReviewAspect.aspect_category,
                func.sum(
                    case((ReviewAspect.sentiment_label == "POSITIVE", 1), else_=0)
                ).label("pos_cnt"),
                func.sum(
                    case((ReviewAspect.sentiment_label == "NEUTRAL", 1), else_=0)
                ).label("neu_cnt"),
                func.sum(
                    case((ReviewAspect.sentiment_label == "NEGATIVE", 1), else_=0)
                ).label("neg_cnt"),
                func.count(ReviewAspect.id).label("total_cnt"),
            )
            .join(Prediction, ReviewAspect.prediction_id == Prediction.id)
            .join(TopicAssignment, TopicAssignment.prediction_id == Prediction.id)
            .join(DiscoveredTopic, TopicAssignment.topic_id == DiscoveredTopic.id)
            .group_by(
                DiscoveredTopic.id,
                DiscoveredTopic.human_label,
                DiscoveredTopic.auto_label,
                DiscoveredTopic.topic_index,
                ReviewAspect.aspect_term,
                ReviewAspect.aspect_category,
            )
            .having(func.count(ReviewAspect.id) >= min_sample_size)
            .order_by(desc("total_cnt"))
            .limit(100)
        )
        res_aspect_topic = await session.execute(stmt_aspect_topic)
        aspect_topic_rows = res_aspect_topic.all()

        aspect_topic_items: List[AspectSentimentByTopicItem] = []
        for r in aspect_topic_rows:
            tot = r.total_cnt or 0
            pos = r.pos_cnt or 0
            neg = r.neg_cnt or 0
            neu = r.neu_cnt or 0
            aspect_topic_items.append(
                AspectSentimentByTopicItem(
                    topic_id=r.topic_id,
                    topic_name=r.topic_name or f"Topic {r.topic_index + 1}",
                    topic_index=r.topic_index,
                    aspect_term=r.aspect_term,
                    aspect_category=r.aspect_category,
                    positive_count=pos,
                    neutral_count=neu,
                    negative_count=neg,
                    total_count=tot,
                    positive_pct=round(pos / max(1, tot), 4),
                    negative_pct=round(neg / max(1, tot), 4),
                    sample_count=tot,
                    low_sample_warning=(tot < 5),
                )
            )

        # 3. Emotion by Overall Sentiment
        stmt_emo_sent = (
            select(
                Prediction.sentiment_label,
                ReviewEmotion.primary_cx_category,
                func.count(Prediction.id).label("cnt"),
            )
            .join(ReviewEmotion, ReviewEmotion.prediction_id == Prediction.id)
            .group_by(Prediction.sentiment_label, ReviewEmotion.primary_cx_category)
            .order_by(Prediction.sentiment_label, desc("cnt"))
        )
        res_emo_sent = await session.execute(stmt_emo_sent)
        emo_sent_rows = res_emo_sent.all()

        # Compute sentiment totals for relative percentage
        sent_totals: Dict[str, int] = {}
        for r in emo_sent_rows:
            sent_totals[r.sentiment_label] = sent_totals.get(r.sentiment_label, 0) + r.cnt

        emo_sent_items: List[EmotionBySentimentItem] = []
        for r in emo_sent_rows:
            tot_s = sent_totals.get(r.sentiment_label, 1)
            emo_sent_items.append(
                EmotionBySentimentItem(
                    sentiment_label=r.sentiment_label,
                    emotion_category=r.primary_cx_category,
                    count=r.cnt,
                    percentage=round(r.cnt / max(1, tot_s), 4),
                    sample_count=tot_s,
                    low_sample_warning=(tot_s < 5),
                )
            )

        # 4. Emotion by Aspect Category
        stmt_emo_aspect = (
            select(
                func.coalesce(ReviewAspect.aspect_category, ReviewAspect.aspect_term).label("category"),
                ReviewEmotion.primary_cx_category,
                func.count(ReviewAspect.id).label("cnt"),
            )
            .join(Prediction, ReviewAspect.prediction_id == Prediction.id)
            .join(ReviewEmotion, ReviewEmotion.prediction_id == Prediction.id)
            .group_by("category", ReviewEmotion.primary_cx_category)
            .having(func.count(ReviewAspect.id) >= min_sample_size)
            .order_by(desc("cnt"))
            .limit(100)
        )
        res_emo_aspect = await session.execute(stmt_emo_aspect)
        emo_aspect_rows = res_emo_aspect.all()

        aspect_totals: Dict[str, int] = {}
        for r in emo_aspect_rows:
            aspect_totals[r.category] = aspect_totals.get(r.category, 0) + r.cnt

        emo_aspect_items: List[EmotionByAspectCategoryItem] = []
        for r in emo_aspect_rows:
            tot_a = aspect_totals.get(r.category, 1)
            emo_aspect_items.append(
                EmotionByAspectCategoryItem(
                    aspect_category=r.category,
                    emotion_category=r.primary_cx_category,
                    count=r.cnt,
                    percentage=round(r.cnt / max(1, tot_a), 4),
                    sample_count=tot_a,
                    low_sample_warning=(tot_a < 5),
                )
            )

        # 5. Topic Summary (Sentiment distribution + Dominant Emotion)
        stmt_topic_sum = (
            select(
                DiscoveredTopic.id.label("topic_id"),
                func.coalesce(DiscoveredTopic.human_label, DiscoveredTopic.auto_label).label("topic_name"),
                DiscoveredTopic.topic_index,
                func.count(func.distinct(Prediction.id)).label("tot_reviews"),
                func.sum(case((Prediction.sentiment_label == "POSITIVE", 1), else_=0)).label("pos_c"),
                func.sum(case((Prediction.sentiment_label == "NEUTRAL", 1), else_=0)).label("neu_c"),
                func.sum(case((Prediction.sentiment_label == "NEGATIVE", 1), else_=0)).label("neg_c"),
            )
            .join(TopicAssignment, TopicAssignment.topic_id == DiscoveredTopic.id)
            .join(Prediction, TopicAssignment.prediction_id == Prediction.id)
            .group_by(
                DiscoveredTopic.id,
                DiscoveredTopic.human_label,
                DiscoveredTopic.auto_label,
                DiscoveredTopic.topic_index,
            )
            .having(func.count(func.distinct(Prediction.id)) >= min_sample_size)
            .order_by(desc("tot_reviews"))
        )
        res_topic_sum = await session.execute(stmt_topic_sum)
        topic_sum_rows = res_topic_sum.all()

        # Dominant emotion per topic
        stmt_top_emo = (
            select(
                TopicAssignment.topic_id,
                ReviewEmotion.primary_cx_category,
                func.count(ReviewEmotion.id).label("cnt"),
            )
            .join(ReviewEmotion, TopicAssignment.prediction_id == ReviewEmotion.prediction_id)
            .group_by(TopicAssignment.topic_id, ReviewEmotion.primary_cx_category)
            .order_by(TopicAssignment.topic_id, desc("cnt"))
        )
        res_top_emo = await session.execute(stmt_top_emo)
        top_emo_rows = res_top_emo.all()
        dom_emo_by_topic: Dict[str, str] = {}
        for r in top_emo_rows:
            if r.topic_id not in dom_emo_by_topic:
                dom_emo_by_topic[r.topic_id] = r.primary_cx_category

        topic_summary_items: List[TopicSummaryItem] = []
        for r in topic_sum_rows:
            tot = r.tot_reviews or 0
            pos = r.pos_c or 0
            neu = r.neu_c or 0
            neg = r.neg_c or 0
            topic_summary_items.append(
                TopicSummaryItem(
                    topic_id=r.topic_id,
                    topic_name=r.topic_name,
                    topic_index=r.topic_index,
                    total_reviews=tot,
                    sentiment_breakdown={
                        "POSITIVE": round(pos / max(1, tot), 4),
                        "NEUTRAL": round(neu / max(1, tot), 4),
                        "NEGATIVE": round(neg / max(1, tot), 4),
                    },
                    dominant_emotion=dom_emo_by_topic.get(r.topic_id, "NEUTRAL"),
                    sample_count=tot,
                    low_sample_warning=(tot < 5),
                )
            )

        # 6. Negative Aspect Trends (daily)
        # Using SQLite date string slicing or func.date
        date_expr = func.substr(ReviewAspect.created_at, 1, 10)
        stmt_trends = (
            select(
                date_expr.label("d_str"),
                ReviewAspect.aspect_term,
                func.sum(
                    case((ReviewAspect.sentiment_label == "NEGATIVE", 1), else_=0)
                ).label("neg_cnt"),
                func.count(ReviewAspect.id).label("tot_cnt"),
            )
            .group_by("d_str", ReviewAspect.aspect_term)
            .order_by(desc("d_str"), desc("tot_cnt"))
            .limit(100)
        )
        res_trends = await session.execute(stmt_trends)
        trend_rows = res_trends.all()

        trend_items: List[NegativeAspectTrendItem] = []
        for r in trend_rows:
            tot = r.tot_cnt or 0
            neg = r.neg_cnt or 0
            trend_items.append(
                NegativeAspectTrendItem(
                    date=str(r.d_str or datetime.utcnow().strftime("%Y-%m-%d")),
                    aspect_term=r.aspect_term,
                    negative_count=neg,
                    total_aspect_count=tot,
                    negative_rate=round(neg / max(1, tot), 4),
                    sample_count=tot,
                    low_sample_warning=(tot < 5),
                )
            )

        return IntelligenceAnalyticsResponse(
            aspect_sentiment_by_topic=aspect_topic_items,
            emotion_by_overall_sentiment=emo_sent_items,
            emotion_by_aspect_category=emo_aspect_items,
            topic_sentiment_and_emotion_summary=topic_summary_items,
            negative_aspect_trends=trend_items,
            methodology_note=(
                "Statistical co-occurrence groupings reflect multi-model association across review corpora "
                "rather than direct causal attribution."
            ),
            min_sample_size_applied=min_sample_size,
            total_analyzed_reviews=total_reviews,
        )

    # ==============================================================================
    # Evidence Drilldown
    # ==============================================================================

    async def get_evidence_drilldown(
        self,
        session: AsyncSession,
        topic_id: Optional[str] = None,
        aspect_term: Optional[str] = None,
        emotion_category: Optional[str] = None,
        sentiment: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> EvidenceDrilldownResponse:
        """Fetch actual review texts and linked model outputs matching cross-feature filters."""
        stmt = (
            select(
                Prediction.id.label("prediction_id"),
                Prediction.text_content,
                Prediction.sentiment_label.label("overall_sentiment"),
                Prediction.confidence_score,
                Prediction.created_at,
                ReviewAspect.aspect_term,
                ReviewAspect.sentiment_label.label("aspect_sentiment"),
                ReviewEmotion.primary_cx_category.label("emotion_category"),
                func.coalesce(DiscoveredTopic.human_label, DiscoveredTopic.auto_label).label("topic_name"),
            )
            .outerjoin(ReviewAspect, ReviewAspect.prediction_id == Prediction.id)
            .outerjoin(ReviewEmotion, ReviewEmotion.prediction_id == Prediction.id)
            .outerjoin(TopicAssignment, TopicAssignment.prediction_id == Prediction.id)
            .outerjoin(DiscoveredTopic, TopicAssignment.topic_id == DiscoveredTopic.id)
        )

        if topic_id:
            stmt = stmt.where(TopicAssignment.topic_id == topic_id)
        if aspect_term:
            stmt = stmt.where(ReviewAspect.aspect_term.ilike(f"%{aspect_term}%"))
        if emotion_category:
            stmt = stmt.where(ReviewEmotion.primary_cx_category == emotion_category)
        if sentiment:
            stmt = stmt.where(Prediction.sentiment_label == sentiment.upper())

        # Count total
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_matched = (await session.execute(count_stmt)).scalar() or 0

        # Paginate
        offset = (max(1, page) - 1) * page_size
        stmt = stmt.order_by(desc(Prediction.created_at)).offset(offset).limit(page_size)
        rows = (await session.execute(stmt)).all()

        items: List[EvidenceItem] = []
        for r in rows:
            created_str = (
                r.created_at.isoformat()
                if hasattr(r.created_at, "isoformat")
                else str(r.created_at)
            )
            items.append(
                EvidenceItem(
                    prediction_id=r.prediction_id,
                    review_text=r.text_content,
                    overall_sentiment=r.overall_sentiment,
                    confidence_score=r.confidence_score,
                    aspect_term=r.aspect_term,
                    aspect_sentiment=r.aspect_sentiment,
                    emotion_category=r.emotion_category,
                    topic_name=r.topic_name,
                    created_at=created_str,
                )
            )

        filters = {
            "topic_id": topic_id,
            "aspect_term": aspect_term,
            "emotion_category": emotion_category,
            "sentiment": sentiment,
        }

        return EvidenceDrilldownResponse(
            filters=filters,
            total_matched=total_matched,
            page=page,
            page_size=page_size,
            items=items,
        )


intelligence_service = IntelligenceService.get_instance()
