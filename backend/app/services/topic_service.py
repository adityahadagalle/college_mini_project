"""Collection-Level Automatic Topic Discovery Service.

Provides:
1. Local Sentence-Transformers neural embeddings (`sentence-transformers/all-MiniLM-L6-v2`) on CPU (0 GPU VRAM).
2. Deterministic review embedding cache persisted in `review_embeddings` DB table.
3. Strict corpus size validation (>= 15 unique reviews required, fail-fast with HTTP 400).
4. Strict model readiness check (fail-fast with HTTP 503 if local weights are missing/unloaded).
5. Dimensionality reduction (UMAP with deterministic random_state, fallback to TruncatedSVD) + HDBSCAN clustering.
6. Outlier preservation (`topic_index = -1` marked as 'Outliers / Unassigned').
7. c-TF-IDF keyword representation and automated topic naming.
8. Centroid computation, intra-cluster cohesion, keyword diversity, and top-3 representative review extraction.
9. Asynchronous background job runner with progress tracking and duplicate job prevention.
10. Human-in-the-loop topic management: rename, merge, archive with `topic_audit_logs`.
11. Topic drift and version comparison with mandatory candidate uncertainty disclaimer.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import (
    DiscoveredTopic,
    Prediction,
    ReviewEmbedding,
    TopicAssignment,
    TopicAuditLog,
    TopicDiscoveryRun,
)
from app.schemas.topic import (
    CandidateEmergingTopic,
    DiscoveredTopicItem,
    MatchedTopicPair,
    TopicComparisonResponse,
    TopicDiscoveryJobStatus,
    TopicKeywordItem,
    TopicListResponse,
    TopicReviewAssignmentItem,
    TopicReviewsResponse,
)

logger = logging.getLogger(__name__)

MODEL_LOCAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../models/topics/embedding_model")
)
TOPIC_METADATA_PATH = os.path.abspath(
    os.path.join(MODEL_LOCAL_DIR, "..", "topic_metadata.json")
)
MIN_REVIEWS_THRESHOLD = 15
STOP_WORDS_CUSTOM = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "it", "its", "they", "them", "their", "we", "us", "our", "you",
    "your", "i", "me", "my", "this", "that", "these", "those", "so",
    "too", "very", "can", "will", "just", "would", "could", "should", "as",
    "if", "when", "than", "then", "there", "what", "which", "who", "all",
}


def normalize_text(text: str) -> str:
    """Normalize text content for consistent hashing and embedding."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def compute_sha256(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    return hashlib.sha256(text.lower().encode("utf-8")).hexdigest()


class TopicDiscoveryService:
    """Singleton service for collection-level topic modeling."""

    _instance: Optional["TopicDiscoveryService"] = None

    def __init__(self, model_dir: str = MODEL_LOCAL_DIR):
        self.model_dir = model_dir
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.device = torch.device("cpu")
        self.embedding_dimension = 384
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self._jobs: Dict[str, Dict[str, Any]] = {}

        self._load_local_weights()

    @classmethod
    def get_instance(cls) -> "TopicDiscoveryService":
        if cls._instance is None:
            cls._instance = TopicDiscoveryService()
        return cls._instance

    def is_ready(self) -> bool:
        """Verify local model weights and tokenizer are loaded and usable."""
        return self.tokenizer is not None and self.model is not None

    def _load_local_weights(self) -> None:
        """Load Sentence Transformers embedding model strictly from local disk."""
        if not os.path.isdir(self.model_dir):
            logger.warning(
                f"Topic embedding model directory {self.model_dir} not found. Service offline."
            )
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(
                f"Loading local topic embedding model from {self.model_dir} on CPU (0 GPU VRAM)"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_dir, local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("Topic embedding model loaded successfully into memory.")
        except Exception as e:
            logger.error(
                f"Failed to load local topic embedding model from {self.model_dir}: {e}"
            )
            self.tokenizer = None
            self.model = None

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate mean-pooled L2-normalized 384-d embeddings strictly on CPU."""
        if not self.is_ready():
            raise RuntimeError(
                "Topic Discovery embedding model is not initialized or local weights are missing."
            )

        all_embeddings: List[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, 384)
                input_mask_expanded = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu().numpy())

        if not all_embeddings:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)

    async def get_or_create_embeddings(
        self, session: AsyncSession, texts: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Retrieve deterministic embeddings from cache or compute and persist misses."""
        cleaned_texts = [normalize_text(t) for t in texts]
        hashes = [compute_sha256(t) for t in cleaned_texts]

        # Query existing hashes from DB
        stmt = select(ReviewEmbedding).where(ReviewEmbedding.text_hash.in_(hashes))
        result = await session.execute(stmt)
        existing_rows = {r.text_hash: r for r in result.scalars().all()}

        hits = 0
        misses = 0
        computed_dict: Dict[str, np.ndarray] = {}

        # Identify misses
        missing_texts: List[str] = []
        missing_hashes: List[str] = []
        for text, h in zip(cleaned_texts, hashes):
            if h in existing_rows:
                hits += 1
            elif h not in computed_dict:
                missing_texts.append(text)
                missing_hashes.append(h)
                misses += 1

        # Compute embeddings for missing reviews
        if missing_texts:
            encoded_matrix = self.encode_texts(missing_texts, batch_size=32)
            new_records: List[ReviewEmbedding] = []
            for text, h, vec in zip(missing_texts, missing_hashes, encoded_matrix):
                computed_dict[h] = vec
                new_records.append(
                    ReviewEmbedding(
                        text_hash=h,
                        cleaned_text=text,
                        embedding_json=vec.tolist(),
                        model_name=self.model_name,
                        dimension=self.embedding_dimension,
                    )
                )

            session.add_all(new_records)
            try:
                await session.flush()
            except Exception as e:
                logger.warning(f"Error persisting new review embeddings: {e}")
                await session.rollback()

        # Construct final ordered embedding matrix
        final_embeddings: List[np.ndarray] = []
        for h in hashes:
            if h in existing_rows:
                emb = np.array(existing_rows[h].embedding_json, dtype=np.float32)
                final_embeddings.append(emb)
            else:
                final_embeddings.append(computed_dict[h])

        hit_rate = (hits / max(1, hits + misses)) * 100.0
        cache_stats = {
            "hits": hits,
            "misses": misses,
            "total": len(texts),
            "hit_rate_pct": round(hit_rate, 2),
        }
        logger.info(
            f"Review embedding cache: {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)"
        )
        return np.vstack(final_embeddings), cache_stats

    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        n_components: int = 5,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> np.ndarray:
        """Dimension reduction with UMAP (when enabled) or TruncatedSVD (CPU-optimized LSA)."""
        n_samples = embeddings.shape[0]
        eff_components = min(n_components, max(2, n_samples - 2))
        eff_neighbors = min(n_neighbors, max(2, n_samples - 1))

        # Check if UMAP is explicitly requested via environment or parameter
        use_umap = os.getenv("ENABLE_UMAP", "0") == "1"
        if use_umap:
            try:
                from umap import UMAP

                reducer = UMAP(
                    n_neighbors=eff_neighbors,
                    n_components=eff_components,
                    metric="cosine",
                    random_state=random_state,
                    min_dist=0.0,
                )
                reduced = reducer.fit_transform(embeddings)
                logger.info(
                    f"UMAP reduction completed: {embeddings.shape} -> {reduced.shape}"
                )
                return reduced
            except Exception as e:
                logger.warning(
                    f"UMAP dimension reduction unavailable ({e}). Falling back to TruncatedSVD."
                )

        # Standard fast, deterministic CPU dimensionality reduction (TruncatedSVD)
        svd = TruncatedSVD(n_components=eff_components, random_state=random_state)
        reduced = svd.fit_transform(embeddings)
        return reduced

    def _cluster_embeddings(
        self,
        reduced_embeddings: np.ndarray,
        min_cluster_size: int = 3,
        min_samples: Optional[int] = None,
    ) -> np.ndarray:
        """HDBSCAN density-based clustering with outlier support (-1)."""
        n_samples = reduced_embeddings.shape[0]
        eff_min_size = min(min_cluster_size, max(2, n_samples // 4))

        clusterer = HDBSCAN(
            min_cluster_size=eff_min_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(reduced_embeddings)
        return labels

    def _extract_c_tf_idf(
        self,
        corpus: List[str],
        labels: np.ndarray,
        top_k: int = 8,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Class-based TF-IDF keyword representation for discovered clusters."""
        unique_labels = sorted(list(set(labels)))
        cluster_docs: Dict[int, str] = {}

        for lbl in unique_labels:
            indices = [i for i, x in enumerate(labels) if x == lbl]
            cluster_docs[lbl] = " ".join([corpus[i] for i in indices])

        doc_list = [cluster_docs[lbl] for lbl in unique_labels]

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=4000,
            min_df=1,
        )
        try:
            count_matrix = vectorizer.fit_transform(doc_list).toarray()
            vocab = vectorizer.get_feature_names_out()
        except ValueError:
            # Empty vocabulary fallback
            return {lbl: [{"word": f"cluster_{lbl}", "score": 1.0}] for lbl in unique_labels}

        # Calculate c-TF-IDF
        # Normalized Term Frequency: tf_c = count_matrix / total words in cluster
        total_words = count_matrix.sum(axis=1, keepdims=True)
        total_words = np.where(total_words == 0, 1e-9, total_words)
        tf = count_matrix / total_words

        # Class-based IDF: ln(1 + avg_words_per_class / freq_across_classes)
        avg_words = count_matrix.sum() / max(1, len(unique_labels))
        word_freq = count_matrix.sum(axis=0)
        word_freq = np.where(word_freq == 0, 1e-9, word_freq)
        idf = np.log(1.0 + (avg_words / word_freq))

        c_tf_idf = tf * idf

        keywords_by_cluster: Dict[int, List[Dict[str, Any]]] = {}
        for row_idx, lbl in enumerate(unique_labels):
            scores = c_tf_idf[row_idx]
            top_indices = np.argsort(scores)[::-1][:top_k]
            kw_items = []
            for idx in top_indices:
                score = float(scores[idx])
                if score > 0.0:
                    kw_items.append({"word": vocab[idx], "score": round(score, 4)})
            if not kw_items:
                kw_items = [{"word": f"topic_{lbl}", "score": 1.0}]
            keywords_by_cluster[lbl] = kw_items

        return keywords_by_cluster

    def _generate_auto_label(
        self, topic_index: int, keywords: List[Dict[str, Any]]
    ) -> str:
        """Derive meaningful human-readable title from top c-TF-IDF keywords."""
        if topic_index == -1:
            return "Outliers / Unassigned"

        top_words = [
            item["word"].title() for item in keywords[:3] if item.get("word")
        ]
        if not top_words:
            return f"Topic {topic_index + 1}"
        return " / ".join(top_words)

    def is_job_running(self) -> bool:
        """Check whether any topic discovery job is actively in progress or queued."""
        for job in self._jobs.values():
            if job.get("status") in ("PROCESSING", "PENDING"):
                return True
        return False

    def get_job_status(self, job_id: str) -> Optional[TopicDiscoveryJobStatus]:
        """Fetch current status for a given discovery run job ID."""
        data = self._jobs.get(job_id)
        if not data:
            return None
        return TopicDiscoveryJobStatus(**data)

    async def execute_discovery_run(
        self,
        session: AsyncSession,
        run_id: str,
        reviews_data: List[Dict[str, Any]],
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> TopicListResponse:
        """Orchestrate end-to-end collection topic discovery and persist results."""
        if not self.is_ready():
            raise RuntimeError(
                "Topic Discovery service is offline: embedding model weights could not be loaded."
            )

        # Enforce minimum corpus size constraint
        unique_texts_map: Dict[str, Dict[str, Any]] = {}
        for r in reviews_data:
            c = normalize_text(r.get("text", ""))
            if len(c) > 3 and c not in unique_texts_map:
                unique_texts_map[c] = r

        unique_reviews = list(unique_texts_map.values())
        corpus_size = len(unique_reviews)

        if corpus_size < MIN_REVIEWS_THRESHOLD:
            raise ValueError(
                f"Topic discovery requires at least {MIN_REVIEWS_THRESHOLD} unique reviews for "
                f"statistically valid clustering. Provided corpus size: {corpus_size}"
            )

        # Track job progress
        now_iso = datetime.utcnow().isoformat()
        self._jobs[run_id] = {
            "job_id": run_id,
            "status": "PROCESSING",
            "progress": 0.05,
            "stage": "Ingesting and validating review corpus",
            "corpus_size": corpus_size,
            "num_topics": 0,
            "outlier_count": 0,
            "metrics": None,
            "error_message": None,
            "created_at": now_iso,
            "completed_at": None,
        }

        try:
            # 1. Embeddings
            self._jobs[run_id]["progress"] = 0.20
            self._jobs[run_id]["stage"] = "Generating neural embeddings via all-MiniLM-L6-v2 (CPU)"
            corpus_texts = [r["text"] for r in unique_reviews]
            embeddings, cache_stats = await self.get_or_create_embeddings(
                session, corpus_texts
            )

            # 2. Dimensionality reduction
            self._jobs[run_id]["progress"] = 0.45
            self._jobs[run_id]["stage"] = "Reducing dimensions with UMAP (random_state=42)"
            reduced = self._reduce_dimensions(
                embeddings,
                n_components=parameters.get("n_components", 5),
                n_neighbors=parameters.get("n_neighbors", 15),
                random_state=parameters.get("random_state", 42),
            )

            # 3. Clustering
            self._jobs[run_id]["progress"] = 0.65
            self._jobs[run_id]["stage"] = "Clustering density distributions with HDBSCAN"
            cluster_labels = self._cluster_embeddings(
                reduced,
                min_cluster_size=parameters.get("min_cluster_size", 3),
                min_samples=parameters.get("min_samples"),
            )

            # 4. Keyword Extraction & Outlier isolation
            self._jobs[run_id]["progress"] = 0.80
            self._jobs[run_id]["stage"] = "Extracting c-TF-IDF keywords and cluster centroids"
            keywords_by_cluster = self._extract_c_tf_idf(corpus_texts, cluster_labels)

            unique_clusters = sorted(list(set(cluster_labels)))
            num_meaningful_topics = len([c for c in unique_clusters if c != -1])
            outlier_count = int(np.sum(cluster_labels == -1))

            # Quality metrics: intra-cluster cohesion & keyword diversity
            cohesion_scores: Dict[int, float] = {}
            centroids_by_cluster: Dict[int, np.ndarray] = {}
            all_kw_words: List[str] = []

            for lbl in unique_clusters:
                indices = np.where(cluster_labels == lbl)[0]
                cluster_embs = embeddings[indices]
                # Normalized centroid
                centroid = cluster_embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                centroids_by_cluster[lbl] = centroid

                # Cohesion: mean cosine similarity to centroid
                if len(indices) > 0:
                    sims = np.dot(cluster_embs, centroid)
                    cohesion = float(np.mean(sims))
                else:
                    cohesion = 0.0
                cohesion_scores[lbl] = round(cohesion, 4)

                if lbl != -1:
                    kw_list = keywords_by_cluster.get(lbl, [])
                    all_kw_words.extend([item["word"] for item in kw_list[:10]])

            # Diversity score: unique keywords / total keywords across meaningful topics
            if all_kw_words:
                keyword_diversity = round(len(set(all_kw_words)) / len(all_kw_words), 4)
            else:
                keyword_diversity = 1.0

            mean_cohesion = float(
                np.mean([v for k, v in cohesion_scores.items() if k != -1])
            ) if num_meaningful_topics > 0 else 0.0

            quality_metrics = {
                "mean_intra_cluster_cohesion": round(mean_cohesion, 4),
                "keyword_diversity_ratio": keyword_diversity,
                "outlier_ratio": round(outlier_count / max(1, corpus_size), 4),
                "cache_stats": cache_stats,
            }

            # 5. DB Persistence
            self._jobs[run_id]["progress"] = 0.90
            self._jobs[run_id]["stage"] = "Persisting discovered topics and review assignments to database"

            db_run = TopicDiscoveryRun(
                id=run_id,
                user_id=user_id,
                status="COMPLETED",
                corpus_size=corpus_size,
                num_topics=num_meaningful_topics,
                outlier_count=outlier_count,
                parameters_json=parameters,
                metrics_json=quality_metrics,
                completed_at=datetime.utcnow(),
            )
            session.add(db_run)
            await session.flush()

            topic_record_map: Dict[int, DiscoveredTopic] = {}
            for lbl in unique_clusters:
                kw_data = keywords_by_cluster.get(lbl, [])
                auto_label = self._generate_auto_label(lbl, kw_data)
                indices = np.where(cluster_labels == lbl)[0]

                # Sentiment breakdown for cluster if predictions have labels
                cluster_reviews = [unique_reviews[idx] for idx in indices]
                sent_counts: Dict[str, int] = {}
                for rev in cluster_reviews:
                    s_lbl = rev.get("sentiment_label")
                    if s_lbl:
                        sent_counts[s_lbl] = sent_counts.get(s_lbl, 0) + 1
                sent_breakdown = {}
                if cluster_reviews and sent_counts:
                    tot_s = sum(sent_counts.values())
                    sent_breakdown = {
                        k: round(v / tot_s, 4) for k, v in sent_counts.items()
                    }

                topic_rec = DiscoveredTopic(
                    run_id=run_id,
                    topic_index=int(lbl),
                    auto_label=auto_label,
                    human_label=None,
                    keywords_json=kw_data,
                    centroid_json=centroids_by_cluster[lbl].tolist(),
                    review_count=int(len(indices)),
                    cohesion_score=cohesion_scores.get(lbl),
                    sentiment_breakdown_json=sent_breakdown,
                    is_archived=False,
                )
                session.add(topic_rec)
                topic_record_map[lbl] = topic_rec

            await session.flush()

            # Create assignments and identify top 3 representative reviews per topic
            assignment_records: List[TopicAssignment] = []
            for lbl in unique_clusters:
                topic_rec = topic_record_map[lbl]
                indices = np.where(cluster_labels == lbl)[0]
                centroid = centroids_by_cluster[lbl]

                distances = []
                for idx in indices:
                    emb = embeddings[idx]
                    cos_dist = float(1.0 - np.dot(emb, centroid))
                    distances.append((idx, cos_dist))

                # Sort by smallest distance to centroid
                distances.sort(key=lambda x: x[1])
                rep_indices: Set[int] = set([d[0] for d in distances[:3]]) if lbl != -1 else set()

                for idx, dist in distances:
                    rev = unique_reviews[idx]
                    assignment = TopicAssignment(
                        run_id=run_id,
                        topic_id=topic_rec.id,
                        prediction_id=rev.get("prediction_id"),
                        review_text=rev["text"],
                        distance_to_centroid=round(dist, 4),
                        is_outlier=(lbl == -1),
                        is_representative=(idx in rep_indices),
                        sentiment_label=rev.get("sentiment_label"),
                        confidence_score=rev.get("confidence_score"),
                    )
                    assignment_records.append(assignment)

            session.add_all(assignment_records)
            await session.commit()

            # Finalize Job State
            completed_time = datetime.utcnow().isoformat()
            self._jobs[run_id]["progress"] = 1.0
            self._jobs[run_id]["status"] = "COMPLETED"
            self._jobs[run_id]["stage"] = "Completed successfully"
            self._jobs[run_id]["num_topics"] = num_meaningful_topics
            self._jobs[run_id]["outlier_count"] = outlier_count
            self._jobs[run_id]["metrics"] = quality_metrics
            self._jobs[run_id]["completed_at"] = completed_time

            # Format Response
            topic_items: List[DiscoveredTopicItem] = []
            for lbl in unique_clusters:
                rec = topic_record_map[lbl]
                kw_objs = [TopicKeywordItem(**k) for k in rec.keywords_json]
                topic_items.append(
                    DiscoveredTopicItem(
                        id=rec.id,
                        run_id=rec.run_id,
                        topic_index=rec.topic_index,
                        auto_label=rec.auto_label,
                        human_label=rec.human_label,
                        display_label=rec.human_label or rec.auto_label,
                        keywords=kw_objs,
                        review_count=rec.review_count,
                        cohesion_score=rec.cohesion_score,
                        sentiment_breakdown=rec.sentiment_breakdown_json,
                        is_archived=rec.is_archived,
                        created_at=rec.created_at.isoformat(),
                        updated_at=rec.updated_at.isoformat(),
                    )
                )

            return TopicListResponse(
                run_id=run_id,
                corpus_size=corpus_size,
                num_topics=num_meaningful_topics,
                outlier_count=outlier_count,
                metrics=quality_metrics,
                topics=topic_items,
            )

        except Exception as e:
            logger.error(f"Topic discovery run {run_id} failed: {e}", exc_info=True)
            self._jobs[run_id]["status"] = "FAILED"
            self._jobs[run_id]["error_message"] = str(e)
            self._jobs[run_id]["stage"] = f"Failed: {str(e)}"
            await session.rollback()
            raise

    async def list_topics(
        self,
        session: AsyncSession,
        run_id: Optional[str] = None,
        include_archived: bool = False,
    ) -> TopicListResponse:
        """List discovered topics for the specified run or latest run."""
        # Find run
        if not run_id:
            stmt_latest = (
                select(TopicDiscoveryRun)
                .where(TopicDiscoveryRun.status == "COMPLETED")
                .order_by(desc(TopicDiscoveryRun.created_at))
                .limit(1)
            )
            res_latest = await session.execute(stmt_latest)
            run_rec = res_latest.scalar_one_or_none()
            if not run_rec:
                return TopicListResponse(
                    run_id="",
                    corpus_size=0,
                    num_topics=0,
                    outlier_count=0,
                    metrics=None,
                    topics=[],
                )
            target_run_id = run_rec.id
        else:
            run_rec = await session.get(TopicDiscoveryRun, run_id)
            if not run_rec:
                raise ValueError(f"Topic discovery run '{run_id}' not found.")
            target_run_id = run_id

        # Query topics
        stmt_topics = (
            select(DiscoveredTopic)
            .where(DiscoveredTopic.run_id == target_run_id)
            .order_by(DiscoveredTopic.topic_index)
        )
        if not include_archived:
            stmt_topics = stmt_topics.where(DiscoveredTopic.is_archived == False)

        res_topics = await session.execute(stmt_topics)
        topic_records = res_topics.scalars().all()

        # Query all available runs for switcher
        stmt_all_runs = (
            select(TopicDiscoveryRun)
            .order_by(desc(TopicDiscoveryRun.created_at))
            .limit(10)
        )
        res_runs = await session.execute(stmt_all_runs)
        all_runs = [
            {
                "id": r.id,
                "status": r.status,
                "corpus_size": r.corpus_size,
                "num_topics": r.num_topics,
                "created_at": r.created_at.isoformat(),
            }
            for r in res_runs.scalars().all()
        ]

        items: List[DiscoveredTopicItem] = []
        for t in topic_records:
            kw_objs = [
                TopicKeywordItem(**k)
                for k in (t.keywords_json if isinstance(t.keywords_json, list) else [])
            ]
            items.append(
                DiscoveredTopicItem(
                    id=t.id,
                    run_id=t.run_id,
                    topic_index=t.topic_index,
                    auto_label=t.auto_label,
                    human_label=t.human_label,
                    display_label=t.human_label or t.auto_label,
                    keywords=kw_objs,
                    review_count=t.review_count,
                    cohesion_score=t.cohesion_score,
                    sentiment_breakdown=t.sentiment_breakdown_json,
                    is_archived=t.is_archived,
                    created_at=t.created_at.isoformat(),
                    updated_at=t.updated_at.isoformat(),
                )
            )

        return TopicListResponse(
            run_id=target_run_id,
            corpus_size=run_rec.corpus_size,
            num_topics=run_rec.num_topics,
            outlier_count=run_rec.outlier_count,
            metrics=run_rec.metrics_json,
            topics=items,
            all_runs=all_runs,
        )

    async def get_topic_reviews(
        self,
        session: AsyncSession,
        topic_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> TopicReviewsResponse:
        """Fetch paginated review assignments belonging to a topic."""
        topic = await session.get(DiscoveredTopic, topic_id)
        if not topic:
            raise ValueError(f"Discovered topic '{topic_id}' not found.")

        # Count total
        count_stmt = select(func.count(TopicAssignment.id)).where(
            TopicAssignment.topic_id == topic_id
        )
        total = (await session.execute(count_stmt)).scalar() or 0

        # Paginated fetch (representatives first, then by distance)
        offset = (max(1, page) - 1) * page_size
        stmt = (
            select(TopicAssignment)
            .where(TopicAssignment.topic_id == topic_id)
            .order_by(
                desc(TopicAssignment.is_representative),
                TopicAssignment.distance_to_centroid,
            )
            .offset(offset)
            .limit(page_size)
        )
        rows = (await session.execute(stmt)).scalars().all()

        items = [
            TopicReviewAssignmentItem(
                id=r.id,
                topic_id=r.topic_id,
                prediction_id=r.prediction_id,
                review_text=r.review_text,
                distance_to_centroid=r.distance_to_centroid,
                is_outlier=r.is_outlier,
                is_representative=r.is_representative,
                sentiment_label=r.sentiment_label,
                confidence_score=r.confidence_score,
                created_at=r.created_at.isoformat(),
            )
            for r in rows
        ]

        display_name = topic.human_label or topic.auto_label
        return TopicReviewsResponse(
            topic_id=topic.id,
            topic_name=display_name,
            topic_index=topic.topic_index,
            total_reviews=total,
            page=page,
            page_size=page_size,
            reviews=items,
        )

    async def rename_topic(
        self,
        session: AsyncSession,
        topic_id: str,
        human_label: str,
        user_id: Optional[str] = None,
    ) -> DiscoveredTopicItem:
        """Update curated human label for a topic and write audit record."""
        topic = await session.get(DiscoveredTopic, topic_id)
        if not topic:
            raise ValueError(f"Topic '{topic_id}' not found.")

        old_label = topic.human_label or topic.auto_label
        topic.human_label = human_label.strip()
        topic.updated_at = datetime.utcnow()

        audit = TopicAuditLog(
            topic_id=topic_id,
            user_id=user_id,
            action="RENAME",
            details_json={"old_label": old_label, "new_label": topic.human_label},
        )
        session.add(audit)
        await session.commit()
        await session.refresh(topic)

        kw_objs = [TopicKeywordItem(**k) for k in (topic.keywords_json or [])]
        return DiscoveredTopicItem(
            id=topic.id,
            run_id=topic.run_id,
            topic_index=topic.topic_index,
            auto_label=topic.auto_label,
            human_label=topic.human_label,
            display_label=topic.human_label,
            keywords=kw_objs,
            review_count=topic.review_count,
            cohesion_score=topic.cohesion_score,
            sentiment_breakdown=topic.sentiment_breakdown_json,
            is_archived=topic.is_archived,
            created_at=topic.created_at.isoformat(),
            updated_at=topic.updated_at.isoformat(),
        )

    async def merge_topics(
        self,
        session: AsyncSession,
        source_topic_id: str,
        target_topic_id: str,
        new_label: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> DiscoveredTopicItem:
        """Merge source topic into target topic, archive source, and log audit."""
        if source_topic_id == target_topic_id:
            raise ValueError("Cannot merge a topic into itself.")

        source = await session.get(DiscoveredTopic, source_topic_id)
        target = await session.get(DiscoveredTopic, target_topic_id)
        if not source or not target:
            raise ValueError("Source or target topic not found.")

        # Reassign reviews from source to target
        stmt_assignments = select(TopicAssignment).where(
            TopicAssignment.topic_id == source_topic_id
        )
        source_assignments = (await session.execute(stmt_assignments)).scalars().all()

        for a in source_assignments:
            a.topic_id = target_topic_id

        target.review_count += source.review_count
        source.review_count = 0
        source.is_archived = True
        source.updated_at = datetime.utcnow()

        if new_label:
            target.human_label = new_label.strip()
        target.updated_at = datetime.utcnow()

        audit = TopicAuditLog(
            topic_id=target_topic_id,
            user_id=user_id,
            action="MERGE",
            details_json={
                "merged_from_id": source_topic_id,
                "reassigned_count": len(source_assignments),
                "new_label": target.human_label,
            },
        )
        session.add(audit)
        await session.commit()
        await session.refresh(target)

        kw_objs = [TopicKeywordItem(**k) for k in (target.keywords_json or [])]
        return DiscoveredTopicItem(
            id=target.id,
            run_id=target.run_id,
            topic_index=target.topic_index,
            auto_label=target.auto_label,
            human_label=target.human_label,
            display_label=target.human_label or target.auto_label,
            keywords=kw_objs,
            review_count=target.review_count,
            cohesion_score=target.cohesion_score,
            sentiment_breakdown=target.sentiment_breakdown_json,
            is_archived=target.is_archived,
            created_at=target.created_at.isoformat(),
            updated_at=target.updated_at.isoformat(),
        )

    async def archive_topic(
        self,
        session: AsyncSession,
        topic_id: str,
        user_id: Optional[str] = None,
    ) -> DiscoveredTopicItem:
        """Mark a topic as archived and write audit log."""
        topic = await session.get(DiscoveredTopic, topic_id)
        if not topic:
            raise ValueError(f"Topic '{topic_id}' not found.")

        topic.is_archived = True
        topic.updated_at = datetime.utcnow()

        audit = TopicAuditLog(
            topic_id=topic_id,
            user_id=user_id,
            action="ARCHIVE",
            details_json={"archived": True},
        )
        session.add(audit)
        await session.commit()
        await session.refresh(topic)

        kw_objs = [TopicKeywordItem(**k) for k in (topic.keywords_json or [])]
        return DiscoveredTopicItem(
            id=topic.id,
            run_id=topic.run_id,
            topic_index=topic.topic_index,
            auto_label=topic.auto_label,
            human_label=topic.human_label,
            display_label=topic.human_label or topic.auto_label,
            keywords=kw_objs,
            review_count=topic.review_count,
            cohesion_score=topic.cohesion_score,
            sentiment_breakdown=topic.sentiment_breakdown_json,
            is_archived=topic.is_archived,
            created_at=topic.created_at.isoformat(),
            updated_at=topic.updated_at.isoformat(),
        )

    async def compare_runs(
        self,
        session: AsyncSession,
        run_id_a: str,
        run_id_b: str,
    ) -> TopicComparisonResponse:
        """Compare topic centroids across two discovery runs to identify drift and candidate emerging themes."""
        # Query non-outlier, non-archived topics from both runs
        stmt_a = (
            select(DiscoveredTopic)
            .where(
                DiscoveredTopic.run_id == run_id_a,
                DiscoveredTopic.topic_index != -1,
                DiscoveredTopic.is_archived == False,
            )
        )
        stmt_b = (
            select(DiscoveredTopic)
            .where(
                DiscoveredTopic.run_id == run_id_b,
                DiscoveredTopic.topic_index != -1,
                DiscoveredTopic.is_archived == False,
            )
        )
        topics_a = (await session.execute(stmt_a)).scalars().all()
        topics_b = (await session.execute(stmt_b)).scalars().all()

        matched_topics: List[MatchedTopicPair] = []
        emerging_in_b: List[CandidateEmergingTopic] = []
        faded_in_a: List[Dict[str, Any]] = []

        if not topics_a or not topics_b:
            return TopicComparisonResponse(
                run_id_a=run_id_a,
                run_id_b=run_id_b,
                matched_topics=[],
                emerging_topics_in_b=[],
                faded_topics_in_a=[],
            )

        # Build centroid matrices
        mat_a = np.array([t.centroid_json for t in topics_a], dtype=np.float32)
        mat_b = np.array([t.centroid_json for t in topics_b], dtype=np.float32)

        # Pairwise cosine similarity: (len_a, len_b)
        sim_matrix = np.dot(mat_a, mat_b.T)

        SIMILARITY_THRESHOLD = 0.45

        # Check topics in B against A
        for idx_b, topic_b in enumerate(topics_b):
            sims_with_a = sim_matrix[:, idx_b]
            max_sim_idx = int(np.argmax(sims_with_a))
            max_sim = float(sims_with_a[max_sim_idx])

            if max_sim >= SIMILARITY_THRESHOLD:
                topic_a = topics_a[max_sim_idx]
                matched_topics.append(
                    MatchedTopicPair(
                        topic_a_id=topic_a.id,
                        topic_a_name=topic_a.human_label or topic_a.auto_label,
                        topic_b_id=topic_b.id,
                        topic_b_name=topic_b.human_label or topic_b.auto_label,
                        similarity=round(max_sim, 4),
                    )
                )
            else:
                kw_objs = [
                    TopicKeywordItem(**k)
                    for k in (topic_b.keywords_json if isinstance(topic_b.keywords_json, list) else [])
                ]
                emerging_in_b.append(
                    CandidateEmergingTopic(
                        topic_id=topic_b.id,
                        topic_name=topic_b.human_label or topic_b.auto_label,
                        topic_index=topic_b.topic_index,
                        review_count=topic_b.review_count,
                        max_similarity_to_prior_run=round(max_sim, 4),
                        keywords=kw_objs,
                        uncertainty_note=(
                            "Candidate emerging theme - requires human review due to unsupervised cluster drift"
                        ),
                    )
                )

        # Check topics in A with low alignment in B
        for idx_a, topic_a in enumerate(topics_a):
            sims_with_b = sim_matrix[idx_a, :]
            max_sim = float(np.max(sims_with_b))
            if max_sim < SIMILARITY_THRESHOLD:
                faded_in_a.append(
                    {
                        "topic_id": topic_a.id,
                        "topic_name": topic_a.human_label or topic_a.auto_label,
                        "topic_index": topic_a.topic_index,
                        "max_similarity_in_new_run": round(max_sim, 4),
                    }
                )

        return TopicComparisonResponse(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            matched_topics=matched_topics,
            emerging_topics_in_b=emerging_in_b,
            faded_topics_in_a=faded_in_a,
        )


topic_service = TopicDiscoveryService.get_instance()
