# 📖 SentimentAI Enterprise — API Reference Guide

FastAPI delivers interactive OpenAPI documentation at `/docs` (Swagger) and `/redoc` (ReDoc). This document summarizes key REST endpoints, request parameters, response schemas, and authentication methods.

---

## 🔐 Authentication Methods

All protected endpoints require HTTP Bearer JWT authentication:

```http
Authorization: Bearer <YOUR_ACCESS_TOKEN>
```

---

## 📡 Key API Endpoints

### 1. Authentication (`/api/v1/auth`)

#### `POST /api/v1/auth/register`
Register a new user account.
- **Request Body:**
  ```json
  {
    "email": "user@example.com",
    "password": "StrongPassword123!",
    "full_name": "Demo User"
  }
  ```
- **Response (201 Created):**
  ```json
  {
    "id": "usr_94821a",
    "email": "user@example.com",
    "full_name": "Demo User",
    "role": "USER",
    "is_active": true
  }
  ```

#### `POST /api/v1/auth/login`
Authenticate and obtain JWT token pair.
- **Request Body:** Form data `username=user@example.com&password=StrongPassword123!`
- **Response (200 OK):**
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni...",
    "refresh_token": "def456...",
    "token_type": "bearer",
    "expires_in": 1800
  }
  ```

---

### 2. Real-Time Sentiment Prediction (`/api/v1/predict`)

#### `POST /api/v1/predict`
Classify text sentiment with token-level gradient attribution.
- **Request Body:**
  ```json
  {
    "text_content": "The support team resolved my issue within minutes. Outstanding service!"
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "id": "pred_84920a",
    "text_content": "The support team resolved my issue within minutes. Outstanding service!",
    "sentiment": "POSITIVE",
    "confidence": 0.964,
    "probabilities": {
      "positive": 0.964,
      "neutral": 0.021,
      "negative": 0.015
    },
    "explainability_tokens": [
      { "token": "resolved", "score": 0.42 },
      { "token": "support", "score": 0.38 },
      { "token": "outstanding", "score": 0.35 }
    ],
    "latency_ms": 18.5,
    "created_at": "2026-08-05T13:30:00Z"
  }
  ```

---

### 3. Batch Ingestion (`/api/v1/batch`)

#### `POST /api/v1/batch/upload`
Upload a CSV or TXT file for asynchronous micro-batch processing.
- **Form Data:** `file: @reviews.csv`
- **Response (202 Accepted):**
  ```json
  {
    "job_id": "job_7712a",
    "filename": "reviews.csv",
    "total_rows": 500,
    "status": "PROCESSING",
    "created_at": "2026-08-05T13:30:00Z"
  }
  ```

#### `GET /api/v1/batch/status/{job_id}`
Get real-time batch processing progress.
- **Response (200 OK):**
  ```json
  {
    "job_id": "job_7712a",
    "status": "COMPLETED",
    "processed_rows": 500,
    "total_rows": 500,
    "progress_percentage": 100.0,
    "throughput_items_per_sec": 1150.4
  }
  ```

---

### 4. Executive Analytics (`/api/v1/analytics`)

#### `GET /api/v1/analytics/overview`
Get aggregate system KPIs and sentiment distribution metrics.
- **Response (200 OK):**
  ```json
  {
    "total_predictions": 12850,
    "positive_count": 7800,
    "neutral_count": 3100,
    "negative_count": 1950,
    "avg_latency_ms": 22.4,
    "avg_confidence": 0.921
  }
  ```

---

### 5. Telemetry & Health (`/metrics` & `/api/v1/health`)

#### `GET /metrics`
Exposes Prometheus operational metrics.

#### `GET /api/v1/health/system`
Returns operational status of CPU, Database, RAM, and Model Engine.

---

### 6. Aspect-Based Sentiment Analysis (`/api/v1/absa`)

Two-stage modular neural pipeline for explicit aspect term extraction (ATE BIO tagging) and aspect-conditioned polarity classification (ALSC cross-encoder).

> **Architectural Note on Categories:** The `aspect_category` field is a deterministic rule-based category normalization tag (e.g. `CAMERA`, `BATTERY`, `DISPLAY`, `FOOD`, `SERVICE`) applied post-inference for analytics aggregation. It is a heuristic taxonomy label, not a neural classification output.
> **Fail-Fast Readiness:** If ABSA neural models or tokenizers fail to load, `POST /api/v1/absa/analyze` and `POST /api/v1/absa/batch` fail fast with HTTP 503 (`ABSA neural models are unavailable or not loaded`), preserving the independent stability of `/api/v1/predict`.

#### `POST /api/v1/absa/analyze`
Decompose a review into aspect spans and independent polarities.
- **Request Body:**
  ```json
  {
    "text": "The camera is excellent, but the battery drains quickly.",
    "save_to_db": true
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "text": "The camera is excellent, but the battery drains quickly.",
    "aspect_count": 2,
    "overall_sentiment_preview": "NEUTRAL",
    "latency_ms": 18.5,
    "aspects": [
      {
        "aspect": "camera",
        "aspect_category": "HARDWARE",
        "is_implicit": false,
        "start_char": 4,
        "end_char": 10,
        "sentiment": "POSITIVE",
        "confidence": 0.9621,
        "probabilities": {
          "POSITIVE": 0.9621,
          "NEUTRAL": 0.0215,
          "NEGATIVE": 0.0164
        },
        "evidence_text": "camera",
        "model_name": "DeBERTa-v3-ABSA-Modular",
        "model_version": "v1.0.0",
        "created_at": "2026-11-14T11:45:00Z"
      },
      {
        "aspect": "battery",
        "aspect_category": "HARDWARE",
        "is_implicit": false,
        "start_char": 33,
        "end_char": 40,
        "sentiment": "NEGATIVE",
        "confidence": 0.9340,
        "probabilities": {
          "POSITIVE": 0.0125,
          "NEUTRAL": 0.0535,
          "NEGATIVE": 0.9340
        },
        "evidence_text": "battery",
        "model_name": "DeBERTa-v3-ABSA-Modular",
        "model_version": "v1.0.0",
        "created_at": "2026-11-14T11:45:00Z"
      }
    ]
  }
  ```

#### `POST /api/v1/absa/batch`
Perform aspect extraction and classification across multiple review texts.
- **Request Body:**
  ```json
  {
    "texts": [
      "Great screen, but terrible customer support.",
      "The food was delicious, but noisy atmosphere."
    ],
    "save_to_db": false
  }
  ```

#### `GET /api/v1/absa/analytics`
Retrieve aggregated aspect frequencies, polarity ratios, top positive/negative aspect terms, and audit records.
- **Query Parameters:** `limit` (int, default: 50)
- **Response (200 OK):** AspectAnalyticsResponse with frequency breakdown and net sentiment scores.

#### `GET /api/v1/absa/metadata`
Expose evaluated benchmark metrics on the MAMS-ATSA Test Split (100-sentence subset, 266 aspects), architecture specifications, baseline comparisons, and calibration scores.
- **Response (200 OK):**
  - ALSC Accuracy: 76.32% (Macro F1: 0.7530, Brier Score: 0.3778)
  - Baseline Comparison: 47.37% Accuracy, 0.3348 Macro F1 (+124.9% improvement)
  - ATE F1 Score: 0.7671 (Precision: 0.8233, Recall: 0.7180)

---

### 7. Emotion Detection & CX Taxonomy (`/api/v1/emotion`)

Multi-label neural emotion detection across 28 GoEmotions classes with per-class calibrated probability thresholds, aggregated to the 6-Category Customer Experience (CX) Taxonomy.

> **Fail-Fast Readiness:** If the RoBERTa multi-label weights or tokenizer are unavailable, `POST /api/v1/emotion/analyze` and `POST /api/v1/emotion/batch` fail fast with HTTP 503 (`Emotion neural models are unavailable or not loaded`), preserving the independent stability of `/api/v1/predict` and `/api/v1/absa/*`.

#### `POST /api/v1/emotion/analyze`
Analyze customer feedback for fine-grained emotional signals and CX category rollup.
- **Request Body:**
  ```json
  {
    "text": "This device makes me furious and annoyed, but I am very grateful for the quick refund.",
    "save_to_db": true
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "text": "This device makes me furious and annoyed, but I am very grateful for the quick refund.",
    "primary_emotion": "Satisfaction / Gratitude",
    "confidence": 0.6556,
    "detected_cx_categories": [
      {
        "category": "Satisfaction / Gratitude",
        "score": 0.6556,
        "threshold": 0.20
      },
      {
        "category": "Frustration / Anger",
        "score": 0.3229,
        "threshold": 0.20
      }
    ],
    "detected_emotions": [
      {
        "label": "gratitude",
        "probability": 0.6556,
        "threshold": 0.20,
        "cx_category": "Satisfaction / Gratitude"
      },
      {
        "label": "annoyance",
        "probability": 0.3229,
        "threshold": 0.20,
        "cx_category": "Frustration / Anger"
      }
    ],
    "raw_scores": {
      "admiration": 0.0412,
      "anger": 0.1347,
      "annoyance": 0.3229,
      "gratitude": 0.6556,
      "...": "28 total classes"
    },
    "model_name": "RoBERTa-GoEmotions-MultiLabel",
    "model_version": "v1.0.0",
    "latency_ms": 28.4,
    "created_at": "2026-11-14T14:05:00Z"
  }
  ```

#### `POST /api/v1/emotion/batch`
Analyze a list of review texts for multi-label emotions with micro-batching `<= 16`.
- **Request Body:**
  ```json
  {
    "texts": [
      "Super excited about this new feature!",
      "Terrible customer service, very disappointed."
    ],
    "save_to_db": false
  }
  ```

#### `GET /api/v1/emotion/analytics`
Retrieve aggregated CX category counts, proportions, top raw GoEmotions signals, and recent feedback records.
- **Query Parameters:** `limit` (int, default: 50)
- **Response (200 OK):** EmotionAnalyticsResponse.

#### `GET /api/v1/emotion/metadata`
Expose benchmark metrics on Google Research GoEmotions test split, per-class calibrated thresholds, Brier calibration, Expected Calibration Error (ECE), and frequency baseline comparison.
- **Response (200 OK):**
  - Micro F1: 0.5895 (vs Frequency Baseline: 0.2829, +108.4% improvement)
  - Macro F1: 0.3714
  - Hamming Loss: 0.0371
  - Mean Brier Score: 0.0239 (ECE: 0.0079)

---

### 5. Automatic Topic Discovery (`/api/v1/topics`)

Collection-level unsupervised topic modeling executing strictly on **CPU** (0 MB GPU VRAM) via `sentence-transformers/all-MiniLM-L6-v2`, dimensionality reduction (UMAP/TruncatedSVD), density clustering (HDBSCAN), and class-based TF-IDF (c-TF-IDF) representation.

#### `POST /api/v1/topics/discover`
Initiate a collection-level topic discovery run over reviews (minimum corpus threshold: $\ge 15$ unique reviews).
- **Query Parameters:**
  - `sync` (bool, default: `true`): Run synchronously for immediate results or queue as background job (`sync=false`).
- **Request Body:**
  ```json
  {
    "source": "custom_texts",
    "custom_texts": [
      "The battery easily lasts two full days with heavy usage.",
      "Fast charging takes less than thirty minutes to reach eighty percent.",
      "The speakers sound tinny and distorted at high volumes."
    ],
    "min_cluster_size": 3,
    "n_neighbors": 15,
    "n_components": 5
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "run_id": "3b2f7a4e-15c9-496b-b209-5e2de22ec437",
    "corpus_size": 24,
    "num_topics": 4,
    "outlier_count": 2,
    "metrics": {
      "mean_intra_cluster_cohesion": 0.6316,
      "keyword_diversity_ratio": 1.0,
      "outlier_ratio": 0.0833,
      "cache_stats": { "hits": 18, "misses": 6, "total": 24, "hit_rate_pct": 75.0 }
    },
    "topics": [
      {
        "id": "top_001",
        "run_id": "3b2f7a4e-15c9-496b-b209-5e2de22ec437",
        "topic_index": 0,
        "auto_label": "Battery / Charging / Endurance",
        "human_label": null,
        "display_label": "Battery / Charging / Endurance",
        "keywords": [
          { "word": "battery", "score": 0.42 },
          { "word": "charging", "score": 0.35 }
        ],
        "review_count": 8,
        "cohesion_score": 0.68,
        "sentiment_breakdown": { "POSITIVE": 0.75, "NEGATIVE": 0.25 },
        "is_archived": false
      },
      {
        "id": "top_outlier",
        "run_id": "3b2f7a4e-15c9-496b-b209-5e2de22ec437",
        "topic_index": -1,
        "auto_label": "Outliers / Unassigned",
        "human_label": null,
        "display_label": "Outliers / Unassigned",
        "keywords": [],
        "review_count": 2,
        "cohesion_score": null,
        "sentiment_breakdown": {},
        "is_archived": false
      }
    ]
  }
  ```

#### `GET /api/v1/topics`
Retrieve discovered topics for a specific discovery run (or the latest completed run).
- **Query Parameters:**
  - `run_id` (string, optional): Run UUID. Defaults to latest completed run.
  - `include_archived` (bool, default: `false`): Include human-archived topics.

#### `GET /api/v1/topics/{topic_id}/reviews`
Fetch paginated review assignments for a topic, highlighting top 3 representative reviews closest to centroid.
- **Query Parameters:** `page` (int, default: 1), `page_size` (int, default: 20)

#### `PATCH /api/v1/topics/{topic_id}/rename`
Set a human-curated title for a discovered topic and persist an immutable record in `topic_audit_logs`.
- **Request Body:**
  ```json
  {
    "human_label": "Battery Endurance & Fast Charging"
  }
  ```

#### `POST /api/v1/topics/merge`
Consolidate two related topics into a target topic. Reassigns member reviews, updates review counts, archives the source topic, and records an audit log.
- **Request Body:**
  ```json
  {
    "source_topic_id": "top_002",
    "target_topic_id": "top_001",
    "new_label": "Consolidated Hardware & Power"
  }
  ```

#### `PATCH /api/v1/topics/{topic_id}/archive`
Deactivate a topic from active views (`is_archived = true`) and record an audit log.

#### `GET /api/v1/topics/compare`
Compare topic centroids across two discovery runs to identify matched topics ($\ge 0.45$) and candidate emerging themes ($< 0.45$) with mandatory uncertainty disclaimer.
- **Query Parameters:** `run_id_a` (UUID string), `run_id_b` (UUID string)
- **Response (200 OK):**
  - `matched_topics`: Centroid similarity $\ge 0.45$ with review volume delta.
  - `emerging_topics_in_b`: Novel clusters with max similarity $< 0.45$, annotated with `"uncertainty_note": "Candidate emerging theme - requires human review due to unsupervised cluster drift"`.

#### `GET /api/v1/topics/metadata`
Expose embedding model specifications (`sentence-transformers/all-MiniLM-L6-v2`), execution target (CPU, 0 MB VRAM), and benchmark evaluation metrics (UMAP+HDBSCAN vs K-Means vs Agglomerative).

---

### 6. Unified Multi-Task Intelligence & Cross-Feature Analytics (`/api/v1/intelligence`)

Orchestrates all platform models (TensorFlow overall sentiment, DeBERTa-v3 ABSA, RoBERTa-GoEmotions CX taxonomy, and MiniLM frozen-centroid topic projection) with strict fault isolation, per-capability status semantics, batch micro-chunking, and SQL cross-feature aggregations.

#### `POST /api/v1/intelligence/analyze`
Execute unified multi-task inference for a single review.
- **Request Body:**
  ```json
  {
    "text": "The battery life lasts two full days, but customer service was completely unhelpful.",
    "save_to_db": true,
    "project_topic": true
  }
  ```
- **Response (200 OK):**
  ```json
  {
    "review_id": "a823b1c2-3d4e-4f5a-8b6c-7d8e9f0a1b2c",
    "text_content": "The battery life lasts two full days, but customer service was completely unhelpful.",
    "overall_sentiment": {
      "sentiment": "NEUTRAL",
      "confidence": 0.6842,
      "probabilities": { "positive": 0.28, "neutral": 0.6842, "negative": 0.0358 },
      "latency_ms": 12.4
    },
    "aspects": [
      {
        "aspect_term": "battery life",
        "aspect_category": "BATTERY",
        "sentiment": "POSITIVE",
        "confidence": 0.9421,
        "start_char": 4,
        "end_char": 16,
        "evidence_text": "The battery life lasts two full days"
      },
      {
        "aspect_term": "customer service",
        "aspect_category": "SERVICE",
        "sentiment": "NEGATIVE",
        "confidence": 0.9812,
        "start_char": 42,
        "end_char": 58,
        "evidence_text": "customer service was completely unhelpful"
      }
    ],
    "emotions": {
      "primary_cx_category": "Frustration / Anger",
      "confidence_score": 0.8845,
      "detected_emotions": [
        { "emotion": "annoyance", "score": 0.8845 },
        { "emotion": "disappointment", "score": 0.7231 }
      ],
      "all_probabilities": { "annoyance": 0.8845, "admiration": 0.124 }
    },
    "topic": {
      "topic_id": "top_001_battery",
      "topic_index": 0,
      "topic_label": "Battery Performance",
      "similarity_score": 0.8234,
      "topic_run_version": "run_001_v1",
      "is_outlier": false
    },
    "capability_status": {
      "sentiment": { "status": "completed", "model_name": "TensorFlow-BiLSTM-SelfAttention", "model_version": "v1.0.0", "latency_ms": 12.4 },
      "absa": { "status": "completed", "model_name": "DeBERTa-v3-ABSA", "model_version": "v1.0.0", "latency_ms": 42.1 },
      "emotion": { "status": "completed", "model_name": "RoBERTa-GoEmotions-MultiLabel", "model_version": "v1.0.0", "latency_ms": 31.8 },
      "topics": { "status": "completed", "model_name": "all-MiniLM-L6-v2 + CentroidProjection", "model_version": "run_001_v1", "latency_ms": 14.5 }
    },
    "total_latency_ms": 100.8,
    "created_at": "2026-11-15T08:30:00Z"
  }
  ```

#### `POST /api/v1/intelligence/batch`
Process a batch of reviews (up to 500) in memory-safe micro-chunks (`<= 16` GPU batch, `<= 32` CPU embeddings). Identical reviews within the payload are automatically deduplicated.
- **Query Parameters:** `sync` (bool, default `true` for $\le 25$ items).
- **Request Body:**
  ```json
  {
    "reviews": [
      "Great sound quality with deep bass.",
      "Delivery arrived three days late and package was damaged."
    ],
    "save_to_db": true,
    "sync": true
  }
  ```

#### `GET /api/v1/intelligence/batch/{job_id}`
Poll progress of an asynchronous batch job (`PENDING`, `PROCESSING`, `COMPLETED`, `FAILED`). Reports `progress_pct`, `processed_records`, `succeeded_records`, `failed_records`, and `partial_records`.

#### `GET /api/v1/intelligence/analytics`
Retrieve real SQL-aggregated cross-feature analytics across topics, aspects, emotions, and overall sentiment.
- **Query Parameters:**
  - `min_sample_size` (int, default: 1): Minimum cell sample count threshold.
  - `start_date`, `end_date` (ISO date strings, optional).
- **Response Structure:**
  - `aspect_sentiment_by_topic`: Aspect polarities grouped by discovered topic.
  - `emotion_by_overall_sentiment`: CX emotion categories relative to overall sentiment.
  - `emotion_by_aspect_category`: CX emotions grouped by aspect term/category.
  - `topic_sentiment_and_emotion_summary`: Per-topic review volume, sentiment distribution, and dominant emotion.
  - `negative_aspect_trends`: Daily frequency and negative rate of problematic aspect terms.
  - `low_sample_warning`: Automatically flagged `true` on cells with denominator $< 5$.
  - `methodology_note`: Discloses that groupings reflect statistical co-occurrence rather than causal proof.

#### `GET /api/v1/intelligence/evidence`
Retrieve actual review evidence and linked predictions matching cross-feature filters.
- **Query Parameters:**
  - `topic_id` (string, optional)
  - `aspect_term` (string, optional)
  - `emotion_category` (string, optional)
  - `sentiment` (string, optional: POSITIVE, NEUTRAL, NEGATIVE)
  - `page` (int, default: 1), `page_size` (int, default: 20)



