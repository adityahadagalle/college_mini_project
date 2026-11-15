# 🏗️ SentimentAI Enterprise — Software Architecture & Design

This document details the architectural design, system boundaries, data flow, component interactions, and Clean Architecture patterns implemented in SentimentAI Enterprise across all 6 production phases.

---

## 1. 📐 System Context & Component Diagram

```mermaid
graph TD
    Client["Client Web Browser<br/>(React 18 + TypeScript + Tailwind SPA)"] -->|HTTPS / REST API| Gateway["API Gateway / Reverse Proxy<br/>(Port 80 / 443)"]
    Gateway -->|Forward /api/v1| Backend["FastAPI Core Application<br/>(Uvicorn ASGI - Port 8000)"]
    
    subgraph "Backend Clean Architecture & Domain Services"
        Backend --> Middlewares["Middleware Stack<br/>(CORS, Rate Limit, Auth, Audit Logging)"]
        Middlewares --> Routers["API V1 Routers<br/>(/predict, /absa, /emotion, /topics, /intelligence)"]
        
        Routers --> IntelOrchestrator["Unified Intelligence Orchestrator<br/>(intelligence_service.py)"]
        
        IntelOrchestrator --> SentService["Overall Sentiment Service<br/>(TF BiLSTM + Self-Attention)"]
        IntelOrchestrator --> ABSAService["ABSA Service<br/>(DeBERTa-v3 ATE + ALSC)"]
        IntelOrchestrator --> EmoService["Emotion Service<br/>(RoBERTa-GoEmotions CX Taxonomy)"]
        IntelOrchestrator --> TopicService["Topic Discovery Service<br/>(MiniLM + HDBSCAN + c-TF-IDF)"]
        
        IntelOrchestrator --> Repositories["Data Repositories<br/>(SQLAlchemy Async ORM)"]
    end
    
    subgraph "Hardware & Execution Engines"
        SentService --> TFEngine["TensorFlow 2.x C++ Engine<br/>(Local .keras / SavedModel)"]
        ABSAService --> PyTorchGPU["PyTorch Transformer Engine<br/>(DeBERTa-v3 ATE + ALSC)"]
        EmoService --> PyTorchGPU
        TopicService --> CPUEmbed["CPU Vector Pipeline<br/>(MiniLM-L6-v2 + HDBSCAN)"]
    end
    
    Repositories --> DB[("Database Engine<br/>(SQLite / PostgreSQL)")]
    Backend --> Prometheus["Prometheus Telemetry<br/>(/metrics Endpoint)"]
```

---

## 2. 🔁 Multi-Task Unified Review Pipeline (`/api/v1/intelligence/analyze`)

The platform integrates four distinct neural/statistical engines under a single transactional unit of work, strictly maintaining fault isolation and linking records under a unified `review_id` (`predictions.id`).

```mermaid
sequenceDiagram
    autonumber
    actor Client as Web Client / API Consumer
    participant API as Intelligence Router (/api/v1/intelligence/analyze)
    participant Orch as Intelligence Service (intelligence_service.py)
    participant TF as TensorFlow BiLSTM Engine
    participant ABSA as DeBERTa-v3 ABSA Service
    participant EMO as RoBERTa Emotion Service
    participant TOP as Topic Service (Centroid Projection)
    participant DB as Async Database Session

    Client->>API: POST /api/v1/intelligence/analyze { text, save_to_db, project_topic }
    API->>Orch: analyze_review(text, save_to_db=true, project_topic=true)
    
    par Task 1: Overall Sentiment
        Orch->>TF: predict_single(text)
        TF-->>Orch: Polarity (POS/NEU/NEG), Confidence, Latency
    and Task 2: Aspect-Based Sentiment (ABSA)
        Orch->>ABSA: analyze(text)
        Note over ABSA: 1. ATE Token Classifier (BIO spans)<br/>2. ALSC Cross-Encoder per aspect
        ABSA-->>Orch: Detected Aspects [{aspect, sentiment, conf, start, end}]
    and Task 3: Emotion Detection
        Orch->>EMO: analyze(text)
        Note over EMO: 28 GoEmotions Sigmoid heads<br/>Mapped to 6-category CX taxonomy
        EMO-->>Orch: Primary CX Category, Raw Signals, Latency
    and Task 4: Topic Projection (Optional)
        Orch->>TOP: project_review_to_topic(text)
        Note over TOP: MiniLM-L6-v2 CPU embedding<br/>Cosine match vs frozen centroids (>=0.45)
        TOP-->>Orch: Topic Match / Outlier (-1), Similarity
    end
    
    opt save_to_db is True
        Orch->>DB: Begin Transaction
        Orch->>DB: INSERT into predictions (id, text, sentiment, confidence)
        Orch->>DB: INSERT into review_aspects (prediction_id, aspect_term, sentiment, start/end)
        Orch->>DB: INSERT into review_emotions (prediction_id, primary_cx_category, scores)
        Orch->>DB: INSERT into topic_assignments (prediction_id, topic_id, similarity)
        Orch->>DB: Commit Transaction
    end
    
    Orch-->>API: UnifiedReviewIntelligenceDTO
    API-->>Client: HTTP 200 OK (Synchronized Multi-Task Intelligence)
```

---

## 3. 🔍 Collection-Level Automatic Topic Discovery (`/api/v1/topics/discover`)

Unlike per-review classifiers, topic discovery is an unsupervised collection-level pipeline executed strictly on CPU ($0\text{ MB}$ GPU VRAM consumed):

```mermaid
graph TD
    Corpus["Review Corpus<br/>(N >= 15 Unique Reviews)"] --> SHA["SHA-256 Text Normalization & Cache Check"]
    
    subgraph "Deterministic Cache Layer"
        SHA -->|Cache Hit| DBVec["review_embeddings Table"]
        SHA -->|Cache Miss| MiniLM["all-MiniLM-L6-v2 (CPU Mean-Pooling)"]
        MiniLM -->|Save Dense Vectors| DBVec
    end
    
    DBVec --> DimRed["Manifold Dimensionality Reduction<br/>(UMAP / TruncatedSVD to d=5)"]
    DimRed --> HDBSCAN["Density-Based Clustering<br/>(sklearn.cluster.HDBSCAN)"]
    
    HDBSCAN --> NoiseFilter{"Is Core Density?"}
    NoiseFilter -->|No| Outlier["Topic -1: Outliers / Unassigned<br/>(Sparse or Anomalous Complaints)"]
    NoiseFilter -->|Yes| Cluster["Dense Topic Clusters (0 .. K-1)"]
    
    Cluster --> cTFIDF["Class-Based c-TF-IDF<br/>W(t,c) = TF(t,c) * log(1 + A / f_t)"]
    cTFIDF --> AutoLabel["Automated 3-5 Keyword Label<br/>(e.g., 'battery, charging, drain')"]
    
    Cluster --> Centroids["Normalized 384-d Centroids<br/>mu_c = sum(e_i) / ||sum(e_i)||"]
    Centroids --> RepReviews["Top-3 Representative Reviews<br/>(Ranked by Euclidean Distance)"]
    
    AutoLabel --> HumanGov["Human-in-the-Loop Governance<br/>(Rename, Merge, Archive, Audit Logs)"]
```

---

## 4. 🏛️ Layered Clean Architecture

The platform strictly isolates responsibilities across 4 discrete layers:

```
+---------------------------------------------------------------------------------------+
|                                  PRESENTATION LAYER                                   |
|   FastAPI Routers: predict.py, absa.py, emotion.py, topics.py, intelligence.py        |
|   Strict Pydantic v2 Request/Response Validation, OpenTelemetry Instrumentation       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             INTELLIGENCE ORCHESTRATION LAYER                          |
|   intelligence_service.py:                                                            |
|   - Multi-task coordination with independent capability fault isolation               |
|   - Micro-chunked batch processing (<=16 GPU batch, <=32 CPU embeddings)             |
|   - Real-time frozen centroid cosine projection (>=0.45 threshold)                   |
|   - Cross-feature SQL aggregation analytics (aspect x topic x emotion x sentiment)    |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                SPECIALIST DOMAIN SERVICES                             |
|   - tf_engine: TensorFlow 2.x BiLSTM, Self-Attention, GradientTape XAI                |
|   - absa_service: DeBERTa-v3 ATE Token Classifier + ALSC Cross-Encoder                |
|   - emotion_service: RoBERTa-GoEmotions Multi-Label Classifier (28 -> 6 CX Taxonomy)  |
|   - topic_service: MiniLM-L6-v2 CPU Embeddings, HDBSCAN, c-TF-IDF, Drift Detection    |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                               DATA & INFRASTRUCTURE LAYER                             |
|   - SQLAlchemy 2.0 Async ORM with aiosqlite / asyncpg                                 |
|   - Tables: predictions, review_aspects, review_emotions, review_embeddings,          |
|             topic_discovery_runs, discovered_topics, topic_assignments, audit_logs     |
|   - Local Model Weights Disk: models/{absa,emotion,topics}/ (local_files_only=True)    |
+---------------------------------------------------------------------------------------+
```

---

## 5. 💻 Enterprise Frontend Architecture

The frontend is constructed with React 18, TypeScript, Tailwind CSS, Lucide icons, and Recharts, verified by Vitest and React Testing Library:

- **Review Intelligence Workbench (`ReviewIntelligence.tsx`):**
  - Interactive multi-span character-offset evidence highlighter with non-destructive slicing.
  - Aspect terms color-coded by sentiment with exact character boundaries.
  - Emotion category badge and radar/signal readout.
  - Frozen-centroid topic assignment badge with similarity score and distance to centroid.
  - Real-time telemetry strip (per-capability latency breakdown, model versions).
  - Batch review processor with progress indicator and deduplication metrics.

- **Cross-Feature Intelligence Analytics (`CrossFeatureAnalytics.tsx`):**
  - **Aspect $\times$ Discovered Topic Distribution:** Stacked horizontal/vertical bar chart and matrix table showing polarities per topic.
  - **Emotion $\times$ Overall Sentiment & Aspect Categories:** Heatmap/stacked distribution revealing psychological drivers of churn.
  - **Topic Summary Table:** Aggregated volume, sentiment ratio, and dominant CX emotion.
  - **Negative Aspect Trend Tracking:** Identifies emerging hardware/software defect trajectories over time.
  - **Sample Denominators & Guardrails:** Strict `n = ...` display and `Low Sample (n < 5)` warning badges on low-volume cells.
  - **Statistical Integrity Notice:** Explicit non-causal disclaimer footer.
  - **Slide-Over Evidence Drilldown Drawer:** Instant inspection of actual source review texts backing any analytics cell, dismissible via `Escape` key or backdrop click.
