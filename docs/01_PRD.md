# Product Requirements Document (PRD)
## Production-Grade AI-Powered Sentiment Analysis Platform

---

## 1. Document Control & Governance
- **Document Version**: 1.1.0
- **Document Status**: Production Blueprint (Approved Contractual Specification)
- **Single Source of Truth Alignment**: Strictly aligned with [Phase_0_Project_Foundation.md](file:///c:/Users/adity/OneDrive/Desktop/SENTIMENT%20ANALYSIS/Phase_0_Project_Foundation.md).
- **Authors**: Principal Product Manager, Principal AI Architect, Principal Software Architect, Senior ML Engineer, Staff Backend Engineer, Senior Frontend Engineer, DevOps Architect, QA Architect.
- **Framework Mandate**: Pure TensorFlow 2.x and Keras (`tf.keras`) — Zero PyTorch / Torch / ONNX dependencies.

---

## 2. Project Vision & Strategic Alignment
The AI-Powered Sentiment Analysis Platform is an enterprise-grade SaaS intelligence engine designed to convert unstructured, high-volume customer text streams (reviews, support tickets, survey responses) into real-time actionable sentiment insights.

### 2.1 Strategic Pillars
1. **Low-Latency Inference (<300ms SLA)**: Deliver instant sentiment predictions and probability spectrums via a pre-compiled, warm TensorFlow graph engine.
2. **Explainable AI (XAI)**: Provide token-level attribution and gradient-based keyphrase highlights (`tf.GradientTape`) to ensure transparent, audit-ready AI decisions.
3. **High-Throughput Batch Processing**: Process multi-thousand row files asynchronously using streaming micro-batches (`tf.data`) without blocking real-time API routes.
4. **Architectural Purity & Zero Technical Debt**: Enforce a pure TensorFlow + FastAPI + React/Vite architecture to prevent cross-framework serialization friction and future refactoring.

---

## 3. Problem Statement & User Persona Mapping

### 3.1 Problem Statement
Modern businesses receive tens of thousands of customer text submissions daily across e-commerce reviews, support channels, and social media. Manual analysis is impossible at scale, slow, and subjective. Existing off-the-shelf APIs operate as opaque black boxes without token explainability, present steep recurring per-request fees, and suffer from latency spikes during batch ingestion.

### 3.2 Target Personas & Use Cases
| Persona | Key Responsibility | Core Platform Pain Point | Platform Solution |
| :--- | :--- | :--- | :--- |
| **Customer Success Manager** | Immediate triage of negative customer feedback | Delayed identification of high-risk churn signals | Real-time single prediction API & instant alerts |
| **Data Analyst & Product Manager** | Tracking sentiment trends across product features | Manual processing of monthly survey CSVs | Asynchronous batch uploader & Recharts analytics |
| **Senior ML Engineer** | Model monitoring, drift detection, and accuracy tracking | Opaque black-box model predictions | Token-level explainability (`tf.GradientTape`) & benchmark panel |
| **Integration Developer** | Connecting sentiment pipeline to internal CRMs | Complex, undocumented vendor REST APIs | OpenAPI-compliant FastAPI endpoints with JWT auth |

---

## 4. Core Objectives & Key Results (OKRs)

### 4.1 Objective 1: Deliver Production-Grade Low-Latency Inference
- **KR 1.1**: Maintain p95 single-prediction end-to-end latency < 300ms (p50 < 50ms) on a warm TensorFlow graph engine.
- **KR 1.2**: Support > 500 requests/second per backend container instance.

### 4.2 Objective 2: Provide Enterprise-Grade Accuracy & Explainability
- **KR 2.1**: Achieve ≥ 92.5% macro F1-score on benchmark sentiment validation datasets.
- **KR 2.2**: Generate token-level gradient attribution keyphrases for 100% of single predictions.

### 4.3 Objective 3: Maintain Asynchronous High-Throughput Batch Pipeline
- **KR 3.1**: Process CSV batch files at ≥ 1,000 text items per second on background worker threads.
- **KR 3.2**: Provide zero-downtime, non-blocking file upload and real-time SSE / progress tracking.

---

## 5. Detailed Product Feature Specifications

### 5.1 Authentication & User Governance Module
- **Why it exists**: Secures multi-tenant platform access, protects prediction routes, and enables user-scoped prediction history tracking.
- **Architecture Interaction**: Intercepts all incoming REST requests via FastAPI OAuth2 Password Bearer middleware.
- **Backend Module Owner**: `backend/app/api/v1/auth.py` & `backend/app/core/security.py`.
- **Frontend Module Consumer**: `frontend/src/context/AuthContext.tsx` & `frontend/src/components/layout/Navbar.tsx`.
- **API Endpoints**:
  - `POST /api/v1/auth/register` (User registration with bcrypt password hashing).
  - `POST /api/v1/auth/login` (Token issuance: access token + refresh token).
  - `GET /api/v1/auth/me` (Current user profile resolution).
- **Database Tables Required**: `users` (id, email, hashed_password, full_name, is_active, created_at).
- **Scope Level**: **MVP**.
- **Evolution Path**: V1 adds API Key generation for programmatic headless pipelines; V2 adds RBAC (Admin, Analyst, Viewer).

---

### 5.2 Real-Time Single Sentiment Prediction & Explainability Module
- **Why it exists**: Serves instantaneous text sentiment classification (Positive, Neutral, Negative) along with calibrated confidence scores and token-level explainability.
- **Architecture Interaction**: Text payload -> FastAPI route -> `tf_inference.py` service -> `AutoTokenizer(return_tensors="tf")` -> `TFAutoModelForSequenceClassification` -> `tf.nn.softmax` -> `tf.GradientTape` token attribution -> JSON response.
- **Backend Module Owner**: `backend/app/api/v1/predict.py` & `backend/app/services/tf_inference.py`.
- **Frontend Module Consumer**: `frontend/src/components/predict/SinglePredict.tsx` & `frontend/src/components/predict/ProbabilityBar.tsx`.
- **API Endpoints**:
  - `POST /api/v1/predict` (Single text input, max 5,000 chars -> sentiment, probabilities, latency_ms, explainability_tokens).
- **Database Tables Required**: `predictions` (id, user_id, text_content, sentiment_label, confidence_score, positive_prob, neutral_prob, negative_prob, latency_ms, explainability_json, created_at).
- **Scope Level**: **MVP**.
- **Evolution Path**: V1 adds Redis result caching based on text hash; V2 adds multi-class emotion breakdown (Joy, Sadness, Anger).

---

### 5.3 Asynchronous Batch Processing & CSV Ingestion Engine
- **Why it exists**: Enables bulk sentiment analysis on uploaded CSV and TXT files without timing out web servers or freezing user interfaces.
- **Architecture Interaction**: Frontend uploads file -> FastAPI saves file to storage -> Enqueues batch job in `batch_service.py` -> Background thread processes micro-batches (`tf.data.Dataset`, batch_size=64) using TensorFlow batched inference -> Updates job progress -> Emits downloadable CSV.
- **Backend Module Owner**: `backend/app/api/v1/batch.py` & `backend/app/services/batch_service.py`.
- **Frontend Module Consumer**: `frontend/src/components/batch/FileUploader.tsx` & `frontend/src/components/batch/BatchProgressBar.tsx`.
- **API Endpoints**:
  - `POST /api/v1/batch/upload` (Multipart file upload -> returns `job_id`).
  - `GET /api/v1/batch/status/{job_id}` (Poll job status, progress percentage, processed count).
  - `GET /api/v1/batch/download/{job_id}` (Download processed results CSV file).
- **Database Tables Required**: `batch_jobs` (id, user_id, filename, total_rows, processed_rows, status, result_file_path, created_at, completed_at).
- **Scope Level**: **MVP** (basic file upload and processing); **V1** adds progress tracking and streaming download.
- **Evolution Path**: V2 adds Celery + RabbitMQ integration for distributed multi-node worker clusters.

---

### 5.4 Historical Prediction Explorer & Export Analytics Module
- **Why it exists**: Allows analysts to search, filter, inspect past predictions, and export aggregate dataset reports.
- **Architecture Interaction**: React History component requests paginated prediction records from FastAPI, querying SQLAlchemy async ORM database tables with index-optimized filters.
- **Backend Module Owner**: `backend/app/api/v1/history.py` & `backend/app/api/v1/analytics.py`.
- **Frontend Module Consumer**: `frontend/src/components/history/HistoryTable.tsx` & `frontend/src/components/dashboard/StatCards.tsx`.
- **API Endpoints**:
  - `GET /api/v1/history` (Query params: search, sentiment_filter, start_date, end_date, page, limit).
  - `GET /api/v1/history/export` (Export filtered prediction history as CSV/JSON).
  - `GET /api/v1/analytics/overview` (Aggregate sentiment breakdown, sentiment ratio, average latency, prediction counts).
- **Database Tables Required**: `predictions` table with composite index on `(user_id, created_at, sentiment_label)`.
- **Scope Level**: **MVP**.
- **Evolution Path**: V1 adds time-series aggregation endpoints for Recharts charts; V2 adds automated weekly email reports.

---

### 5.5 Model Benchmark & Performance Monitoring Dashboard Module
- **Why it exists**: Provides ML transparency by exposing model accuracy, precision, recall, F1-score, version history, and confusion matrix data.
- **Architecture Interaction**: Reads structured metadata from `models/model_metadata.json` and database metric history, serving it to the frontend model benchmark view.
- **Backend Module Owner**: `backend/app/api/v1/model.py`.
- **Frontend Module Consumer**: `frontend/src/components/dashboard/ModelMetrics.tsx`.
- **API Endpoints**:
  - `GET /api/v1/model/info` (Returns model version, architecture, parameter count, training date).
  - `GET /api/v1/model/metrics` (Returns accuracy, macro F1, precision, recall, confusion matrix matrix values).
- **Database Tables Required**: Read-only access to `models/model_metadata.json` & `metrics` history table.
- **Scope Level**: **MVP**.
- **Evolution Path**: V1 adds live drift detection alert threshold indicators.

---

### 5.6 System Health, Observability & Resource Monitoring Module
- **Why it exists**: Ensures operational reliability by monitoring API ping latency, CPU usage, RAM consumption, and simulated GPU VRAM allocation.
- **Architecture Interaction**: Backend collects system metrics via `psutil` and TensorFlow memory interfaces, returning real-time health JSON to the frontend health dashboard.
- **Backend Module Owner**: `backend/app/api/v1/health.py`.
- **Frontend Module Consumer**: `frontend/src/components/health/SystemHealthGauges.tsx`.
- **API Endpoints**:
  - `GET /api/v1/health` (Overall API health status `OK` / `DEGRADED`).
  - `GET /api/v1/health/system` (CPU utilization %, RAM usage MB/%, model warm-up status, uptime seconds).
- **Database Tables Required**: None (In-memory system checks).
- **Scope Level**: **MVP**.
- **Evolution Path**: V1 adds Prometheus metrics exporter (`/metrics`).

---

### 5.7 Configuration & Settings Governance Module
- **Why it exists**: Grants users and admins control over default prediction thresholds, dark/light theme preferences, and API keys.
- **Architecture Interaction**: Updates user preferences in database and local browser state.
- **Backend Module Owner**: `backend/app/api/v1/settings.py`.
- **Frontend Module Consumer**: `frontend/src/components/settings/SettingsPanel.tsx`.
- **API Endpoints**:
  - `GET /api/v1/settings` (Fetch current user settings).
  - `PUT /api/v1/settings` (Update confidence threshold, default batch size).
- **Database Tables Required**: `user_settings` (user_id, confidence_threshold, default_export_format, theme).
- **Scope Level**: **MVP**.
- **Evolution Path**: V1 adds API Key management interface (Generate, Revoke API Keys).

---

## 6. End-to-End User Journeys & Flow Mappings

### 6.1 Single Prediction Flow
```
[ User enters review text in UI ]
               │
               ▼
[ Client-side validation: len > 0 & len <= 5000 ]
               │
               ▼
[ POST /api/v1/predict (JWT Header) ]
               │
               ▼
[ FastAPI Auth Middleware verifies JWT ]
               │
               ▼
[ tf_inference.py: Tokenize with AutoTokenizer(return_tensors="tf") ]
               │
               ▼
[ TFAutoModelForSequenceClassification forward pass ]
               │
               ▼
[ tf.nn.softmax computes probability distribution ]
               │
               ▼
[ tf.GradientTape computes token gradient attribution ]
               │
               ▼
[ Save prediction record to DB asynchronously ]
               │
               ▼
[ Return 200 OK: Label, Confidence, Probabilities, Latency, Explainability ]
               │
               ▼
[ UI renders animated probability bar, latency badge & highlighted keyphrases ]
```

### 6.2 Batch CSV Ingestion Flow
```
[ User drops CSV file onto Uploader component ]
               │
               ▼
[ POST /api/v1/batch/upload ]
               │
               ▼
[ FastAPI stores file to upload directory & creates DB batch_job record ]
               │
               ▼
[ Background task spawns batch_service.py worker ]
               │
               ▼
[ tf.data.Dataset streams CSV in micro-batches (batch_size=64) ]
               │
               ▼
[ Batched TF inference executes & appends results to CSV ]
               │
               ▼
[ Job status updated to COMPLETED in DB ]
               │
               ▼
[ UI polling / SSE updates progress bar to 100% & shows Download CSV button ]
```

---

## 7. Functional & Non-Functional Requirements Matrix

### 7.1 Functional Requirements (FR)
- **FR-1**: User registration, login, and JWT access token validation.
- **FR-2**: Real-time sentiment prediction returning Positive, Neutral, Negative labels.
- **FR-3**: Calibrated sentiment probability scores summing to 1.0 (100%).
- **FR-4**: Token-level explainability displaying key phrases influencing prediction.
- **FR-5**: CSV and TXT multi-row batch file uploading and processing.
- **FR-6**: Real-time progress bar updating batch execution status.
- **FR-7**: Exporting batch results and historical prediction tables as downloadable CSV files.
- **FR-8**: Historical prediction searching, sentiment filtering, and date range query filters.
- **FR-9**: Model performance display showing Accuracy, Precision, Recall, F1 Score, and version metadata.
- **FR-10**: System operational monitoring displaying CPU, RAM, API latency, and status health.

### 7.2 Non-Functional Requirements (NFR)
- **NFR-1 Latency**: End-to-end single prediction response time < 300ms (p95) on CPU/GPU targets.
- **NFR-2 Availability**: 99.9% API availability for all core REST endpoints.
- **NFR-3 Scalability**: Asynchronous non-blocking architecture supporting > 500 concurrent req/sec per node.
- **NFR-4 Security**: Passwords hashed with bcrypt; JWT tokens signed using HS256/RS256; CORS strictly bounded.
- **NFR-5 Maintainability**: 100% adherence to Clean Architecture and PEP8 style guidelines; zero code duplication.
- **NFR-6 Framework Purity**: 100% native TensorFlow 2.x and Keras (`tf.keras`); zero PyTorch dependencies.

---

## 8. Pure TensorFlow & Keras AI/ML Pipeline Specification

> [!IMPORTANT]
> **Strict ML Framework Constraint:**
> All machine learning operations are restricted to **TensorFlow 2.x** and **Keras (`tf.keras`)**. PyTorch (`torch`), TorchVision, ONNX, and `return_tensors="pt"` are strictly forbidden.

### 8.1 ML Pipeline Stages
1. **Dataset Ingestion**: Loaded via `tf.data.Dataset.from_tensor_slices()` for high-performance memory streaming.
2. **Preprocessing**: Lowercasing, HTML tag stripping, and whitespace normalization executed via TensorFlow string operations (`tf.strings`).
3. **Tokenization**: HuggingFace `AutoTokenizer` initialized for TensorFlow compatibility, invoked strictly with `return_tensors="tf"`.
4. **Model Architecture**: `TFAutoModelForSequenceClassification` or custom `tf.keras.Model` subclass with a HuggingFace Transformer backbone (`distilbert-base-uncased-finetuned-sst-2-english`) and a dense Keras classification head (`tf.keras.layers.Dense(3, activation='softmax')`).
5. **Model Compilation & Graph Optimization**: Compiled with `tf.keras.optimizers.Adam(learning_rate=2e-5)` and `tf.keras.losses.SparseCategoricalCrossentropy()`. Inference methods decorated with `@tf.function` for XLA compilation.
6. **Model Artifact Export**: Saved natively to disk using the Keras native format (`models/tf_sentiment_model.keras`) or TensorFlow `SavedModel` folder structure.
7. **Explainability Engine**: Native `tf.GradientTape()` calculating the gradient of the predicted class logit with respect to input token embeddings, generating exact attribution scores per token.

---

## 9. Data Architecture & Database Schema Specification

The backend database utilizes **SQLAlchemy 2.0 Async Engine** with PostgreSQL in production and SQLite for local development.

```
┌───────────────────────────────────────┐
│                 users                 │
├───────────────────────────────────────┤
│ id (PK, UUID/INT)                     │
│ email (VARCHAR(255), UNIQUE, INDEX)   │
│ hashed_password (VARCHAR(255))        │
│ full_name (VARCHAR(255))              │
│ is_active (BOOLEAN, DEFAULT TRUE)     │
│ created_at (TIMESTAMP)                │
└──────────────────┬────────────────────┘
                   │ 1:N
                   ▼
┌───────────────────────────────────────┐
│              predictions              │
├───────────────────────────────────────┤
│ id (PK, UUID/INT)                     │
│ user_id (FK -> users.id, INDEX)       │
│ text_content (TEXT)                   │
│ sentiment_label (VARCHAR(20), INDEX)  │
│ confidence_score (FLOAT)              │
│ positive_prob (FLOAT)                 │
│ neutral_prob (FLOAT)                  │
│ negative_prob (FLOAT)                 │
│ latency_ms (FLOAT)                    │
│ explainability_json (JSON/TEXT)       │
│ created_at (TIMESTAMP, INDEX)         │
└───────────────────────────────────────┘

┌───────────────────────────────────────┐
│              batch_jobs               │
├───────────────────────────────────────┤
│ id (PK, UUID/INT)                     │
│ user_id (FK -> users.id, INDEX)       │
│ filename (VARCHAR(255))               │
│ total_rows (INT)                      │
│ processed_rows (INT)                  │
│ status (VARCHAR(50), INDEX)           │
│ result_file_path (VARCHAR(500))       │
│ created_at (TIMESTAMP)                │
│ completed_at (TIMESTAMP, NULLABLE)    │
└───────────────────────────────────────┘

┌───────────────────────────────────────┐
│             user_settings             │
├───────────────────────────────────────┤
│ id (PK, UUID/INT)                     │
│ user_id (FK -> users.id, UNIQUE)      │
│ confidence_threshold (FLOAT)          │
│ default_export_format (VARCHAR(10))   │
│ theme (VARCHAR(20))                   │
└───────────────────────────────────────┘
```

---

## 10. Dependency Safety & Approved Third-Party Ecosystem

Every recommended third-party dependency has been vetted for stability, active maintenance, production adoption, and zero overlap.

### 10.1 Backend Python Dependencies (`backend/requirements.txt`)
- `tensorflow>=2.14.0`: Core machine learning engine and Keras neural network APIs.
- `tf-keras>=2.14.0`: Standalone Keras 3 / TensorFlow integration library.
- `transformers>=4.35.0`: HuggingFace Transformer architecture (used strictly with `return_tensors="tf"`).
- `fastapi>=0.104.0`: High-performance asynchronous REST web framework.
- `uvicorn[standard]>=0.24.0`: Production ASGI server.
- `pydantic>=2.5.0`: Fast schema validation and configuration management.
- `sqlalchemy>=2.0.23`: Async ORM for database abstraction.
- `asyncpg>=0.29.0`: PostgreSQL async database driver.
- `python-jose[cryptography]>=3.3.0`: JWT token generation and verification.
- `passlib[bcrypt]>=1.7.4`: Password hashing with bcrypt.
- `psutil>=5.9.6`: Hardware and system resource monitoring.
- `pytest>=7.4.3` & `pytest-asyncio>=0.21.1`: Automated testing suite.

### 10.2 Frontend Node Dependencies (`frontend/package.json`)
- `react` & `react-dom` (^18.2.0): Core UI framework.
- `typescript` (^5.2.2): Type safety across components and services.
- `vite` (^5.0.0): Fast frontend build tool and dev server.
- `tailwindcss` (^3.3.5): Utility-first CSS styling framework.
- `lucide-react` (^0.292.0): Modern icon library.
- `recharts` (^2.9.3): Interactive chart visualizations for sentiment trends and metrics.
- `axios` (^1.6.2): HTTP client with JWT request/response interceptors.

---

## 11. MVP vs. V1 vs. V2 vs. Enterprise Release Matrix

| Feature / Capability | MVP (Phase 1) | Version 1.0 (Production) | Version 2.0 | Enterprise Edition |
| :--- | :---: | :---: | :---: | :---: |
| **FastAPI REST Gateway & JWT Auth** | ✅ | ✅ | ✅ | ✅ |
| **Pure TensorFlow/Keras Inference** | ✅ | ✅ | ✅ | ✅ |
| **Single Prediction & Latency Badge** | ✅ | ✅ | ✅ | ✅ |
| **Batch CSV Upload & Processing** | ✅ (Basic) | ✅ (Streaming) | ✅ | ✅ |
| **Recharts Analytics & Stat Cards** | ✅ | ✅ | ✅ | ✅ |
| **Token Explainability (`tf.GradientTape`)**| ❌ | ✅ | ✅ | ✅ |
| **Redis Result Caching** | ❌ | ✅ | ✅ | ✅ |
| **API Key Management** | ❌ | ✅ | ✅ | ✅ |
| **Celery Multi-Node Worker Queue** | ❌ | ❌ | ✅ | ✅ |
| **Multi-Lingual Sentiment Models** | ❌ | ❌ | ✅ | ✅ |
| **Active Learning Retraining Pipeline** | ❌ | ❌ | ❌ | ✅ |

---

## 12. Quality Assurance & Testing Protocol (Definition of Done)

A feature is officially declared **Complete & Production-Ready** only when it satisfies the 10-point **Definition of Done**:
1. **Implementation**: Clean Architecture code written with zero placeholders (`pass`, `TODO`).
2. **Unit Tests**: `pytest` unit tests verifying core logic and TensorFlow tensor outputs (`return_tensors="tf"`).
3. **Integration Tests**: End-to-end route tests passing cleanly in `pytest`.
4. **Documentation**: Google-style docstrings and updated module documentation.
5. **Logging**: Structured JSON logs with correlation IDs for operational tracing.
6. **Error Handling**: Graceful exception boundaries returning standard OpenAPI error schemas.
7. **API Documentation**: OpenAPI / Swagger schema annotations verified.
8. **Performance Validation**: Latency < 300ms SLA verified on single prediction endpoints.
9. **Security Review**: JWT authorization verified; zero hardcoded secrets.
10. **Production Readiness**: Docker container build verified with clean exit status.

---

## 13. Production Readiness & Risk Mitigation Matrix

| Risk | Impact | Root Cause | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **TensorFlow Cold-Start Latency** | High | First execution compiles XLA `@tf.function` graph | Execute warm-up dummy prediction during FastAPI `lifespan` startup event. |
| **Memory Exhaustion on Large CSV Uploads** | High | Loading whole CSV file into RAM | Stream CSV rows using `tf.data.Dataset` in micro-batches of size 64. |
| **Cross-Framework Package Contamination** | Critical | Accidental import of PyTorch by developer | Enforce automated `flake8` / AST linter rule blocking `import torch`. |
| **Database Connection Pool Bottleneck** | Medium | High volume of concurrent async writes | Configure SQLAlchemy connection pool size (`pool_size=20, max_overflow=10`). |
| **Dual JWT / API Key Auth Failures** | Medium | Improper middleware token extraction | Unified auth dependency resolving both Bearer JWT and `X-API-Key` headers. |

---

## 14. Requirements Traceability Matrix (RTM)

| Req ID | Business Objective | User Story | Feature | Backend Module | Frontend Module | Database Tables | API Endpoint | Testing Requirement | Acceptance Criteria |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RTM-01** | Secure Multi-Tenancy | As a user, I want to securely log in so my prediction history remains private. | User Auth & JWT | `app/api/v1/auth.py` | `AuthContext.tsx` | `users` | `POST /api/v1/auth/login` | Pytest Auth Unit Test | Valid JWT token returned on valid credentials |
| **RTM-02** | Real-Time Sentiment Analysis | As a user, I want instant sentiment classification of customer reviews. | Single Prediction | `app/api/v1/predict.py` | `SinglePredict.tsx` | `predictions` | `POST /api/v1/predict` | Pytest Model Latency Test | Sentiment label + probabilities returned in <300ms |
| **RTM-03** | AI Explainability | As an analyst, I want to see which words drove the sentiment prediction. | Token Explainability | `app/services/tf_inference.py` | `ProbabilityBar.tsx` | `predictions` | `POST /api/v1/predict` | Pytest Tensor Gradient Test | Highlighted keyphrases returned in explainability JSON |
| **RTM-04** | Bulk CSV Processing | As a product manager, I want to upload a CSV of 1,000 reviews for batch analysis. | Batch Ingestion | `app/api/v1/batch.py` | `FileUploader.tsx` | `batch_jobs` | `POST /api/v1/batch/upload` | Pytest File Upload Test | Processed CSV downloadable with progress tracking |
| **RTM-05** | Historical Analytics | As a manager, I want to search past predictions and filter by sentiment label. | Prediction History | `app/api/v1/history.py` | `HistoryTable.tsx` | `predictions` | `GET /api/v1/history` | Pytest DB Query Filter Test | Paginated search & sentiment filter results returned |
| **RTM-06** | Model Observability | As an ML engineer, I want to view model accuracy, precision, recall, and F1 score. | Model Benchmark | `app/api/v1/model.py` | `ModelMetrics.tsx` | `metrics` | `GET /api/v1/model/metrics` | Pytest Metadata JSON Test | Valid metadata returned matching benchmark scores |
| **RTM-07** | Health Monitoring | As a DevOps engineer, I want to monitor API latency, CPU, and RAM utilization. | System Health | `app/api/v1/health.py` | `SystemHealthGauges.tsx` | N/A (In-memory) | `GET /api/v1/health/system` | Pytest Health Ping Test | Returns CPU %, RAM MB %, and API status OK |

---

## 15. Feature Dependency Matrix

```
[ Authentication Module ] (Mandatory Base)
           │
           ▼
[ Dashboard & Navigation ] (Mandatory)
           │
           ├───────────────────────────────┐
           ▼                               ▼
[ Single Prediction Engine ]     [ Batch Ingestion Engine ]
(Mandatory Core Feature)         (Mandatory Bulk Feature)
           │                               │
           └───────────────┬───────────────┘
                           │
                           ▼
             [ Historical Prediction Explorer ] (Mandatory Data Layer)
                           │
                           ├───────────────────────────────┐
                           ▼                               ▼
             [ Analytics & Recharts Overview ]     [ CSV/JSON Export Service ]
             (Mandatory Visualizer)                 (Optional / Secondary Feature)
```

- **Mandatory Dependencies**: Auth -> Dashboard -> Single Prediction & Batch Engine -> History Explorer -> Analytics.
- **Optional / Auxiliary Dependencies**: Result Export Service, Dark/Light Theme Switcher, API Key Generation.

---

## 16. Data Flow Overview

### 16.1 Single Prediction Data Flow
```
User Text Input ──► React Component ──► Axios Client (JWT Header)
                                                │
                                                ▼
FastAPI Gateway ◄── JSON Response ◄── Database (SQLAlchemy)
      │                                         ▲
      ▼                                         │
tf_inference.py ──► AutoTokenizer(return_tensors="tf") ──► TF Keras Model
```

### 16.2 Batch Prediction Data Flow
```
CSV File Upload ──► React Drag & Drop ──► POST /api/v1/batch/upload
                                                 │
                                                 ▼
FastAPI Storage ──► Enqueue job in batch_service.py ──► tf.data.Dataset
                                                             │
                                                             ▼
Downloadable CSV ◄── Update Status in DB ◄── Batched TF Inference (size=64)
```

### 16.3 Authentication Data Flow
```
Credentials (Email/Pass) ──► React Login Form ──► POST /api/v1/auth/login
                                                         │
                                                         ▼
JWT Access Token ◄── Password Bcrypt Verify ◄── Query DB `users` Table
      │
      ▼
Attached to subsequent REST calls via `Authorization: Bearer <token>`
```

---

## 17. Feature Ownership Matrix

| Feature Module | Backend Owner | Frontend Owner | AI Owner | DB Owner | DevOps Owner | QA Owner | Doc Owner |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Authentication & Auth Context** | Staff Backend | Senior Frontend | ML Engineer | Staff Backend | DevOps Arch | QA Arch | Tech Writer |
| **Single Prediction Engine** | Staff Backend | Senior Frontend | Principal AI | Staff Backend | DevOps Arch | QA Arch | Principal AI |
| **Token Explainability (`tf.GradientTape`)**| Staff Backend | Senior Frontend | Senior ML | N/A | DevOps Arch | QA Arch | Senior ML |
| **Batch CSV Ingestion Engine** | Staff Backend | Senior Frontend | Senior ML | Staff Backend | DevOps Arch | QA Arch | Staff Backend |
| **History & Export Module** | Staff Backend | Senior Frontend | N/A | Staff Backend | DevOps Arch | QA Arch | Tech Writer |
| **Model Metrics Dashboard** | Staff Backend | Senior Frontend | Principal AI | Staff Backend | DevOps Arch | QA Arch | Principal AI |
| **System Health & Monitoring** | Staff Backend | Senior Frontend | N/A | N/A | DevOps Arch | QA Arch | DevOps Arch |

---

## 18. Version Compatibility Matrix

| Technology Component | Minimum Compatible Version | Production Standard Version | Maximum Tested Version | Notes / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **Python** | 3.10.0 | 3.10.12 | 3.11.x | Mandatory for async SQLAlchemy & Pydantic v2 |
| **TensorFlow** | 2.14.0 | 2.15.0 | 2.15.x | Pure `tf.keras` ecosystem (Zero PyTorch) |
| **tf-keras** | 2.14.0 | 2.15.0 | 2.15.x | Native Keras 3 integration |
| **FastAPI** | 0.104.0 | 0.109.0 | 0.110.x | Async web gateway |
| **Transformers** | 4.35.0 | 4.36.2 | 4.37.x | Invoked with `return_tensors="tf"` ONLY |
| **React** | 18.2.0 | 18.2.0 | 18.3.x | Functional components with Hooks |
| **TypeScript** | 5.0.0 | 5.2.2 | 5.3.x | Strict mode enabled |
| **Vite** | 5.0.0 | 5.0.12 | 5.1.x | Fast build tool & dev proxy |
| **TailwindCSS** | 3.3.0 | 3.3.5 | 3.4.x | Utility-first styling |
| **PostgreSQL** | 14.0 | 15.4 | 16.x | Async driver (`asyncpg`) |
| **Docker** | 24.0.0 | 24.0.7 | 25.x | Multi-stage production container build |

---

## 19. Configuration Management Specification

### 19.1 Environment Variables (`backend/.env`)
- `PROJECT_NAME`: `AI Sentiment Analysis Platform`
- `API_V1_STR`: `/api/v1`
- `SECRET_KEY`: Cryptographically secure JWT secret key string.
- `ACCESS_TOKEN_EXPIRE_MINUTES`: `60`
- `DATABASE_URL`: `postgresql+asyncpg://user:pass@localhost:5432/sentiment_db` (or `sqlite+aiosqlite:///./sentiment.db` for local dev).
- `REDIS_URL`: `redis://localhost:6379/0`
- `TF_MODEL_PATH`: `models/tf_sentiment_model.keras`
- `MODEL_METADATA_PATH`: `models/model_metadata.json`
- `MAX_SINGLE_PREDICT_LEN`: `5000`
- `BATCH_PROCESSING_SIZE`: `64`

### 19.2 Logging Configuration
Structured JSON logging configured via Python `logging` dictConfig, injecting `correlation_id`, `timestamp`, `log_level`, `module`, and `message`.

---

## 20. Non-Negotiable Project Constraints
> [!CAUTION]
> **Strict Engineering Guardrails:**
> 1. **No Framework Switching**: Zero PyTorch (`torch`), TorchVision, TorchAudio, ONNX, or `return_tensors="pt"`. TensorFlow 2.x and `tf.keras` are non-negotiable.
> 2. **No Duplicated APIs or Routes**: All API routes must reside strictly under `backend/app/api/v1/`.
> 3. **No Duplicate ML Models or Tokenizers**: Model loading must be encapsulated strictly within singleton instances in `tf_inference.py`.
> 4. **No Duplicate Preprocessing Pipelines**: Text normalization rules must be defined once in `tf_inference.py` and reused across single and batch pipelines.
> 5. **No Business Logic inside API Routes**: FastAPI router functions must be thin delegates passing requests to Service classes.
> 6. **No Direct Database Access from Frontend**: React must interact exclusively with FastAPI REST endpoints.
> 7. **No Hardcoded Paths or Credentials**: All paths, ports, and secret keys must be resolved dynamically via environment variables (`config.py`).

---

## 21. Architectural Decision Records (ADR)

### ADR-001: Pure TensorFlow 2.x & Keras over PyTorch
- **Status**: Approved / Non-Negotiable
- **Context**: The platform requires deterministic graph compilation (`@tf.function`), enterprise model serialization (`SavedModel` / `.keras`), and high-throughput deployment consistency.
- **Decision**: Standardize 100% of ML operations on TensorFlow 2.x and `tf.keras`.
- **Consequences**: Eliminates PyTorch memory overhead, CUDA conflicts, and tensor format conversion bugs (`return_tensors="tf"` strictly enforced).

### ADR-002: FastAPI over Flask / Django
- **Status**: Approved
- **Context**: High-concurrency async REST operations with automated OpenAPI documentation and Pydantic schema validation.
- **Decision**: Select FastAPI as the backend web gateway.
- **Consequences**: Enables non-blocking async IO for database and background task handling with p95 < 300ms SLA.

### ADR-003: React + TypeScript + Vite over Angular / Vue
- **Status**: Approved
- **Context**: Modern dark-mode SaaS UI requiring fast hot module replacement, strong type safety, and rich visualization library support (Recharts).
- **Decision**: Select React 18 + TypeScript + Vite.
- **Consequences**: High frontend developer velocity, small initial bundle size (<250 KB gzipped), and component modularity.

### ADR-004: PostgreSQL over SQLite for Production
- **Status**: Approved
- **Context**: Enterprise production requires robust concurrency, connection pooling, and JSON field support.
- **Decision**: Use PostgreSQL for production and SQLite as zero-config local development fallback via SQLAlchemy 2.0.
- **Consequences**: Smooth transition from local development to production database deployment.

### ADR-005: Clean Architecture & Repository Pattern
- **Status**: Approved
- **Context**: Long-term code maintainability, testability, and decoupling business rules from external frameworks.
- **Decision**: Implement Clean Architecture (Presentation -> Application -> Domain -> Infrastructure).
- **Consequences**: Highly testable components; API routes contain zero SQL queries or ML tensor calls.

### ADR-006: JWT OAuth2 Bearer Authentication
- **Status**: Approved
- **Context**: Stateless, scalable API authentication for web clients and third-party integrations.
- **Decision**: Implement OAuth2 Password Bearer flow with JWT access tokens signed via HS256.
- **Consequences**: Scalable stateless session verification without database session lookup overhead on every API call.

---

## 22. Engineering Review Checklist
Before any future phase is accepted for release, the engineering team must verify:
- [x] All requirements match `Phase_0_Project_Foundation.md` without contradiction.
- [x] All requirements match `01_PRD.md` specifications.
- [x] Zero conflicting tech stack recommendations (Zero PyTorch).
- [x] Zero duplicated API endpoints or ML preprocessing routines.
- [x] Zero unnecessary third-party dependencies introduced.
- [x] Zero hidden technical debt or `TODO` placeholder comments.
- [x] Production readiness verified via complete `pytest` suites and Docker builds.

---

## 23. Phase Exit Criteria
> [!IMPORTANT]
> **Strict Phase Locking Rule:**
> No future phase (TRD, UI/UX Design, Implementation, Code Delivery) may modify the **Technology Stack**, **Folder Structure**, **Architecture**, or **Engineering Principles** defined herein without explicitly updating both [Phase_0_Project_Foundation.md](file:///c:/Users/adity/OneDrive/Desktop/SENTIMENT%20ANALYSIS/Phase_0_Project_Foundation.md) and [01_PRD.md](file:///c:/Users/adity/OneDrive/Desktop/SENTIMENT%20ANALYSIS/docs/01_PRD.md) and obtaining formal architecture approval.

---
*End of Product Requirements Document (01_PRD.md).*
