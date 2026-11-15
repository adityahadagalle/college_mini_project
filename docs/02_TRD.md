# Technical Requirements Document (TRD)
## Production-Grade AI-Powered Sentiment Analysis Platform

---

## 1. Document Control & Governance
- **Document Version**: 1.0.0
- **Status**: Production Engineering Specification (Approved Architectural Blueprint)
- **Immutable Source Documents**: [Phase_0_Project_Foundation.md](file:///c:/Users/adity/OneDrive/Desktop/SENTIMENT%20ANALYSIS/Phase_0_Project_Foundation.md) & [01_PRD.md](file:///c:/Users/adity/OneDrive/Desktop/SENTIMENT%20ANALYSIS/docs/01_PRD.md).
- **Authors**: Principal Software Architect, Senior AI Engineer, Staff Backend Engineer, Senior Frontend Engineer, DevOps Architect, QA Architect.
- **Framework Constraint**: Pure TensorFlow 2.x & Keras (`tf.keras`) — Zero PyTorch (`torch`), TorchVision, ONNX, or mixed-framework dependencies.

### 1.1 Revision History
| Version | Date | Author | Description of Changes |
| :--- | :--- | :--- | :--- |
| **1.0.0** | 2026-08-01 | Engineering Architecture Board | Initial Production Release of Technical Specification |

---

## 2. System Architecture

### 2.1 High-Level Architecture
The system implements a stateless, decoupled Clean Architecture comprising a modern React SaaS frontend, an asynchronous FastAPI REST gateway, a pure TensorFlow 2.x / Keras inference engine, a Redis caching tier, and a PostgreSQL database.

```
[ Browser / Client (React 18 + TS + Tailwind) ]
                     │
                     │ HTTPS / REST (JWT Auth)
                     ▼
      [ NGINX Reverse Proxy / TLS Termination ]
                     │
                     ▼
       [ FastAPI Application Gateway (Port 8000) ]
   ┌─────────────────┼─────────────────┐
   │ Auth Middleware │ Rate Limiter    │ Pydantic Validation │
   └─────────────────┬─────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [ Service Layer ]     [ Batch Ingestion Worker ]
         │                       │
         ▼                       ▼
  [ TF Inference Engine ] [ tf.data Micro-Batches ]
  (tf.keras Model)               │
         │                       │
         ├───────────────────────┘
         ▼
  [ Infrastructure Layer ]
   ├──► Redis (Cache & Session)
   └──► PostgreSQL (SQLAlchemy Async ORM)
```

### 2.2 Container Architecture (`docker-compose.yml`)
- `web`: NGINX reverse proxy (Port 80/443).
- `frontend`: Node/Vite static build container served via NGINX.
- `backend`: FastAPI Python 3.10 ASGI container (`uvicorn app.main:app --workers 4`).
- `db`: PostgreSQL 15 container storing users, predictions, and batch job metadata.
- `cache`: Redis 7 container holding prediction caches and rate limit counters.

### 2.3 Component & Communication Sequence
```
Client                FastAPI             tf_inference.py        TF Keras Model         PostgreSQL
  │                      │                       │                      │                    │
  ├─ POST /api/v1/predict ──►                    │                      │                    │
  │  (JWT + Text Payload)│                       │                      │                    │
  │                      ├─ Tokenize Text ──────►│                      │                    │
  │                      │  (return_tensors="tf")│                      │                    │
  │                      │                       ├─ forward(tensors) ──►│                    │
  │                      │                       │◄─ logits (tf.Tensor) ┤                    │
  │                      │                       ├─ tf.nn.softmax ──────┤                    │
  │                      │                       ├─ tf.GradientTape ────┤                    │
  │                      │◄─ JSON Result ────────┤                      │                    │
  │                      ├─ Save Async DB Record ───────────────────────────────────────────►│
  │◄─ 200 OK Response ───┤                       │                      │                    │
```

---

## 3. Backend Architecture

### 3.1 Directory Structure & Layer Responsibilities
```
backend/
├── app/
│   ├── api/v1/                 # API Layer: Router definitions & endpoint contracts
│   │   ├── auth.py             # Authentication endpoints (/login, /register, /me)
│   │   ├── predict.py          # Single prediction REST routes
│   │   ├── batch.py            # CSV file batch upload & status polling
│   │   ├── history.py          # Historical prediction query & export
│   │   ├── analytics.py        # Trend stats & aggregated metrics
│   │   ├── health.py           # System resource & API health checks
│   │   ├── model.py            # Model metrics metadata endpoints
│   │   └── settings.py         # User configuration preferences
│   ├── core/                   # Infrastructure Core: Config, Security, DB Initialization
│   │   ├── config.py           # Pydantic BaseSettings environment manager
│   │   ├── security.py         # JWT creation/verification & bcrypt hashing
│   │   └── database.py        # Async SQLAlchemy engine & session factory
│   ├── models/                 # Domain / Data Layer: SQLAlchemy ORM Entities
│   │   └── database.py         # User, Prediction, BatchJob, UserSettings models
│   ├── schemas/                # Data Transfer Objects: Pydantic v2 validation models
│   │   └── sentiment.py        # Request / Response JSON schemas
│   ├── services/               # Business Logic / Service Layer
│   │   ├── tf_inference.py     # Pure TensorFlow inference & explainability engine
│   │   ├── batch_service.py    # Async streaming batch file processor
│   │   └── analytics_service.py# SQL aggregation & metrics service
│   └── main.py                 # Application Gateway entrypoint & Lifespan setup
├── requirements.txt
└── Dockerfile
```

### 3.2 Clean Architecture Design Patterns
- **Dependency Injection**: FastAPI `Depends()` injects database sessions (`get_db`), current user context (`get_current_user`), and inference instances without tight coupling.
- **Repository Pattern**: Database operations isolated within service helper classes, preventing raw SQL queries inside API route controllers.
- **Service Layer Pattern**: Business rules and TensorFlow inference wrapped in service components (`tf_inference.py`), ensuring API routers remain lightweight request delegators.

---

## 4. Frontend Architecture

### 4.1 React Directory Structure
```
frontend/
├── src/
│   ├── components/
│   │   ├── layout/             # Sidebar, Navbar, Header, ThemeToggle
│   │   ├── dashboard/          # StatCards, RechartsOverview, RecentFeed
│   │   ├── predict/            # SinglePredict, ProbabilityBar, ExplainabilityTags
│   │   ├── batch/              # FileUploader, BatchProgressBar, ResultsTable
│   │   ├── history/            # HistoryTable, FilterBar, ExportModal
│   │   ├── health/             # SystemHealthGauges, ServiceBadges
│   │   └── settings/           # SettingsForm, APIKeyManager
│   ├── context/                # React Contexts: AuthContext, ThemeContext
│   ├── services/               # Axios API client & endpoints mapping
│   │   └── api.ts
│   ├── types/                  # TypeScript Interfaces matching Pydantic schemas
│   │   └── sentiment.ts
│   ├── App.tsx                 # View Routing & Router Provider
│   ├── index.css               # Tailwind CSS & Glassmorphism utility rules
│   └── main.tsx                # Application Entrypoint
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

### 4.2 Routing & Theme Design
- **Routing**: React Router DOM with `ProtectedRoute` wrapper checking `AuthContext.isAuthenticated`.
- **Views**: Landing Page (`/`), Login/Register (`/login`), Dashboard (`/dashboard`), Single Prediction (`/predict`), Batch Processing (`/batch`), History (`/history`), Model Benchmark (`/model`), System Health (`/health`), Settings (`/settings`).
- **Theme Palette**: Modern AI SaaS dark mode theme: `#0F172A` background, `#1E293B` card surface, `#334155` borders, `#3B82F6` primary blue accent, glassmorphism overlays (`backdrop-blur-md bg-opacity-60`).

---

## 5. AI/ML Architecture (Pure TensorFlow + Keras)

> [!IMPORTANT]
> **Strict ML Framework Mandate:**
> All AI/ML components must use **TensorFlow 2.x** and **Keras (`tf.keras`)** natively. Tokenization must strictly use `return_tensors="tf"`. Zero PyTorch (`torch`) imports permitted.

### 5.1 TensorFlow Model Lifecycle & Preprocessing
```
Text String Payload ──► Normalization (tf.strings lowercasing & regex cleanup)
                               │
                               ▼
        AutoTokenizer(return_tensors="tf") ──► tf.Tensor(shape=(1, seq_len))
                               │
                               ▼
TFAutoModelForSequenceClassification ──► tf.Tensor(logits, shape=(1, 3))
                               │
                               ▼
            tf.nn.softmax(logits) ──► Probabilities [Pos, Neu, Neg]
                               │
                               ▼
 tf.GradientTape() Attribution ──► Token-level Gradient Scores
```

### 5.2 Thread Safety & Warm-Up Execution
- **Graph Compilation**: Inference routine wrapped in `@tf.function(experimental_relax_shapes=True)` for XLA (Accelerated Linear Algebra) compilation.
- **Warm-Up Cycle**: During FastAPI startup (`lifespan`), an initial dummy tensor pass (`model(tf.constant([["warmup text"]]))`) executes to pre-compile execution graphs and eliminate first-request cold-start latency.
- **Thread Safety**: Model loading executed as a thread-safe Singleton pattern in `tf_inference.py`.

---

## 6. Database Design

### 6.1 Entity-Relationship (ER) Schema
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    text_content TEXT NOT NULL,
    sentiment_label VARCHAR(20) NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    positive_prob DOUBLE PRECISION NOT NULL,
    neutral_prob DOUBLE PRECISION NOT NULL,
    negative_prob DOUBLE PRECISION NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    explainability_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE batch_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    total_rows INT NOT NULL DEFAULT 0,
    processed_rows INT NOT NULL DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    result_file_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE user_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    confidence_threshold DOUBLE PRECISION DEFAULT 0.50,
    default_export_format VARCHAR(10) DEFAULT 'csv',
    theme VARCHAR(20) DEFAULT 'dark'
);

-- Performance Indexes
CREATE INDEX idx_predictions_user_created ON predictions(user_id, created_at DESC);
CREATE INDEX idx_predictions_label ON predictions(sentiment_label);
CREATE INDEX idx_batch_jobs_user_status ON batch_jobs(user_id, status);
```

### 6.2 Migration Strategy
- Managed via **Alembic** (`alembic revision --autogenerate -m "Initial schema"` and `alembic upgrade head`).

---

## 7. API Specification

### 7.1 Single Prediction (`POST /api/v1/predict`)
- **Request Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Request Body**:
  ```json
  {
    "text": "The customer service representative resolved my issue immediately. Outstanding support!"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "id": "c9b2f6e1-4a1d-4e9b-8f3a-2b1c0d9e8f7a",
    "sentiment": "POSITIVE",
    "confidence": 0.9842,
    "probabilities": {
      "positive": 0.9842,
      "neutral": 0.0125,
      "negative": 0.0033
    },
    "latency_ms": 38.45,
    "explainability": [
      {"token": "Outstanding", "score": 0.421},
      {"token": "resolved", "score": 0.315},
      {"token": "immediately", "score": 0.182}
    ],
    "created_at": "2026-08-01T22:00:00Z"
  }
  ```
- **Validation Errors (422 Unprocessable Entity)**: Empty string, text exceeding 5,000 characters.

### 7.2 Batch Upload (`POST /api/v1/batch/upload`)
- **Request Body**: `multipart/form-data` with `file: reviews.csv`.
- **Response (202 Accepted)**:
  ```json
  {
    "job_id": "f8a7e6d5-c4b3-4a2b-1c0d-9e8f7a6b5c4d",
    "status": "PROCESSING",
    "total_rows": 1500,
    "created_at": "2026-08-01T22:00:05Z"
  }
  ```

---

## 8. Security Architecture
- **Authentication**: JWT signed via HS256 algorithm with expiration (60 mins).
- **Password Hashing**: Passlib with `bcrypt` rounds = 12.
- **CORS Protection**: Restricted to explicit domain whitelist specified in `.env` (`ALLOW_ORIGINS`).
- **SQL Injection Safeguards**: SQLAlchemy parameterized queries via ORM objects.
- **XSS Prevention**: Strict Pydantic HTML string sanitization on text inputs.
- **Rate Limiting**: Token-bucket algorithm via Redis enforcing 60 requests/minute per authenticated user.

---

## 9. Logging Architecture
Structured JSON logging output formatted via standard stdout logger:
```json
{
  "timestamp": "2026-08-01T22:00:00.123Z",
  "level": "INFO",
  "correlation_id": "a1b2c3d4-e5f6-7890-1234-56789abcdef0",
  "module": "backend.app.api.v1.predict",
  "user_id": "c9b2f6e1-4a1d-4e9b-8f3a-2b1c0d9e8f7a",
  "action": "SINGLE_PREDICTION",
  "latency_ms": 38.45,
  "status_code": 200
}
```

---

## 10. Monitoring Architecture
- **Health Check Endpoint (`GET /api/v1/health`)**: Verifies DB ping, Redis connection, and TF model memory status.
- **Resource Monitoring (`GET /api/v1/health/system`)**: Reports CPU usage percentage, RAM consumption (MB and %), active thread count, and system uptime.
- **Model Observability**: Logs prediction confidence distribution for continuous drift monitoring.

---

## 11. Caching Strategy
- **Redis Cache Key Scheme**: `prediction:<sha256_hash_of_text>`
- **TTL**: 86,400 seconds (24 hours).
- **Behavior**: Exact text match fetches cached sentiment and probabilities immediately, yielding response latency < 5ms.

---

## 12. Performance Strategy
- **Graph Optimization**: `@tf.function` XLA graph compilation.
- **Async DB I/O**: `asyncpg` non-blocking database queries.
- **Batch Processing**: Streaming file processing with `tf.data.Dataset` micro-batch size = 64.
- **Connection Pooling**: SQLAlchemy pool size = 20, max overflow = 10.

---

## 13. Testing Strategy
- **Backend Unit & Integration Tests**: Executed via `pytest tests/` checking:
  - TensorFlow model forward pass and output shapes (`return_tensors="tf"`).
  - Pydantic schema validation for valid/invalid text payloads.
  - JWT token creation, expiration, and protected route access.
- **Frontend Unit Tests**: React components verified via Vitest and React Testing Library.

---

## 14. CI/CD Architecture (GitHub Actions Workflow)
```yaml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test-and-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install Dependencies
        run: |
          pip install -r backend/requirements.txt
      - name: Run Backend Tests
        run: |
          pytest tests/
      - name: Build Docker Container
        run: |
          docker compose build
```

---

## 15. Configuration Specification
Managed strictly via `backend/app/core/config.py` using `pydantic_settings.BaseSettings`:
- `DATABASE_URL`: `postgresql+asyncpg://user:pass@localhost:5432/sentiment_db`
- `REDIS_URL`: `redis://localhost:6379/0`
- `SECRET_KEY`: Long random secret string for JWT token signatures.
- `TF_MODEL_PATH`: `models/tf_sentiment_model.keras`
- `BATCH_PROCESSING_SIZE`: `64`

---

## 16. Failure Recovery Strategy
- **Database Connection Retries**: 5 retries with exponential backoff on startup.
- **Batch Job Failures**: Failed batch jobs tagged as `FAILED` in `batch_jobs` table with explicit error message logged.
- **Graceful Shutdown**: FastAPI `lifespan` handler completes in-flight requests and closes DB connection pools cleanly.

---

## 17. Scalability Strategy
- **Stateless Backend Nodes**: FastAPI containers auto-scaled horizontally behind NGINX.
- **Database Scaling**: Read replicas for analytical history queries.
- **Async Batch Offloading**: Async processing workers isolated from main HTTP worker threads.

---

## 18. Engineering Standards
- **Python**: PEP 8 formatting enforced via `black`, `isort`, `flake8`, and `mypy` type annotations.
- **TypeScript**: Strict mode enabled (`noImplicitAny: true`).
- **Commit Messages**: Conventional commits (`feat:`, `fix:`, `docs:`, `test:`).

---

## 19. Implementation Contracts

| Component Module | Inputs | Outputs | Dependencies | Thread Safety | Target SLA / Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tf_inference.py` | Text string | Sentiment Label, Confidence, Probabilities, Explainability JSON | TensorFlow 2.x, Transformers (`return_tensors="tf"`) | Thread-Safe (Singleton Instance) | < 50ms model execution time |
| `auth.py` | Email & Password | JWT Access & Refresh Tokens | Passlib (`bcrypt`), Python-Jose | Thread-Safe | < 100ms request latency |
| `batch_service.py` | File path string | Output CSV file path | `tf.data`, SQLAlchemy Async | Thread-Safe | ≥ 1,000 items/sec throughput |

---

## 20. Definition of Ready (DoR)
A phase or task is **Ready for Coding** only when:
- [x] Technical specifications match `Phase_0_Project_Foundation.md` and `01_PRD.md`.
- [x] Database schemas and Pydantic DTOs are fully documented.
- [x] API endpoint request/response JSON contracts are specified.
- [x] Pure TensorFlow 2.x implementation approach confirmed.

---

## 21. Definition of Done (DoD)
A task is **Done** when:
- [x] Implementation satisfies Clean Architecture rules with zero `TODO` comments.
- [x] Unit and integration tests pass cleanly in `pytest`.
- [x] API documentation (OpenAPI) is validated.
- [x] Performance SLA (<300ms single inference) is verified.
- [x] Docker build completes successfully.

---

## 22. Risk Register

| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **TensorFlow Startup Latency** | High | Run warm-up inference cycle during FastAPI startup lifespan event. |
| **PyTorch Package Contamination** | Critical | Enforce automated linter check blocking `import torch`. |
| **Batch Processing Memory Exhaustion** | High | Stream CSV rows via `tf.data.Dataset` in micro-batches (batch_size=64). |

---

## 23. Architecture Compliance Checklist
Before completing any engineering implementation step, verify:
- [x] Zero PyTorch (`torch`) imports or dependencies introduced.
- [x] Pure TensorFlow 2.x & `tf.keras` used for model inference.
- [x] Tokenizers return TensorFlow tensors (`return_tensors="tf"`).
- [x] FastAPI routes delegate logic to Service layer classes.
- [x] Code passes `black`, `flake8`, and `pytest` checks cleanly.

---
*End of Technical Requirements Document (02_TRD.md).*
