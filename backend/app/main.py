import os
# Prevent OpenMP and oneDNN collisions between TensorFlow and PyTorch on Windows CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from app.api.v1 import (
    absa,
    analytics,
    auth,
    batch,
    emotion,
    health,
    history,
    intelligence,
    model,
    monitoring,
    predict,
    topics,
)
from app.api.v1 import settings as user_settings_router
from app.core.config import settings
from app.core.database import Base, engine
import app.models.database  # noqa: F401 - Register all models with Base.metadata
from app.core.exceptions import register_exception_handlers
from app.core.logging_config import CorrelationIdMiddleware, logger
from app.core.prometheus import prometheus_exporter
from app.services.tf_inference import tf_engine
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Initializing Database Tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Warming up TensorFlow Graph Inference Engine...")
    _ = tf_engine.predict_single("Warmup text sequence for TensorFlow Keras graph compilation.")
    logger.info("TensorFlow Inference Engine is warm and ready.")

    yield

    logger.info("Shutting down application...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# Register Exception Handlers
register_exception_handlers(app)

# Correlation ID Middleware
app.add_middleware(CorrelationIdMiddleware)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(
    absa.router,
    prefix=f"{settings.API_V1_STR}/absa",
    tags=["Aspect-Based Sentiment Analysis"],
)
app.include_router(
    emotion.router,
    prefix=f"{settings.API_V1_STR}/emotion",
    tags=["Emotion Detection"],
)
app.include_router(
    topics.router,
    prefix=f"{settings.API_V1_STR}/topics",
    tags=["Automatic Topic Discovery"],
)
app.include_router(
    intelligence.router,
    prefix=f"{settings.API_V1_STR}/intelligence",
    tags=["Unified Intelligence & Cross-Feature Analytics"],
)
app.include_router(
    predict.router,
    prefix=f"{settings.API_V1_STR}/predict",
    tags=["Prediction"],
)
app.include_router(
    batch.router,
    prefix=f"{settings.API_V1_STR}/batch",
    tags=["Batch Processing"],
)
app.include_router(history.router, prefix=f"{settings.API_V1_STR}/history", tags=["History"])
app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_STR}/analytics",
    tags=["Analytics"],
)
app.include_router(
    model.router,
    prefix=f"{settings.API_V1_STR}/model",
    tags=["Model Metadata"],
)
app.include_router(
    health.router,
    prefix=f"{settings.API_V1_STR}/health",
    tags=["System Health"],
)
app.include_router(
    user_settings_router.router,
    prefix=f"{settings.API_V1_STR}/settings",
    tags=["Settings"],
)
app.include_router(
    monitoring.router,
    prefix=f"{settings.API_V1_STR}",
    tags=["MLOps & Monitoring"],
)


@app.get("/metrics")  # type: ignore[misc]
def root_metrics() -> Response:
    """Expose Prometheus metrics at root level /metrics for standard scrapers."""
    content = prometheus_exporter.generate_prometheus_text()
    return Response(content=content, media_type="text/plain; version=0.0.4")


@app.get("/")  # type: ignore[misc]
async def root() -> Dict[str, Any]:
    return {
        "message": f"Welcome to {settings.PROJECT_NAME} API",
        "docs": "/docs",
        "metrics": "/metrics",
        "version": "v1.0.0",
    }
