"""MLOps Database Models.

Defines SQLAlchemy ORM entities for Model Registry, Experiment Tracking,
Drift Metrics, and Monitoring Alert Records.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.core.database import Base
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(Base):
    """Model Registry Entity for version management, active state, and rollbacks."""

    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    architecture: Mapped[str] = mapped_column(String(100), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), default="TensorFlow 2.x Keras")
    status: Mapped[str] = mapped_column(String(20), default="STAGING")  # STAGING, ACTIVE, ARCHIVED
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    artifact_path: Mapped[str] = mapped_column(String(255), nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, default=0.0)
    parameters_count: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class ExperimentRun(Base):
    """Experiment Tracking Entity for recording training runs and metrics."""

    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    run_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="COMPLETED"
    )  # RUNNING, COMPLETED, FAILED
    hyperparameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    val_loss: Mapped[float] = mapped_column(Float, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, default=0.0)
    roc_auc: Mapped[float] = mapped_column(Float, default=0.0)
    training_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class DriftRecord(Base):
    """Model Drift Detection Entity for tracking statistical feature & confidence drift."""

    __tablename__ = "drift_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    drift_score: Mapped[float] = mapped_column(Float, nullable=False)
    drift_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_drift: Mapped[float] = mapped_column(Float, default=0.0)
    class_distribution: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    sample_size: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class AlertRecord(Base):
    """Monitoring Alert Entity for storing active & historical system/model alerts."""

    __tablename__ = "alert_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    alert_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # HIGH_LATENCY, HIGH_ERROR_RATE, DRIFT_DETECTED, LOW_CONFIDENCE, HIGH_CPU, HIGH_RAM
    severity: Mapped[str] = mapped_column(String(20), default="WARNING")  # INFO, WARNING, CRITICAL
    message: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
