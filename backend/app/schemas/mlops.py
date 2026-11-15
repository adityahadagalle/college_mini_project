"""MLOps Schemas for Model Registry, Experiment Tracking, Drift, Monitoring & Alerts."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelVersionSchema(BaseModel):
    """Model Registry Version Response Schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    version: str
    architecture: str
    framework: str
    status: str
    is_active: bool
    artifact_path: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    parameters_count: int
    description: Optional[str] = None
    created_at: datetime


class ModelVersionCreate(BaseModel):
    """Schema for registering a new model version."""

    version: str
    architecture: str = "Bi-LSTM Embedding"
    framework: str = "TensorFlow 2.x Keras"
    artifact_path: str
    accuracy: float = 0.948
    precision: float = 0.951
    recall: float = 0.942
    f1_score: float = 0.946
    parameters_count: int = 1245800
    description: Optional[str] = None


class ExperimentRunSchema(BaseModel):
    """Experiment Tracking Response Schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    experiment_name: str
    run_name: str
    status: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]
    accuracy: float
    val_loss: float
    f1_score: float
    roc_auc: float
    training_duration_seconds: float
    created_at: datetime


class ExperimentRunCreate(BaseModel):
    """Schema for recording a training experiment run."""

    experiment_name: str
    run_name: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    accuracy: float = 0.948
    val_loss: float = 0.182
    f1_score: float = 0.946
    roc_auc: float = 0.985
    training_duration_seconds: float = 120.0


class DriftMetricSchema(BaseModel):
    """Model Drift Analytics Schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    model_version: str
    drift_score: float
    drift_detected: bool
    confidence_drift: float
    class_distribution: Dict[str, Any]
    sample_size: int
    recorded_at: datetime


class AlertRecordSchema(BaseModel):
    """Monitoring Alert Record Schema."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    alert_type: str
    severity: str
    message: str
    is_active: bool
    created_at: datetime


class SystemMetricsSchema(BaseModel):
    """Real-time System & Infrastructure Metrics Schema."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_percent: Optional[float] = None
    inference_latency_p95_ms: float
    request_rate_per_sec: float
    error_rate_percent: float
    batch_throughput_per_sec: float
    queue_depth: int
    model_loading_time_ms: float
    active_users: int


class MlOpsOverviewSchema(BaseModel):
    """Aggregated MLOps System Summary."""

    active_model: ModelVersionSchema
    drift: DriftMetricSchema
    metrics: SystemMetricsSchema
    active_alerts: List[AlertRecordSchema]
    recent_experiments: List[ExperimentRunSchema]
