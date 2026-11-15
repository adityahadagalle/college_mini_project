"""MLOps, Observability & Monitoring API Endpoints."""

from typing import List

from app.core.database import get_db
from app.core.prometheus import prometheus_exporter
from app.models.mlops import ExperimentRun, ModelVersion
from app.repositories.mlops_repository import (
    ExperimentRepository,
    ModelRegistryRepository,
    MonitoringRepository,
)
from app.schemas.mlops import (
    AlertRecordSchema,
    DriftMetricSchema,
    ExperimentRunCreate,
    ExperimentRunSchema,
    ModelVersionCreate,
    ModelVersionSchema,
    SystemMetricsSchema,
)
from app.services.drift_service import drift_detector
from app.services.monitoring_service import monitoring_service
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

router = APIRouter(prefix="/monitoring", tags=["MLOps & Monitoring"])


@router.get("/system", response_model=SystemMetricsSchema)  # type: ignore[misc]
def get_system_monitoring_metrics() -> SystemMetricsSchema:
    """Fetch current infrastructure CPU, RAM, throughput, and latency metrics."""
    return monitoring_service.get_system_metrics()


@router.get("/models", response_model=List[ModelVersionSchema])  # type: ignore[misc]
def list_registered_models(db: Session = Depends(get_db)) -> List[ModelVersionSchema]:
    """Retrieve all model versions registered in Model Registry."""
    repo = ModelRegistryRepository(db)
    models = repo.get_all()
    if not models:
        # Seed default active model v1.0.0 if empty
        default_model = ModelVersion(
            version="1.0.0",
            architecture="Bi-LSTM Embedding",
            framework="TensorFlow 2.x Keras",
            status="ACTIVE",
            is_active=True,
            artifact_path="models/sentiment_bilstm_v1.h5",
            accuracy=0.948,
            precision=0.951,
            recall=0.942,
            f1_score=0.946,
            parameters_count=1245800,
            description="Production Baseline Bi-LSTM Model",
        )
        models = [repo.create(default_model)]
    return [ModelVersionSchema.model_validate(m) for m in models]


@router.post(
    "/models",
    response_model=ModelVersionSchema,
    status_code=status.HTTP_201_CREATED,
)  # type: ignore[misc]
def register_model_version(
    payload: ModelVersionCreate,
    db: Session = Depends(get_db),
) -> ModelVersionSchema:
    """Register a new model version candidate in Model Registry."""
    repo = ModelRegistryRepository(db)
    if repo.get_by_version(payload.version):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model version '{payload.version}' is already registered.",
        )

    model_entity = ModelVersion(**payload.model_dump())
    created = repo.create(model_entity)
    return ModelVersionSchema.model_validate(created)  # type: ignore[no-any-return]


@router.post("/models/{version}/activate", response_model=ModelVersionSchema)  # type: ignore[misc]
def activate_model_version(
    version: str,
    db: Session = Depends(get_db),
) -> ModelVersionSchema:
    """Promote or rollback active production model deployment to target version."""
    repo = ModelRegistryRepository(db)
    activated = repo.activate_version(version)
    if not activated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version '{version}' not found in registry.",
        )
    return ModelVersionSchema.model_validate(activated)  # type: ignore[no-any-return]


@router.get("/drift", response_model=DriftMetricSchema)  # type: ignore[misc]
def get_model_drift_metrics(db: Session = Depends(get_db)) -> DriftMetricSchema:
    """Compute and retrieve statistical model feature and confidence drift."""
    repo = MonitoringRepository(db)
    latest = repo.get_latest_drift()
    if not latest:
        # Calculate initial drift
        recent_confs = [0.94, 0.91, 0.95, 0.88, 0.96, 0.92]
        recent_sentiments = [
            "POSITIVE",
            "POSITIVE",
            "NEGATIVE",
            "NEUTRAL",
            "POSITIVE",
            "NEGATIVE",
        ]
        drift_entity = drift_detector.calculate_drift("1.0.0", recent_confs, recent_sentiments)
        latest = repo.record_drift(drift_entity)

    return DriftMetricSchema.model_validate(latest)  # type: ignore[no-any-return]


@router.get("/experiments", response_model=List[ExperimentRunSchema])  # type: ignore[misc]
def list_experiment_runs(db: Session = Depends(get_db)) -> List[ExperimentRunSchema]:
    """Retrieve experiment tracking records."""
    repo = ExperimentRepository(db)
    experiments = repo.get_all()
    if not experiments:
        # Seed initial experiment run
        exp = ExperimentRun(
            experiment_name="sentiment-bilstm-opt",
            run_name="run-adam-lr001",
            status="COMPLETED",
            hyperparameters={
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 10,
                "optimizer": "Adam",
            },
            metrics={"train_acc": 0.962, "val_loss": 0.182, "roc_auc": 0.985},
            accuracy=0.948,
            val_loss=0.182,
            f1_score=0.946,
            roc_auc=0.985,
            training_duration_seconds=142.5,
        )
        experiments = [repo.create(exp)]
    return [ExperimentRunSchema.model_validate(e) for e in experiments]


@router.post(
    "/experiments",
    response_model=ExperimentRunSchema,
    status_code=status.HTTP_201_CREATED,
)  # type: ignore[misc]
def record_experiment_run(
    payload: ExperimentRunCreate,
    db: Session = Depends(get_db),
) -> ExperimentRunSchema:
    """Record a new training experiment run with hyperparameters and metrics."""
    repo = ExperimentRepository(db)
    run_entity = ExperimentRun(**payload.model_dump())
    created = repo.create(run_entity)
    return ExperimentRunSchema.model_validate(created)  # type: ignore[no-any-return]


@router.get("/alerts", response_model=List[AlertRecordSchema])  # type: ignore[misc]
def get_monitoring_alerts(db: Session = Depends(get_db)) -> List[AlertRecordSchema]:
    """Retrieve active system and model alerts."""
    metrics = monitoring_service.get_system_metrics()
    alerts = monitoring_service.evaluate_alert_rules(metrics)
    return [AlertRecordSchema.model_validate(a) for a in alerts]


@router.get("/metrics")  # type: ignore[misc]
def get_prometheus_metrics() -> Response:
    """Expose Prometheus text-formatted metrics."""
    content = prometheus_exporter.generate_prometheus_text()
    return Response(content=content, media_type="text/plain; version=0.0.4")
