"""MLOps Repository for Model Registry, Experiment Tracking, Drift Records, and Alerts."""

from typing import List, Optional

from app.models.mlops import AlertRecord, DriftRecord, ExperimentRun, ModelVersion
from sqlalchemy import desc, select
from sqlalchemy.orm import Session


class ModelRegistryRepository:
    """Repository for managing registered model versions and active model deployment rollbacks."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_all(self) -> List[ModelVersion]:
        stmt = select(ModelVersion).order_by(desc(ModelVersion.created_at))
        return list(self.db.scalars(stmt).all())

    def get_active(self) -> Optional[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.is_active == True)  # noqa: E712
        return self.db.scalars(stmt).first()  # type: ignore[no-any-return]

    def get_by_version(self, version: str) -> Optional[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.version == version)
        return self.db.scalars(stmt).first()  # type: ignore[no-any-return]

    def create(self, model_ver: ModelVersion) -> ModelVersion:
        self.db.add(model_ver)
        self.db.commit()
        self.db.refresh(model_ver)
        return model_ver

    def activate_version(self, version: str) -> Optional[ModelVersion]:
        """Activate a specific model version and demote previous active model versions."""
        target = self.get_by_version(version)
        if not target:
            return None

        # Deactivate current active models
        active_models = self.db.scalars(
            select(ModelVersion).where(ModelVersion.is_active == True)  # noqa: E712
        ).all()
        for m in active_models:
            m.is_active = False
            m.status = "ARCHIVED"

        target.is_active = True
        target.status = "ACTIVE"
        self.db.commit()
        self.db.refresh(target)
        return target


class ExperimentRepository:
    """Repository for experiment tracking and metric records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_all(self, limit: int = 20) -> List[ExperimentRun]:
        stmt = select(ExperimentRun).order_by(desc(ExperimentRun.created_at)).limit(limit)
        return list(self.db.scalars(stmt).all())

    def create(self, run: ExperimentRun) -> ExperimentRun:
        self.db.add(run)
        self.db.commit()
        self.db.refresh(run)
        return run


class MonitoringRepository:
    """Repository for drift records and active alert records."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def record_drift(self, drift: DriftRecord) -> DriftRecord:
        self.db.add(drift)
        self.db.commit()
        self.db.refresh(drift)
        return drift

    def get_latest_drift(self) -> Optional[DriftRecord]:
        stmt = select(DriftRecord).order_by(desc(DriftRecord.recorded_at))
        return self.db.scalars(stmt).first()  # type: ignore[no-any-return]

    def create_alert(self, alert: AlertRecord) -> AlertRecord:
        self.db.add(alert)
        self.db.commit()
        self.db.refresh(alert)
        return alert

    def get_active_alerts(self) -> List[AlertRecord]:
        stmt = (
            select(AlertRecord)
            .where(AlertRecord.is_active == True)  # noqa: E712
            .order_by(desc(AlertRecord.created_at))
        )
        return list(self.db.scalars(stmt).all())
