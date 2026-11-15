"""Unit and Integration Tests for Phase 9 MLOps, Monitoring & Observability.
"""

from typing import Generator
from app.core.database import Base, get_db
from app.core.prometheus import prometheus_exporter
from app.models.mlops import ExperimentRun, ModelVersion
from app.repositories.mlops_repository import ExperimentRepository, ModelRegistryRepository, MonitoringRepository
from app.services.drift_service import drift_detector
from app.services.monitoring_service import monitoring_service
from fastapi.testclient import TestClient
from app.main import app
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def setup_db() -> Generator[None, None, None]:
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_db: None) -> Generator[Session, None, None]:
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestModelRegistry:
    def test_model_version_creation_and_activation(self, db_session: Session) -> None:
        repo = ModelRegistryRepository(db_session)
        v1 = repo.create(
            ModelVersion(
                version="1.0.0",
                architecture="Bi-LSTM",
                framework="TensorFlow",
                artifact_path="path/v1.h5",
                is_active=True,
                status="ACTIVE",
            )
        )
        v2 = repo.create(
            ModelVersion(
                version="2.0.0",
                architecture="Transformer",
                framework="TensorFlow",
                artifact_path="path/v2.h5",
                is_active=False,
                status="STAGING",
            )
        )

        assert v1.is_active is True
        assert v2.is_active is False

        # Activate v2 (Rollback/Promotion)
        activated = repo.activate_version("2.0.0")
        assert activated is not None
        assert activated.is_active is True
        assert activated.status == "ACTIVE"

        v1_updated = repo.get_by_version("1.0.0")
        assert v1_updated is not None
        assert v1_updated.is_active is False
        assert v1_updated.status == "ARCHIVED"


class TestExperimentTracking:
    def test_record_experiment_run(self, db_session: Session) -> None:
        repo = ExperimentRepository(db_session)
        exp = repo.create(
            ExperimentRun(
                experiment_name="exp_test",
                run_name="run_1",
                hyperparameters={"lr": 0.001},
                metrics={"acc": 0.95},
                accuracy=0.95,
                val_loss=0.15,
                f1_score=0.94,
                roc_auc=0.98,
                training_duration_seconds=60.0,
            )
        )

        assert exp.id is not None
        all_exp = repo.get_all()
        assert len(all_exp) == 1
        assert all_exp[0].run_name == "run_1"


class TestDriftAndMonitoringServices:
    def test_drift_detector_service(self) -> None:
        recent_confs = [0.95, 0.90, 0.92, 0.88, 0.94]
        recent_sentiments = ["POSITIVE", "POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE"]
        drift_res = drift_detector.calculate_drift("1.0.0", recent_confs, recent_sentiments)

        assert drift_res.sample_size == 5
        assert drift_res.drift_score >= 0.0
        assert "POSITIVE" in drift_res.class_distribution

    def test_prometheus_metrics_exporter(self) -> None:
        prometheus_exporter.record_prediction(25.0)
        prometheus_exporter.record_api_request(200)
        prometheus_exporter.record_api_request(500)
        text_output = prometheus_exporter.generate_prometheus_text()

        assert "prediction_count" in text_output
        assert "api_requests_total" in text_output
        assert "api_errors_total" in text_output

    def test_monitoring_service(self) -> None:
        metrics = monitoring_service.get_system_metrics()
        assert metrics.cpu_percent >= 0.0
        assert metrics.memory_percent >= 0.0

        alerts = monitoring_service.evaluate_alert_rules(metrics, drift_score=0.20)
        assert len(alerts) >= 1  # Drift alert triggered


class TestMlOpsApiEndpoints:
    def test_mlops_system_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/v1/monitoring/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu_percent" in data
        assert "inference_latency_p95_ms" in data

    def test_mlops_models_and_activation_flow(self, client: TestClient) -> None:
        # GET models
        get_resp = client.get("/api/v1/monitoring/models")
        assert get_resp.status_code == 200
        models = get_resp.json()
        assert len(models) >= 1

        # POST register candidate version
        reg_resp = client.post(
            "/api/v1/monitoring/models",
            json={
                "version": "1.1.0",
                "architecture": "Bi-LSTM Enhanced",
                "artifact_path": "models/v110.h5",
                "accuracy": 0.955,
                "precision": 0.958,
                "recall": 0.950,
                "f1_score": 0.954,
                "parameters_count": 1300000,
            },
        )
        assert reg_resp.status_code == 201

        # Activate version 1.1.0
        act_resp = client.post("/api/v1/monitoring/models/1.1.0/activate")
        assert act_resp.status_code == 200
        assert act_resp.json()["is_active"] is True

    def test_mlops_drift_and_experiments_endpoints(self, client: TestClient) -> None:
        drift_resp = client.get("/api/v1/monitoring/drift")
        assert drift_resp.status_code == 200
        assert "drift_score" in drift_resp.json()

        exp_resp = client.get("/api/v1/monitoring/experiments")
        assert exp_resp.status_code == 200
        assert len(exp_resp.json()) >= 1

    def test_prometheus_metrics_endpoint(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "prediction_count" in resp.text
