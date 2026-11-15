import pytest
from app.core.config import settings
from app.core.logging_config import logger
from app.main import app
from fastapi.testclient import TestClient


def test_phase1_config_loads():
    assert settings.PROJECT_NAME == "AI Sentiment Analysis Platform"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.MAX_SINGLE_PREDICT_LEN == 5000


def test_phase1_logging_initializes():
    assert logger is not None
    logger.info("Phase 1 Logging Verification Check Passed.")


def test_phase1_application_starts():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()


def test_phase1_imports_succeed():
    import tensorflow as tf
    import fastapi
    import sqlalchemy
    import pydantic
    
    assert tf.__version__ is not None
    assert fastapi.__version__ is not None
