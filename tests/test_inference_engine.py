"""Unit and integration tests for Phase 4 Inference Engine."""

import os
import tempfile
import pytest

from app.core.inference_exceptions import (
    BatchPredictionError,
    InferenceError,
    ModelNotLoadedError,
)
from app.inference.engine import InferenceEngine
from app.inference.loader import ModelLoader
from app.schemas.sentiment import (
    BatchPredictionRequest,
    PredictionRequest,
)
from app.services.prediction_service import PredictionService
from app.training.config import TrainingConfig
from app.training.engine import TFTrainingEngine
from app.training.exporter import ModelExporter


@pytest.fixture
def trained_model_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config = TrainingConfig(
            epochs=1,
            batch_size=4,
            max_sequence_length=16,
            random_seed=42,
            export_directory=tmp_dir,
        )
        engine = TFTrainingEngine(train_config)
        dummy_metrics = {"loss": 0.12, "accuracy": 0.96}

        paths = ModelExporter.export_model(
            engine.model, train_config, dummy_metrics, version="1.0.0"
        )
        yield paths["keras_path"]


def test_model_loader_singleton(trained_model_path):
    loader1 = ModelLoader()
    loader2 = ModelLoader()
    assert loader1 is loader2

    model1 = loader1.load_model(trained_model_path, version="1.0.0")
    model2 = loader2.get_model()
    assert model1 is model2


def test_model_loader_not_found():
    loader = ModelLoader()
    with pytest.raises(ModelNotLoadedError):
        loader.load_model("non_existent_model.keras")


def test_inference_engine_single_prediction(trained_model_path):
    loader = ModelLoader()
    loader.load_model(trained_model_path)
    engine = InferenceEngine(loader)

    response = engine.predict_single(
        "Outstanding performance and great service!"
    )
    assert response.text == "Outstanding performance and great service!"
    assert response.label in ("POSITIVE", "NEUTRAL", "NEGATIVE")
    assert 0.0 <= response.confidence <= 1.0
    assert "POSITIVE" in response.probabilities
    assert response.metadata.inference_latency_ms > 0


def test_inference_engine_batch_prediction(trained_model_path):
    loader = ModelLoader()
    loader.load_model(trained_model_path)
    engine = InferenceEngine(loader)

    texts = [
        "Fantastic experience!",
        "Horrible service, completely disappointed.",
        "It was okay, average quality.",
    ]
    batch_resp = engine.predict_batch(texts)
    assert len(batch_resp.predictions) == 3
    assert batch_resp.statistics.total_samples == 3
    assert batch_resp.statistics.avg_latency_ms > 0


def test_inference_engine_special_cases(trained_model_path):
    loader = ModelLoader()
    loader.load_model(trained_model_path)
    engine = InferenceEngine(loader)

    # 1. Unicode & Emojis
    resp_emoji = engine.predict_single("Super happy! 😀🚀🔥")
    assert resp_emoji.confidence > 0

    # 2. Long text
    long_text = "Word " * 200
    resp_long = engine.predict_single(long_text)
    assert resp_long.confidence > 0

    # 3. Temperature scaling
    resp_temp = engine.predict_single("Test text", temperature=2.0)
    assert resp_temp.confidence > 0


def test_prediction_service_layer(trained_model_path):
    loader = ModelLoader()
    loader.load_model(trained_model_path)
    engine = InferenceEngine(loader)
    service = PredictionService(engine)

    req = PredictionRequest(text="Excellent product quality!")
    resp = service.predict_single(req)
    assert resp.label in ("POSITIVE", "NEUTRAL", "NEGATIVE")

    batch_req = BatchPredictionRequest(
        texts=["Good quality", "Bad quality"]
    )
    batch_resp = service.predict_batch(batch_req)
    assert len(batch_resp.predictions) == 2


def test_zero_pytorch_imports_in_inference_layer():
    import sys

    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "onnx" not in sys.modules
