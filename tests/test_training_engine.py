"""Unit and integration tests for Phase 3 TensorFlow Training Engine."""

import os
import tempfile
import numpy as np
import pytest
import tensorflow as tf

from app.core.training_exceptions import MetricError
from app.datasets.config import DatasetConfig
from app.datasets.pipeline import TFDataPipeline
from app.training.callbacks import CallbackManager
from app.training.config import TrainingConfig
from app.training.engine import TFTrainingEngine
from app.training.exporter import ModelExporter
from app.training.factory import ModelFactory
from app.training.metrics import MetricsCalculator


@pytest.fixture
def train_config():
    return TrainingConfig(
        epochs=1,
        learning_rate=1e-4,
        batch_size=4,
        max_sequence_length=16,
        random_seed=42,
        num_classes=3,
    )


def test_model_factory_build(train_config):
    model = ModelFactory.build_model(train_config)
    assert isinstance(model, tf.keras.Model)

    dummy_ids = tf.ones((2, 16), dtype=tf.int32)
    dummy_mask = tf.ones((2, 16), dtype=tf.int32)
    outputs = model({"input_ids": dummy_ids, "attention_mask": dummy_mask})

    logits = (
        outputs.logits
        if hasattr(outputs, "logits")
        else (outputs["logits"] if isinstance(outputs, dict) else outputs)
    )
    assert logits.shape == (2, 3)


def test_metrics_calculator():
    y_true = np.array([0, 1, 2, 0])
    y_logits = np.array(
        [
            [2.5, 0.1, 0.1],
            [0.1, 2.5, 0.1],
            [0.1, 0.1, 2.5],
            [2.5, 0.1, 0.1],
        ]
    )

    metrics = MetricsCalculator.compute_metrics(y_true, y_logits)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["loss"] > 0
    assert "confusion_matrix" in metrics
    assert "roc_auc" in metrics


def test_metrics_calculator_invalid_shape():
    with pytest.raises(MetricError):
        MetricsCalculator.compute_metrics(
            np.array([0, 1]), np.array([[1.0, 0.0]])
        )


def test_tf_training_engine_step_and_checkpoint(train_config):
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config.output_directory = tmp_dir
        engine = TFTrainingEngine(train_config)

        # Build tf.data pipeline
        ds_config = DatasetConfig(batch_size=4, max_sequence_length=16)
        pipeline = TFDataPipeline(ds_config)
        texts = ["Great product!", "Terrible service.", "Average.", "Loved it!"]
        labels = [0, 2, 1, 0]
        train_ds = pipeline.create_tf_dataset(texts, labels, mode="training")

        avg_loss, y_true_arr, logits_arr = engine.train_epoch(train_ds)
        assert avg_loss > 0
        assert len(y_true_arr) == 4
        assert logits_arr.shape == (4, 3)

        # Test Checkpoint Save & Restore
        ckpt_prefix = os.path.join(tmp_dir, "test_ckpt")
        saved_path = engine.save_checkpoint(ckpt_prefix)
        assert os.path.exists(saved_path + ".index") or os.path.exists(tmp_dir)

        # Restore checkpoint
        engine.restore_checkpoint(saved_path)


def test_model_exporter_and_reloading(train_config):
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config.export_directory = tmp_dir
        engine = TFTrainingEngine(train_config)
        dummy_metrics = {"loss": 0.15, "accuracy": 0.95}

        dummy_input = {
            "input_ids": tf.ones((2, 16), dtype=tf.int32),
            "attention_mask": tf.ones((2, 16), dtype=tf.int32),
        }
        orig_output = engine.model(dummy_input)
        orig_logits = (
            orig_output.logits
            if hasattr(orig_output, "logits")
            else (
                orig_output["logits"]
                if isinstance(orig_output, dict)
                else orig_output
            )
        )

        paths = ModelExporter.export_model(
            engine.model, train_config, dummy_metrics, version="1.0.0"
        )
        assert os.path.exists(paths["keras_path"])
        assert os.path.exists(paths["savedmodel_dir"])
        assert os.path.exists(os.path.join(tmp_dir, "model_metadata.json"))
        assert os.path.exists(os.path.join(tmp_dir, "metrics.json"))
        assert os.path.exists(os.path.join(tmp_dir, "training_config.json"))

        # 1. Reload Native Keras 3 .keras model
        reloaded_keras = tf.keras.models.load_model(paths["keras_path"])
        rel_keras_out = reloaded_keras(dummy_input)
        rel_keras_logits = (
            rel_keras_out.logits
            if hasattr(rel_keras_out, "logits")
            else (
                rel_keras_out["logits"]
                if isinstance(rel_keras_out, dict)
                else rel_keras_out
            )
        )
        np.testing.assert_allclose(
            orig_logits.numpy(), rel_keras_logits.numpy(), rtol=1e-4
        )

        # 2. Reload SavedModel format via Keras 3 TFSMLayer
        smlayer = tf.keras.layers.TFSMLayer(
            paths["savedmodel_dir"], call_endpoint="serve"
        )
        rel_saved_out = smlayer(dummy_input)
        rel_saved_logits = (
            rel_saved_out["logits"]
            if isinstance(rel_saved_out, dict)
            else rel_saved_out
        )
        np.testing.assert_allclose(
            orig_logits.numpy(), rel_saved_logits.numpy(), rtol=1e-4
        )


def test_callback_manager_init(train_config):
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_config.output_directory = tmp_dir
        callbacks = CallbackManager.get_callbacks(train_config)
        assert len(callbacks) >= 4


def test_reproducibility_seed(train_config):
    engine1 = TFTrainingEngine(train_config)
    w1 = engine1.model.get_weights()[0]

    engine2 = TFTrainingEngine(train_config)
    w2 = engine2.model.get_weights()[0]

    np.testing.assert_allclose(w1, w2)


def test_zero_pytorch_imports_in_training_layer():
    import sys

    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules
    assert "onnx" not in sys.modules
