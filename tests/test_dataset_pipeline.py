"""Comprehensive unit tests for Phase 2 Dataset Engineering & Data Pipeline."""

import os
import tempfile
import pytest
import tensorflow as tf

from app.core.dataset_exceptions import (
    CorruptedDatasetError,
    DatasetNotFoundError,
    InvalidDatasetError,
    PreprocessingError,
)
from app.datasets.config import DatasetConfig
from app.datasets.loader import DatasetLoader
from app.datasets.pipeline import TFDataPipeline
from app.datasets.preprocessor import TextPreprocessor
from app.datasets.tokenizer import TFTokenizerLayer
from app.datasets.validator import DataValidator


@pytest.fixture
def sample_config():
    return DatasetConfig(
        batch_size=4,
        max_sequence_length=16,
        shuffle_buffer_size=10,
        lowercase=True,
        remove_html_tags=True,
        remove_urls=True,
        remove_emails=True,
        handle_emojis=True,
        preserve_sentiment_punctuation=True,
    )


def test_text_preprocessor(sample_config):
    preprocessor = TextPreprocessor(sample_config)

    raw_text = "Check out <p>this</p> amazing review! Visit https://example.com or contact user@test.com 😀"
    cleaned = preprocessor.clean_text(raw_text)

    assert "<p>" not in cleaned
    assert "https://" not in cleaned
    assert "user@test.com" not in cleaned
    assert "!" in cleaned  # Sentiment punctuation preserved
    assert cleaned.islower()


def test_text_preprocessor_invalid_type(sample_config):
    preprocessor = TextPreprocessor(sample_config)
    with pytest.raises(PreprocessingError):
        preprocessor.clean_text(12345)  # type: ignore


def test_data_validator_sample():
    assert DataValidator.validate_sample("Great movie!", 0) is True
    assert DataValidator.validate_sample("Terrible!", "NEGATIVE") is True

    with pytest.raises(InvalidDatasetError):
        DataValidator.validate_sample("", 0)

    with pytest.raises(InvalidDatasetError):
        DataValidator.validate_sample("Good", 99)


def test_data_validator_statistics():
    texts = [
        "Great movie!",
        "Terrible movie!",
        "Great movie!",
        "Average film."
    ]
    labels = [0, 2, 0, 1]

    stats = DataValidator.compute_dataset_statistics(texts, labels)
    assert stats["total_samples"] == 4
    assert stats["duplicate_count"] == 1
    assert stats["missing_count"] == 0
    assert "vocabulary_size" in stats
    assert "class_imbalance_report" in stats


def test_tokenizer_layer_tf_output(sample_config):
    tokenizer_layer = TFTokenizerLayer(sample_config)
    texts = ["This is a test sentence.", "Another test."]

    encoded = tokenizer_layer.tokenize(texts)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded

    assert isinstance(encoded["input_ids"], tf.Tensor)
    assert isinstance(encoded["attention_mask"], tf.Tensor)
    assert encoded["input_ids"].shape == (2, 16)


def test_tf_data_pipeline_modes(sample_config):
    pipeline = TFDataPipeline(sample_config)
    texts = ["Great product!", "Bad experience.", "Average.", "Loved it!"]
    labels = [0, 2, 1, 0]

    # Training mode dataset
    train_ds = pipeline.create_tf_dataset(texts, labels, mode="training")
    assert isinstance(train_ds, tf.data.Dataset)

    # Inspect first batch
    for batch_x, batch_y in train_ds.take(1):
        assert "input_ids" in batch_x
        assert "attention_mask" in batch_x
        assert batch_x["input_ids"].shape[0] <= 4
        assert batch_y.shape[0] <= 4

    # Inference mode dataset (no labels)
    infer_ds = pipeline.create_tf_dataset(texts, labels=None, mode="inference")
    for batch_x in infer_ds.take(1):
        assert "input_ids" in batch_x
        assert isinstance(batch_x["input_ids"], tf.Tensor)


def test_dataset_loader_imdb_and_split(sample_config):
    loader = DatasetLoader(sample_config)
    texts, labels = loader.load_imdb_sample(num_samples=20)

    assert len(texts) == 20
    assert len(labels) == 20

    splits = loader.split_dataset(texts, labels)
    assert "train" in splits
    assert "val" in splits
    assert "test" in splits

    meta = loader.generate_metadata(texts, labels, splits)
    assert meta["version"] == "1.0.0"
    assert "checksum" in meta
    assert "fingerprint" in meta
    assert meta["total_samples"] == 20


def test_dataset_loader_local_csv_and_errors(sample_config):
    loader = DatasetLoader(sample_config)

    # 1. Test missing file
    with pytest.raises(DatasetNotFoundError):
        loader.load_local_csv("non_existent_file.csv")

    # 2. Test empty file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_name = tmp.name

    try:
        with pytest.raises(CorruptedDatasetError):
            loader.load_local_csv(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

    # 3. Test valid CSV loading
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as tmp:
        tmp.write("text,label\n")
        tmp.write("Fantastic experience!,0\n")
        tmp.write("Very bad service.,2\n")
        valid_csv = tmp.name

    try:
        texts, labels = loader.load_local_csv(valid_csv)
        assert len(texts) == 2
        assert labels == [0, 2]
    finally:
        if os.path.exists(valid_csv):
            os.remove(valid_csv)


def test_zero_pytorch_imports_in_dataset_layer():
    """Ensure dataset preprocessors and loaders have zero PyTorch imports in their source."""
    import inspect
    from app.datasets import config, loader, pipeline, preprocessor, tokenizer
    for mod in [config, loader, pipeline, preprocessor, tokenizer]:
        src = inspect.getsource(mod)
        assert "import torch" not in src
        assert "from torch" not in src
