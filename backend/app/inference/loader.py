"""Thread-safe singleton ModelLoader service for Keras models."""

import json
import os
import threading
from typing import Any, Optional

import tensorflow as tf

# Isolate TensorFlow from allocating GPU VRAM, reserving GPU exclusively for PyTorch
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass
from app.core.inference_exceptions import ModelNotLoadedError
from app.core.logging_config import logger


class ModelLoader:
    """Thread-safe singleton model loader for TensorFlow models."""

    _instance: Optional["ModelLoader"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        """Double-checked locking singleton instantiation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize ModelLoader attributes once."""
        if getattr(self, "_initialized", False):
            return
        self.model: Optional[Any] = None
        self.model_path: Optional[str] = None
        self.version: str = "1.0.0"
        self.max_sequence_length: int = 128  # updated from metadata on load
        self._load_lock = threading.Lock()
        self._initialized = True

    def load_model(self, model_path: str, version: str = "1.0.0") -> Any:
        """Load Keras .keras model or SavedModel format safely.

        Args:
            model_path (str): File path to .keras or SavedModel directory.
            version (str): Model version string.

        Returns:
            Any: Loaded Keras model instance or TFSMLayer object.

        Raises:
            ModelNotLoadedError: If file path is invalid or fails to load.
        """
        with self._load_lock:
            if self.model is not None and self.model_path == model_path and self.version == version:
                return self.model

            if not os.path.exists(model_path):
                raise ModelNotLoadedError(f"Model file path not found: {model_path}")

            try:
                if model_path.endswith(".keras"):
                    loaded = tf.keras.models.load_model(model_path)
                elif os.path.isdir(model_path):
                    try:
                        loaded = tf.keras.models.load_model(model_path)
                    except Exception:
                        loaded = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serve")
                else:
                    loaded = tf.keras.models.load_model(model_path)

                self.model = loaded
                self.model_path = model_path
                self.version = version
                self._load_sequence_length_from_metadata(model_path)
                self._warmup_inference()

                logger.info(f"Successfully loaded model from {model_path} " f"(version: {version})")
                return self.model

            except Exception as exc:
                raise ModelNotLoadedError(
                    f"Failed to load model from {model_path}: {str(exc)}"
                ) from exc

    def _load_sequence_length_from_metadata(self, model_path: str) -> None:
        """Read max_sequence_length from the model_metadata.json companion file.

        The exporter writes ``model_metadata.json`` in the same directory as the
        ``.keras`` file. If it cannot be read, the existing default is kept.
        """
        # Arguments documentation
        # Args:
        #     model_path (str): Path to the loaded model file or directory.

        try:
            model_dir = os.path.dirname(model_path) if not os.path.isdir(model_path) else model_path
            meta_path = os.path.join(model_dir, "model_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                seq_len = meta.get("hyperparameters", {}).get("max_sequence_length")
                if seq_len is not None:
                    self.max_sequence_length = int(seq_len)
                    logger.info(
                        f"max_sequence_length set to {self.max_sequence_length} "
                        "from model_metadata.json"
                    )
        except Exception:
            pass  # keep the current default silently

    def _warmup_inference(self) -> None:
        """Perform graph warm-up inference step on initial loading."""
        if self.model is None:
            return

        dummy = {
            "input_ids": tf.ones((1, self.max_sequence_length), dtype=tf.int32),
            "attention_mask": tf.ones((1, self.max_sequence_length), dtype=tf.int32),
        }
        try:
            _ = self.model(dummy)
            logger.info("ModelLoader inference warm-up step completed.")
        except Exception:
            pass

    def get_model(self) -> Any:
        """Retrieve active loaded model instance.

        Returns:
            Any: Active loaded model object.

        Raises:
            ModelNotLoadedError: If no model has been loaded yet.
        """
        if self.model is None:
            raise ModelNotLoadedError("No model is currently loaded in ModelLoader.")
        return self.model
