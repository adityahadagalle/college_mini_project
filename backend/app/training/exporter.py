"""Model export and versioning metadata manager using Keras 3 APIs."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import tensorflow as tf
from app.core.training_exceptions import ModelExportError
from app.training.config import TrainingConfig


class ModelExporter:
    """Exports Keras models (.keras, SavedModel) using Keras 3 APIs."""

    @staticmethod
    def export_model(
        model: tf.keras.Model,
        config: TrainingConfig,
        metrics: Dict[str, Any],
        version: str = "1.0.0",
        dataset_version: str = "1.0.0",
    ) -> Dict[str, str]:
        """Export Keras model and versioning metadata registry.

        Args:
            model (tf.keras.Model): Trained Keras model.
            config (TrainingConfig): Training configuration settings.
            metrics (Dict[str, Any]): Evaluated test/validation metrics.
            version (str): Model version string.
            dataset_version (str): Dataset version identifier string.

        Returns:
            Dict[str, str]: Dict of exported paths.

        Raises:
            ModelExportError: If model export fails.
        """
        export_dir = config.export_directory
        os.makedirs(export_dir, exist_ok=True)

        try:
            keras_path = os.path.join(export_dir, "model.keras")
            savedmodel_dir = os.path.join(export_dir, "saved_model")

            # 1. Export Native Keras 3 format (.keras)
            model.save(keras_path)

            # 2. Export TensorFlow SavedModel format via Keras export API
            if hasattr(model, "export"):
                model.export(savedmodel_dir)
            else:
                model.save(savedmodel_dir)

            # 3. Save metadata registry files
            now_str = datetime.now(timezone.utc).isoformat()
            metadata = {
                "model_name": f"TF-Sentiment-{config.model_architecture}",
                "version": version,
                "training_date": now_str,
                "dataset_version": dataset_version,
                "framework": "TensorFlow 2.x tf.keras 3",
                "hyperparameters": {
                    "learning_rate": config.learning_rate,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "optimizer": config.optimizer,
                    "max_sequence_length": config.max_sequence_length,
                },
            }

            with open(os.path.join(export_dir, "model_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            with open(os.path.join(export_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            with open(os.path.join(export_dir, "training_config.json"), "w") as f:
                json.dump(config.model_dump(), f, indent=2)

            return {
                "keras_path": keras_path,
                "savedmodel_dir": savedmodel_dir,
                "export_directory": export_dir,
            }

        except Exception as exc:
            raise ModelExportError(f"Model export failed: {str(exc)}") from exc
