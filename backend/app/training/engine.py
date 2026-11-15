"""TensorFlow training engine with @tf.function graph compilation."""

import os
import random
from typing import Any, Dict, Tuple

import numpy as np
import tensorflow as tf
from app.core.logging_config import logger
from app.core.training_exceptions import CheckpointError
from app.training.config import TrainingConfig
from app.training.factory import ModelFactory


class TFTrainingEngine:
    """Production-grade TensorFlow 2.x training engine."""

    def __init__(self, config: TrainingConfig):
        """Initialize training engine with deterministic seed settings.

        Args:
            config (TrainingConfig): Training configuration settings.
        """
        self.config = config
        self._set_deterministic_seed(config.random_seed)
        self.model = ModelFactory.build_model(config)
        self.optimizer = self._init_optimizer()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self._warmup_graph()

    def _set_deterministic_seed(self, seed: int) -> None:
        """Set deterministic random seeds across Python, NumPy, TensorFlow.

        Args:
            seed (int): Seed number.
        """
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        logger.info(f"Set deterministic seed to {seed}")

    def _init_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Initialize optimizer based on configuration.

        Returns:
            tf.keras.optimizers.Optimizer: Configured TensorFlow optimizer.
        """
        lr = self.config.learning_rate
        opt_name = self.config.optimizer.lower()

        if opt_name == "adamw":
            return tf.keras.optimizers.AdamW(learning_rate=lr)
        elif opt_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            return tf.keras.optimizers.SGD(learning_rate=lr)

    def _warmup_graph(self) -> None:
        """Warm up TensorFlow compilation graph with a dummy tensor step."""
        dummy_x = {
            "input_ids": tf.ones((2, self.config.max_sequence_length), dtype=tf.int32),
            "attention_mask": tf.ones((2, self.config.max_sequence_length), dtype=tf.int32),
        }
        dummy_y = tf.constant([0, 1], dtype=tf.int32)
        _ = self._train_step(dummy_x, dummy_y)
        logger.info("TensorFlow graph warm-up completed.")

    @tf.function  # type: ignore[misc]
    def _train_step(
        self, x_batch: Dict[str, tf.Tensor], y_batch: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compiled TensorFlow training step with gradient clipping.

        Args:
            x_batch (Dict[str, tf.Tensor]): Batch features.
            y_batch (tf.Tensor): Batch target labels.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: Computed loss and output logits.
        """
        with tf.GradientTape() as tape:
            outputs = self.model(x_batch, training=True)
            logits = (
                outputs.logits
                if hasattr(outputs, "logits")
                else (outputs["logits"] if isinstance(outputs, dict) else outputs)
            )
            loss_value = self.loss_fn(y_batch, logits)

        grads = tape.gradient(loss_value, self.model.trainable_variables)

        if self.config.gradient_clipping > 0:
            grads, _ = tf.clip_by_global_norm(grads, self.config.gradient_clipping)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value, logits

    def train_epoch(
        self, dataset: tf.data.Dataset
    ) -> Tuple[float, np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Execute a single epoch custom training loop over tf.data.Dataset.

        Args:
            dataset (tf.data.Dataset): Batched training dataset.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]: Loss, ground truth, logits.

        """
        total_loss = 0.0
        steps = 0
        all_y_true = []
        all_logits = []

        for x_batch, y_batch in dataset:
            loss_val, logits = self._train_step(x_batch, y_batch)
            total_loss += float(loss_val.numpy())
            steps += 1
            all_y_true.extend(y_batch.numpy().tolist())
            all_logits.append(logits.numpy())

        avg_loss = total_loss / max(steps, 1)
        y_true_arr = np.array(all_y_true)
        num_c = self.config.num_classes
        logits_arr = np.vstack(all_logits) if all_logits else np.empty((0, num_c))

        return avg_loss, y_true_arr, logits_arr

    def save_checkpoint(self, checkpoint_path: str) -> str:
        """Save TensorFlow checkpoint to disk.

        Args:
            checkpoint_path (str): File path prefix for checkpoint.

        Returns:
            str: Saved checkpoint path.

        Raises:
            CheckpointError: If checkpoint save fails.
        """
        try:
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            saved_path: str = str(ckpt.save(file_prefix=checkpoint_path))
            logger.info(f"Saved checkpoint to {saved_path}")
            return saved_path
        except Exception as exc:
            raise CheckpointError(f"Checkpoint save failed: {str(exc)}") from exc

    def restore_checkpoint(self, checkpoint_path: str) -> None:
        """Restore TensorFlow checkpoint from disk.

        Args:
            checkpoint_path (str): File path to checkpoint.

        Raises:
            CheckpointError: If checkpoint restore fails.
        """
        try:
            ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
            status = ckpt.restore(checkpoint_path)
            status.expect_partial()
            logger.info(f"Restored checkpoint from {checkpoint_path}")
        except Exception as exc:
            raise CheckpointError(f"Checkpoint restore failed: {str(exc)}") from exc
