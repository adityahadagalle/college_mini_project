"""Keras callback manager for training logging and checkpointing."""

import os
from typing import List

import tensorflow as tf
from app.training.config import TrainingConfig


class CallbackManager:
    """Configures Keras callbacks for training monitoring."""

    @staticmethod
    def get_callbacks(
        config: TrainingConfig,
    ) -> List[tf.keras.callbacks.Callback]:
        """Build standard list of Keras callbacks.

        Args:
            config (TrainingConfig): Training configuration settings.

        Returns:
            List[tf.keras.callbacks.Callback]: List of Keras callbacks.
        """
        callbacks: List[tf.keras.callbacks.Callback] = []

        log_dir = os.path.join(config.output_directory, "logs")
        checkpoint_dir = os.path.join(config.output_directory, "checkpoints")
        backup_dir = os.path.join(config.output_directory, "backup")

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)

        if config.early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config.patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            )

        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            )
        )

        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
            )
        )

        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=os.path.join(log_dir, "training_history.csv"),
                append=True,
            )
        )

        ckpt_path = os.path.join(checkpoint_dir, "ckpt_epoch_{epoch:02d}.keras")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            )
        )

        callbacks.append(
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=backup_dir,
                delete_checkpoint=True,
            )
        )

        return callbacks
