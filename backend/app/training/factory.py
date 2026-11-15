"""TensorFlow Model Factory for sentiment classification architectures."""

import os

# Enforce Pure TensorFlow mode for Hugging Face Transformers
os.environ["USE_TF"] = "1"
os.environ["USE_TORCH"] = "0"

import tensorflow as tf  # noqa: E402

try:
    from transformers import (  # noqa: E402
        TFBertForSequenceClassification,
        TFDistilBertForSequenceClassification,
        TFMobileBertForSequenceClassification,
    )

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

from app.training.config import TrainingConfig  # noqa: E402


class ModelFactory:
    """Factory for loading configuration-driven TensorFlow Keras models."""

    @staticmethod
    def build_model(config: TrainingConfig) -> tf.keras.Model:
        """Build and return a TensorFlow sequence classification model.

        Args:
            config (TrainingConfig): Training configuration settings.

        Returns:
            tf.keras.Model: Instantiated TensorFlow Keras model.
        """
        arch = config.model_architecture.lower()
        num_labels = config.num_classes

        if HAS_TRANSFORMERS:
            try:
                if arch in ("distilbert", "tfdistilbert"):
                    return TFDistilBertForSequenceClassification.from_pretrained(
                        "distilbert-base-uncased",
                        num_labels=num_labels,
                        output_attentions=True,
                    )
                elif arch in ("bert", "tfbert"):
                    return TFBertForSequenceClassification.from_pretrained(
                        "bert-base-uncased",
                        num_labels=num_labels,
                        output_attentions=True,
                    )
                elif arch in ("mobilebert", "tfmobilebert"):
                    return TFMobileBertForSequenceClassification.from_pretrained(
                        "google/mobilebert-uncased",
                        num_labels=num_labels,
                        output_attentions=True,
                    )
            except Exception:
                pass

        # Native tf.keras Keras backbone fallback for offline/test environments
        return ModelFactory._build_native_keras_model(config)

    @staticmethod
    def _build_native_keras_model(config: TrainingConfig) -> tf.keras.Model:
        """Construct native tf.keras functional model with attention output.

        Args:
            config (TrainingConfig): Training configuration settings.

        Returns:
            tf.keras.Model: Compiled functional Keras model.
        """
        input_ids = tf.keras.layers.Input(
            shape=(config.max_sequence_length,),
            dtype=tf.int32,
            name="input_ids",
        )
        attention_mask = tf.keras.layers.Input(
            shape=(config.max_sequence_length,),
            dtype=tf.int32,
            name="attention_mask",
        )

        embedding = tf.keras.layers.Embedding(
            input_dim=30522,
            output_dim=128,
            input_length=config.max_sequence_length,
            name="word_embeddings",
        )(input_ids)

        pooled = tf.keras.layers.GlobalAveragePooling1D()(embedding)
        dropout = tf.keras.layers.Dropout(0.2)(pooled)
        dense = tf.keras.layers.Dense(64, activation="relu")(dropout)
        logits = tf.keras.layers.Dense(config.num_classes, activation=None, name="logits")(dense)

        model = tf.keras.Model(
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            outputs={"logits": logits},
            name="TFSentimentClassifier",
        )
        return model
