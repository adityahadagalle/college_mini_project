"""TensorFlow tf.data Dataset Pipeline builder."""

from typing import List, Optional

import tensorflow as tf
from app.datasets.config import DatasetConfig
from app.datasets.tokenizer import TFTokenizerLayer


class TFDataPipeline:
    """High-performance TensorFlow tf.data.Dataset pipeline builder."""

    def __init__(self, config: DatasetConfig):
        """Initialize pipeline builder.

        Args:
            config (DatasetConfig): Dataset configuration.
        """
        self.config = config
        self.tokenizer_layer = TFTokenizerLayer(config)

    def create_tf_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        mode: str = "training",
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset pipeline optimized with AUTOTUNE.

        Args:
            texts (List[str]): Input text sequences.
            labels (Optional[List[int]]): Integer label sequences.
            mode (str): Pipeline mode: 'training', 'validation', 'inference'.

        Returns:
            tf.data.Dataset: TensorFlow native dataset object.
        """
        tokenized = self.tokenizer_layer.tokenize(texts)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        if labels is not None:
            labels_tensor = tf.constant(labels, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices(
                (
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    },
                    labels_tensor,
                )
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )

        if mode == "training":
            if self.config.shuffle_buffer_size > 0:
                buf_size = min(len(texts), self.config.shuffle_buffer_size)
                dataset = dataset.shuffle(buffer_size=buf_size)
            dataset = dataset.batch(self.config.batch_size)
            if self.config.cache_dataset:
                dataset = dataset.cache()
            if self.config.prefetch_autotune:
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        elif mode == "validation":
            dataset = dataset.batch(self.config.batch_size)
            if self.config.cache_dataset:
                dataset = dataset.cache()
            if self.config.prefetch_autotune:
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        elif mode == "inference":
            dataset = dataset.batch(self.config.batch_size)
            if self.config.prefetch_autotune:
                dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset
