"""Tokenization layer using Hugging Face AutoTokenizer for TensorFlow."""

import os
from typing import Any, Dict, List, Optional, Union

# Enforce Pure TensorFlow mode for Hugging Face Transformers
os.environ["USE_TF"] = "1"
os.environ["USE_TORCH"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import tensorflow as tf  # noqa: E402

try:
    from transformers import AutoTokenizer  # noqa: E402

    HAS_TRANSFORMERS = True
except Exception:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False

from app.core.dataset_exceptions import TokenizerError  # noqa: E402
from app.datasets.config import DatasetConfig  # noqa: E402
from app.datasets.preprocessor import NEGATION_WHITELIST  # noqa: E402


class TFTokenizerLayer:
    """TensorFlow-native tokenizer wrapper over Hugging Face AutoTokenizer."""

    def __init__(self, config: DatasetConfig):
        """Initialize Tokenizer for TensorFlow.

        Args:
            config (DatasetConfig): Dataset configuration settings.
        """
        self.config = config
        self.tokenizer: Optional[Any] = None
        if HAS_TRANSFORMERS and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            except Exception:
                self.tokenizer = None

    def tokenize(self, texts: Union[str, List[str]]) -> Dict[str, tf.Tensor]:
        """Tokenize input text into TensorFlow Tensors (return_tensors='tf').

        Args:
            texts (Union[str, List[str]]): Sequence or batch of sequences.

        Returns:
            Dict[str, tf.Tensor]: Dict with 'input_ids' and 'attention_mask'.

        Raises:
            TokenizerError: If tokenization fails or outputs non-TF objects.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not self.tokenizer:
            input_ids_list = []
            mask_list = []
            max_len = self.config.max_sequence_length
            for t in texts:
                ids = [abs(hash(w)) % 30000 + 1 for w in t.split()][:max_len]
                pad_len = max_len - len(ids)
                attention = [1] * len(ids) + [0] * pad_len
                ids = ids + [0] * pad_len
                input_ids_list.append(ids)
                mask_list.append(attention)

            return {
                "input_ids": tf.constant(input_ids_list, dtype=tf.int32),
                "attention_mask": tf.constant(mask_list, dtype=tf.int32),
            }

        try:
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="np",
            )

            input_ids = tf.constant(encoded["input_ids"], dtype=tf.int32)
            attention_mask = tf.constant(encoded["attention_mask"], dtype=tf.int32)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        except Exception as exc:
            raise TokenizerError(f"TF Tokenization failed: {str(exc)}") from exc
