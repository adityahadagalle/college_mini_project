"""Dataset loading, splitting, validation, and versioning repository."""

import csv
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from app.core.dataset_exceptions import (
    CorruptedDatasetError,
    DatasetNotFoundError,
    InvalidDatasetError,
)
from app.datasets.config import DatasetConfig
from app.datasets.preprocessor import TextPreprocessor
from app.datasets.validator import DataValidator


class DatasetLoader:
    """Repository for loading, validating, splitting datasets."""

    def __init__(self, config: DatasetConfig):
        """Initialize loader with dataset configuration.

        Args:
            config (DatasetConfig): Dataset configuration settings.
        """
        self.config = config
        self.preprocessor = TextPreprocessor(config)

    def load_local_csv(
        self, file_path: str, text_col: str = "text", label_col: str = "label"
    ) -> Tuple[List[str], List[int]]:
        """Load and validate dataset from local CSV file.

        Args:
            file_path (str): Path to CSV file.
            text_col (str): Column name for text content.
            label_col (str): Column name for labels.

        Returns:
            Tuple[List[str], List[int]]: Raw texts and integer labels.

        Raises:
            DatasetNotFoundError: If file path does not exist.
            CorruptedDatasetError: If file is empty or corrupted UTF-8.
            InvalidDatasetError: If required columns or rows are missing.
        """
        if not os.path.exists(file_path):
            raise DatasetNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise CorruptedDatasetError(f"Dataset file is empty: {file_path}")

        texts: List[str] = []
        labels: List[int] = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="strict") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames or text_col not in reader.fieldnames:
                    raise InvalidDatasetError(
                        f"Missing required column '{text_col}' in CSV header."
                    )

                for row in reader:
                    text_val = row.get(text_col, "")
                    label_val = row.get(label_col, None)

                    if not text_val or not str(text_val).strip():
                        continue

                    try:
                        int_label = int(str(label_val).strip())
                    except (TypeError, ValueError):
                        raise InvalidDatasetError(f"Invalid label '{label_val}' in CSV dataset.")

                    if int_label not in (0, 1, 2):
                        raise InvalidDatasetError(
                            f"Unsupported label value {int_label}. " f"Must be 0, 1, or 2."
                        )

                    cleaned_text = self.preprocessor.clean_text(str(text_val))
                    DataValidator.validate_sample(cleaned_text, int_label)

                    texts.append(cleaned_text)
                    labels.append(int_label)

        except UnicodeDecodeError as err:
            raise CorruptedDatasetError(f"File contains corrupted UTF-8 bytes: {str(err)}") from err
        except Exception as err:
            if isinstance(
                err,
                (
                    DatasetNotFoundError,
                    CorruptedDatasetError,
                    InvalidDatasetError,
                ),
            ):
                raise err
            raise InvalidDatasetError(f"Failed to parse CSV dataset: {str(err)}") from err

        if len(texts) == 0:
            raise InvalidDatasetError("No valid samples were loaded from CSV file.")

        return texts, labels

    def load_imdb_sample(self, num_samples: int = 500) -> Tuple[List[str], List[int]]:
        """Load benchmark IMDB sentiment dataset samples.

        Args:
            num_samples (int): Number of synthetic/benchmark samples.

        Returns:
            Tuple[List[str], List[int]]: Cleaned texts and labels.
        """
        pos_templates = [
            "The movie was fantastic! Brilliant acting and direction.",
            "Outstanding performance! Enjoyed every minute of the film.",
            "Great storyline and wonderful customer experience.",
        ]
        neg_templates = [
            "Terrible film. Horrible acting and complete waste of time.",
            "Poor quality product. Broke after two days of use.",
            "Disappointing service and extremely rude staff.",
        ]

        raw_texts = []
        labels = []
        for i in range(num_samples):
            if i % 2 == 0:
                raw_texts.append(pos_templates[i % len(pos_templates)])
                labels.append(0)
            else:
                raw_texts.append(neg_templates[i % len(neg_templates)])
                labels.append(2)

        cleaned = self.preprocessor.transform_batch(raw_texts)
        return cleaned, labels

    def split_dataset(
        self, texts: List[str], labels: List[int]
    ) -> Dict[str, Tuple[List[str], List[int]]]:
        """Split dataset into Train, Validation, and Test subsets.

        Args:
            texts (List[str]): Full dataset texts.
            labels (List[int]): Full dataset labels.

        Returns:
            Dict[str, Tuple[List[str], List[int]]]: Split datasets dict.
        """
        total = len(texts)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.val_ratio)

        return {
            "train": (texts[:train_end], labels[:train_end]),
            "val": (texts[train_end:val_end], labels[train_end:val_end]),
            "test": (texts[val_end:], labels[val_end:]),
        }

    def generate_metadata(
        self,
        texts: List[str],
        labels: List[int],
        splits: Dict[str, Tuple[List[str], List[int]]],
        version: str = "1.0.0",
    ) -> Dict[str, Any]:
        """Generate dataset metadata, fingerprint, and split statistics.

        Args:
            texts (List[str]): Raw/cleaned texts.
            labels (List[int]): Labels.
            splits (Dict[str, Tuple[List[str], List[int]]]): Dataset splits.
            version (str): Semantic version string.

        Returns:
            Dict[str, Any]: Metadata registry dictionary.
        """
        hasher = hashlib.sha256()
        for t in texts:
            hasher.update(t.encode("utf-8"))
        checksum = hasher.hexdigest()

        stats = DataValidator.compute_dataset_statistics(texts, labels)

        split_stats = {}
        for k, (s_texts, s_labels) in splits.items():
            split_stats[k] = {
                "sample_count": len(s_texts),
                "positive_count": sum(1 for lbl in s_labels if lbl == 0),
                "neutral_count": sum(1 for lbl in s_labels if lbl == 1),
                "negative_count": sum(1 for lbl in s_labels if lbl == 2),
            }

        return {
            "version": version,
            "fingerprint": f"sha256:{checksum[:16]}",
            "checksum": checksum,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(texts),
            "split_statistics": split_stats,
            "dataset_statistics": stats,
        }
