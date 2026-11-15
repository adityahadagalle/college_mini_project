"""Data validation and dataset statistics generation module."""

from collections import Counter
from typing import Any, Dict, List, Tuple

from app.core.dataset_exceptions import InvalidDatasetError


class DataValidator:
    """Validates raw dataset samples and generates descriptive statistics."""

    @staticmethod
    def validate_sample(text: Any, label: Any, valid_labels: Tuple[int, ...] = (0, 1, 2)) -> bool:
        """Validate a single (text, label) sample.

        Args:
            text (Any): Input text content.
            label (Any): Sample sentiment label.
            valid_labels (Tuple[int, ...]): Set of allowed label integers.

        Returns:
            bool: True if sample is valid.

        Raises:
            InvalidDatasetError: If sample structure or values are invalid.
        """
        if not isinstance(text, str):
            raise InvalidDatasetError(f"Invalid text type: expected str, got {type(text).__name__}")

        if len(text.strip()) == 0:
            raise InvalidDatasetError("Empty text sequence encountered.")

        if isinstance(label, str):
            label_map = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
            if label.upper() not in label_map:
                raise InvalidDatasetError(
                    f"Unknown string label '{label}'. " f"Expected one of {list(label_map.keys())}"
                )
            int_label = label_map[label.upper()]
        elif isinstance(label, int):
            int_label = label
        else:
            raise InvalidDatasetError(
                f"Invalid label type: expected int or str, " f"got {type(label).__name__}"
            )

        if int_label not in valid_labels:
            raise InvalidDatasetError(f"Invalid label value {int_label}. Must be in {valid_labels}")

        return True

    @staticmethod
    def compute_dataset_statistics(texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Generate dataset statistics and class imbalance report.


        Args:
            texts (List[str]): Cleaned dataset text samples.
            labels (List[int]): Integer sentiment labels.

        Returns:
            Dict[str, Any]: Descriptive statistics dictionary.
        """
        if len(texts) != len(labels):
            raise InvalidDatasetError(
                f"Mismatched dataset lengths: {len(texts)} texts vs " f"{len(labels)} labels."
            )

        total_samples = len(texts)
        if total_samples == 0:
            return {
                "total_samples": 0,
                "missing_count": 0,
                "duplicate_count": 0,
                "class_distribution": {},
                "average_sentence_length": 0.0,
                "vocabulary_size": 0,
                "token_length_histogram": {},
                "class_imbalance_report": {
                    "imbalance_ratio": 1.0,
                    "status": "balanced",
                },
            }

        missing_count = sum(1 for t in texts if not t or not t.strip())
        duplicate_count = total_samples - len(set(texts))

        words_per_text = [len(t.split()) for t in texts if t and t.strip()]
        avg_sentence_len = sum(words_per_text) / len(words_per_text) if words_per_text else 0.0

        vocab = set()
        for t in texts:
            if t:
                vocab.update(t.lower().split())

        label_counts = dict(Counter(labels))
        class_dist = {str(lbl): count for lbl, count in label_counts.items()}

        hist_bins = {"1-10": 0, "11-50": 0, "51-100": 0, "100+": 0}
        for length in words_per_text:
            if length <= 10:
                hist_bins["1-10"] += 1
            elif length <= 50:
                hist_bins["11-50"] += 1
            elif length <= 100:
                hist_bins["51-100"] += 1
            else:
                hist_bins["100+"] += 1

        counts = list(label_counts.values())
        if counts:
            max_c = max(counts)
            min_c = min(counts) if min(counts) > 0 else 1
            imbalance_ratio = round(max_c / min_c, 2)
        else:
            imbalance_ratio = 1.0

        is_imbalanced = imbalance_ratio > 1.5

        return {
            "total_samples": total_samples,
            "missing_count": missing_count,
            "duplicate_count": duplicate_count,
            "class_distribution": class_dist,
            "average_sentence_length": round(avg_sentence_len, 2),
            "vocabulary_size": len(vocab),
            "token_length_histogram": hist_bins,
            "class_imbalance_report": {
                "imbalance_ratio": imbalance_ratio,
                "status": "imbalanced" if is_imbalanced else "balanced",
            },
        }
