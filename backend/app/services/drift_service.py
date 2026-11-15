"""Model Drift Detection Engine.

Provides statistical drift calculations comparing current inference confidence scores
and prediction class distributions against baseline distributions.
"""

from typing import Any, Dict, List

import numpy as np
from app.models.mlops import DriftRecord


class DriftDetectorService:
    """Service for computing confidence drift and prediction class distribution drift."""

    def __init__(
        self, baseline_confidence_mean: float = 0.92, drift_threshold: float = 0.15
    ) -> None:
        self.baseline_confidence_mean = baseline_confidence_mean
        self.drift_threshold = drift_threshold

    def calculate_drift(
        self,
        model_version: str,
        recent_confidences: List[float],
        recent_sentiments: List[str],
    ) -> DriftRecord:
        """Compute statistical drift over recent inference batch window."""
        sample_size = len(recent_confidences)
        if sample_size == 0:
            return DriftRecord(
                model_version=model_version,
                drift_score=0.0,
                drift_detected=False,
                confidence_drift=0.0,
                class_distribution={"POSITIVE": 0.33, "NEUTRAL": 0.33, "NEGATIVE": 0.34},
                sample_size=0,
            )

        current_mean_conf = float(np.mean(recent_confidences))
        confidence_drift = abs(self.baseline_confidence_mean - current_mean_conf)

        # Class distribution ratios
        pos_count = recent_sentiments.count("POSITIVE")
        neu_count = recent_sentiments.count("NEUTRAL")
        neg_count = recent_sentiments.count("NEGATIVE")

        class_dist: Dict[str, Any] = {
            "POSITIVE": round(pos_count / sample_size, 3),
            "NEUTRAL": round(neu_count / sample_size, 3),
            "NEGATIVE": round(neg_count / sample_size, 3),
        }

        # Calculate composite drift score
        drift_score = round(
            confidence_drift + (0.5 * abs(class_dist["POSITIVE"] - class_dist["NEGATIVE"])), 4
        )
        drift_detected = drift_score > self.drift_threshold

        return DriftRecord(
            model_version=model_version,
            drift_score=drift_score,
            drift_detected=drift_detected,
            confidence_drift=round(confidence_drift, 4),
            class_distribution=class_dist,
            sample_size=sample_size,
        )


drift_detector = DriftDetectorService()
