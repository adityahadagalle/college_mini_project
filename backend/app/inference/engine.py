"""High-performance TensorFlow InferenceEngine for sentiment analysis."""

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from app.core.inference_exceptions import InferenceError
from app.core.logging_config import logger
from app.datasets.config import DatasetConfig
from app.datasets.preprocessor import TextPreprocessor
from app.datasets.tokenizer import TFTokenizerLayer
from app.inference.loader import ModelLoader
from app.schemas.sentiment import (
    BatchPredictionResponse,
    PredictionMetadata,
    PredictionResponse,
    PredictionStatistics,
)


class InferenceEngine:
    """Production-grade TensorFlow inference engine."""

    LABEL_MAP: Dict[int, str] = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}

    def __init__(self, model_loader: Optional[ModelLoader] = None):
        """Initialize engine with preprocessor, tokenizer, and model loader.

        ``max_sequence_length`` is taken directly from the loader, which reads
        it from ``model_metadata.json`` written alongside the exported model.
        This guarantees tokenization always matches the model in memory.

        Args:
            model_loader (Optional[ModelLoader]): ModelLoader singleton.
        """
        self.model_loader = model_loader or ModelLoader()
        seq_len = self.model_loader.max_sequence_length
        self.config = DatasetConfig(max_sequence_length=seq_len)
        self.preprocessor = TextPreprocessor(self.config)
        self.tokenizer_layer = TFTokenizerLayer(self.config)

    def predict_single(self, text: str, temperature: float = 1.0) -> PredictionResponse:
        """Execute single-text sentiment prediction pipeline.

        Args:
            text (str): Input text sequence.
            temperature (float): Temperature scaling factor for calibration.

        Returns:
            PredictionResponse: Formatted prediction DTO response.

        Raises:
            InferenceError: If preprocessing, tokenization, or model fails.
        """
        start_time = time.perf_counter()
        logger.info("Prediction pipeline started for single input.")

        try:
            cleaned_text = self.preprocessor.clean_text(text)
            tokenized = self.tokenizer_layer.tokenize(cleaned_text)

            logits, attentions = self._run_model_forward(tokenized)

            probs, top_label, confidence = self._calibrate_probs(logits[0], temperature=temperature)

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            probabilities_dict = {self.LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}

            metadata = PredictionMetadata(
                model_name=f"TF-Sentiment-{self.model_loader.version}",
                version=self.model_loader.version,
                inference_latency_ms=round(latency_ms, 2),
                device="CPU",
            )

            logger.info(
                f"Prediction finished in {latency_ms:.2f}ms "
                f"(label: {top_label}, confidence: {confidence:.4f})"
            )

            return PredictionResponse(
                text=text,
                label=top_label,
                confidence=round(confidence, 4),
                probabilities=probabilities_dict,
                metadata=metadata,
            )

        except Exception as exc:
            logger.error(f"Inference pipeline error: {str(exc)}")
            raise InferenceError(f"Prediction failed: {str(exc)}") from exc

    def predict_batch(self, texts: List[str], temperature: float = 1.0) -> BatchPredictionResponse:
        """Execute batch sentiment prediction pipeline with micro-batching.

        Args:
            texts (List[str]): List of raw input texts.
            temperature (float): Temperature scaling factor.

        Returns:
            BatchPredictionResponse: Formatted batch prediction DTO response.
        """
        start_time = time.perf_counter()
        logger.info(f"Batch prediction started for {len(texts)} samples.")

        responses: List[PredictionResponse] = []
        pos_c, neu_c, neg_c = 0, 0, 0

        for text in texts:
            resp = self.predict_single(text, temperature=temperature)
            responses.append(resp)

            if resp.label == "POSITIVE":
                pos_c += 1
            elif resp.label == "NEUTRAL":
                neu_c += 1
            else:
                neg_c += 1

        total_latency = (time.perf_counter() - start_time) * 1000.0
        avg_latency = total_latency / max(len(texts), 1)

        stats = PredictionStatistics(
            total_samples=len(texts),
            positive_count=pos_c,
            neutral_count=neu_c,
            negative_count=neg_c,
            avg_latency_ms=round(avg_latency, 2),
        )

        return BatchPredictionResponse(predictions=responses, statistics=stats)

    def _run_model_forward(
        self, tokenized: Dict[str, tf.Tensor]
    ) -> Tuple[np.ndarray[Any, Any], Any]:
        """Execute model forward pass and return raw logits and attentions.

        Args:
            tokenized (Dict[str, tf.Tensor]): Tokenized tensor inputs.

        Returns:
            Tuple[np.ndarray, Any]: Raw logits matrix and attention tensors.
        """
        model = self.model_loader.get_model()
        outputs = model(tokenized)

        if isinstance(outputs, dict):
            logits_tensor = outputs.get("logits", list(outputs.values())[0])
            attentions = outputs.get("attentions", None)
        elif hasattr(outputs, "logits"):
            logits_tensor = outputs.logits
            attentions = getattr(outputs, "attentions", None)
        else:
            logits_tensor = outputs
            attentions = None

        logits_arr = (
            logits_tensor.numpy() if hasattr(logits_tensor, "numpy") else np.array(logits_tensor)
        )
        return logits_arr, attentions

    def _calibrate_probs(
        self, logits: np.ndarray[Any, Any], temperature: float = 1.0
    ) -> Tuple[np.ndarray[Any, Any], str, float]:
        """Apply temperature scaling softmax and return top prediction.

        Args:
            logits (np.ndarray): Unnormalized 1D logits array.
            temperature (float): Temperature scaling value.

        Returns:
            Tuple[np.ndarray, str, float]: Probabilities, label, confidence.
        """
        scaled_logits = logits / max(temperature, 1e-5)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        top_idx = int(np.argmax(probs))
        top_label = self.LABEL_MAP.get(top_idx, "UNKNOWN")
        confidence = float(probs[top_idx])

        return probs, top_label, confidence
