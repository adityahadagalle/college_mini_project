"""Explainability Service"""

from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from app.core.logging_config import logger
from app.explainability.exceptions import (
    AttentionError,
    ExplainabilityError,
    GradientError,
    TokenImportanceError,
)
from app.inference.engine import InferenceEngine
from app.schemas.explainability import (
    AttentionMatrix,
    ConfidenceExplanation,
    ExplainabilityRequest,
    ExplainabilityResponse,
    IntegratedGradientResult,
    TokenImportance,
)


class ExplainabilityService:
    """Orchestrates the explainability pipeline for a single text input."""

    def __init__(self, inference_engine: InferenceEngine | None = None):
        """Inject the existing InferenceEngine (or create a default one).

        The service re‑uses the same ModelLoader, tokenizer and preprocessing
        logic already present in the inference stack – no duplicate forward passes
        or tokenizers are created.
        """
        self.engine = inference_engine or InferenceEngine()
        self._cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def explain(self, request: ExplainabilityRequest) -> ExplainabilityResponse:
        """Generate an explainability response for ``request.text``.

        Steps (all reuse existing components):
        1. Tokenize the text using the shared ``TFTokenizerLayer``.
        2. Run a single forward pass (logits + optional attentions).
        3. Compute Integrated Gradients for the chosen class.
        4. Extract (and possibly summarise) attention matrices.
        5. Derive token importance scores from the attributions.
        6. Build confidence explanation metrics.
        7. Assemble and return an immutable ``ExplainabilityResponse`` DTO.
        """
        try:
            # Tokenization & forward pass
            tokenized = self.engine.tokenizer_layer.tokenize(request.text)
            logits, attentions = self.engine._run_model_forward(tokenized)
            pred_idx = int(np.argmax(logits))
            pred_label = self.engine.LABEL_MAP[pred_idx]

            self._cache[request.text] = {
                "tokenized": tokenized,
                "logits": logits,
                "attentions": attentions,
                "prediction_idx": pred_idx,
            }

            target_idx = request.target_class if request.target_class is not None else pred_idx
            if target_idx < 0 or target_idx >= len(self.engine.LABEL_MAP):
                raise ExplainabilityError(f"Invalid target_class index {target_idx}")

            ig_result = self._compute_integrated_gradients(
                tokenized, target_idx, request.interpolation_steps
            )
            token_imp = self._build_token_importance(
                tokenized, ig_result.normalized, request.importance_threshold
            )
            att_matrices = self._extract_attentions(attentions)
            confidence_expl = self._build_confidence_explanation(logits, request.top_k)

            response = ExplainabilityResponse(
                text=request.text,
                prediction=pred_label,
                confidence=confidence_expl.confidence,
                token_importance=token_imp,
                attention=att_matrices,
                confidence_explanation=confidence_expl,
                integrated_gradients=IntegratedGradientResult(
                    attributions=ig_result.attributions,
                    normalized=ig_result.normalized,
                ),
            )
            logger.info({"event": "explainability_completed", "text_len": len(request.text)})
            return response
        except (GradientError, AttentionError, TokenImportanceError, ExplainabilityError) as exc:
            logger.error({"event": "explainability_failed", "error": str(exc)})
            raise
        except Exception as exc:
            logger.exception({"event": "explainability_unexpected_error", "error": str(exc)})
            raise ExplainabilityError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _compute_integrated_gradients(
        self, tokenized: Dict[str, tf.Tensor], target_idx: int, steps: int
    ) -> IntegratedGradientResult:
        """TensorFlow only Integrated Gradients implementation.
        - Baseline is a zero tensor of the same shape as the input embeddings.
        - Supports configurable interpolation ``steps`` (validated by the request).
        - Returns raw attributions per token and L2‑normalized attributions.
        """
        try:
            model = self.engine.model_loader.get_model()
            embed_layer = None
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Embedding) or hasattr(layer, "embeddings"):
                    embed_layer = layer
                    break
            if embed_layer is None:
                raise GradientError("Model does not expose an embedding layer for IG.")

            input_ids = tokenized["input_ids"]
            vocab_embeddings = embed_layer.embeddings
            token_embeddings = tf.nn.embedding_lookup(vocab_embeddings, input_ids)
            baseline = tf.zeros_like(token_embeddings)
            alphas = tf.linspace(0.0, 1.0, steps + 1)
            accumulated_grads = tf.zeros_like(token_embeddings, dtype=tf.float32)

            for alpha in alphas[1:]:
                _ = baseline + alpha * (token_embeddings - baseline)
                with tf.GradientTape() as tape:
                    tape.watch(vocab_embeddings)
                    logits = (
                        model(tokenized)[0]
                        if isinstance(model(tokenized), dict)
                        else model(tokenized)
                    )
                    loss = tf.gather(logits[0], target_idx)
                grads = tape.gradient(loss, vocab_embeddings)
                token_grads = tf.nn.embedding_lookup(grads, input_ids)
                accumulated_grads += token_grads

            avg_grads = accumulated_grads / tf.cast(steps, tf.float32)
            attributions = tf.reduce_sum(avg_grads, axis=-1)[0].numpy().tolist()
            norm = np.linalg.norm(attributions) if np.linalg.norm(attributions) != 0 else 1.0
            normalized = (np.array(attributions) / norm).tolist()
            return IntegratedGradientResult(attributions=attributions, normalized=normalized)
        except Exception as exc:
            raise GradientError(f"Integrated Gradients failed: {exc}") from exc

    def _build_token_importance(
        self, tokenized: Dict[str, tf.Tensor], normalized: List[float], threshold: float
    ) -> List[TokenImportance]:
        """Create ``TokenImportance`` objects from normalized IG scores.
        - Importance values are already in ``[-1, 1]`` after L2 normalization.
        - ``contribution`` is ``positive`` when the value > 0, else ``negative``.
        - Tokens with ``abs(importance) < threshold`` are omitted.
        """
        try:
            tokenizer = self.engine.tokenizer_layer.tokenizer
            if tokenizer is None:
                raise TokenImportanceError("Tokenizer unavailable for token reconstruction.")
            input_ids = tokenized["input_ids"][0].numpy().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            results: List[TokenImportance] = []
            for tok, imp in zip(tokens, normalized):
                if abs(imp) < threshold:
                    continue
                contribution = "positive" if imp > 0 else "negative"
                results.append(
                    TokenImportance(token=tok, importance=imp, contribution=contribution)
                )
            return results
        except Exception as exc:
            raise TokenImportanceError(str(exc)) from exc

    def _extract_attentions(self, attentions: Any) -> List[AttentionMatrix]:
        """Return normalized attention matrices, handling optional missing data.
        If ``attentions`` is ``None`` or not iterable, an empty list is returned.
        Large matrices (>512) are truncated to 512×512. Each matrix is L1‑normalized
        so rows sum to 1.
        """
        try:
            if attentions is None:
                return []
            att_list: List[AttentionMatrix] = []
            for layer_idx, layer_att in enumerate(attentions):
                batch_att = layer_att[0]
                for head_idx, head_mat in enumerate(batch_att):
                    mat = head_mat.numpy().tolist()
                    seq_len = len(mat)
                    if seq_len > 512:
                        mat = [row[:512] for row in mat[:512]]
                    norm_mat = []
                    for row in mat:
                        row_sum = sum(row) or 1.0
                        norm_mat.append([float(v) / row_sum for v in row])
                    att_list.append(
                        AttentionMatrix(layer=layer_idx, head=head_idx, matrix=norm_mat)
                    )
            return att_list
        except Exception as exc:
            raise AttentionError(str(exc)) from exc

    def _build_confidence_explanation(
        self, logits: np.ndarray[Any, Any], top_k: int
    ) -> ConfidenceExplanation:
        """Derive confidence metrics from ``logits``.
        - ``confidence``: softmax probability of the predicted class.
        - ``entropy``: -∑p·log p.
        - ``margin``: difference between top‑1 and top‑2 probabilities.
        - ``top_k``: list of (label, prob) for the ``top_k`` highest probabilities.
        """
        try:
            scaled = logits - np.max(logits)
            exp = np.exp(scaled)
            probs = exp / np.sum(exp)
            sorted_idx = np.argsort(probs)[::-1]
            top_labels = [self.engine.LABEL_MAP[i] for i in sorted_idx[:top_k]]
            top_probs = probs[sorted_idx[:top_k]].tolist()
            confidence = float(probs[sorted_idx[0]])
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            margin = float(probs[sorted_idx[0]] - probs[sorted_idx[1]]) if len(probs) > 1 else 0.0
            top_k_list = list(zip(top_labels, top_probs))
            return ConfidenceExplanation(
                confidence=confidence,
                entropy=entropy,
                margin=margin,
                top_k=top_k_list,
            )
        except Exception as exc:
            raise ExplainabilityError(str(exc)) from exc
