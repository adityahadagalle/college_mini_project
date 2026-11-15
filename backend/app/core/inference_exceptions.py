"""Custom exceptions for inference engine operations."""


class InferenceEngineError(Exception):
    """Base exception for inference engine operations."""

    pass


class ModelNotLoadedError(InferenceEngineError):
    """Raised when inference is requested before loading model."""

    pass


class InferenceError(InferenceEngineError):
    """Raised when model inference execution fails."""

    pass


class PredictionError(InferenceEngineError):
    """Raised when confidence calculation fails."""

    pass


class BatchPredictionError(InferenceEngineError):
    """Raised when batch inference or file parsing fails."""

    pass


class TokenizerFailure(InferenceEngineError):
    """Raised when tokenizer execution fails during inference."""

    pass
