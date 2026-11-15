"""Custom exception definitions for TensorFlow model training."""


class TrainingEngineError(Exception):
    """Base exception class for training pipeline operations."""

    pass


class TrainingError(TrainingEngineError):
    """Raised when compilation or step execution fails."""

    pass


class CheckpointError(TrainingEngineError):
    """Raised when saving or restoring checkpoints fails."""

    pass


class ModelExportError(TrainingEngineError):
    """Raised when exporting SavedModel or .keras fails."""

    pass


class MetricError(TrainingEngineError):
    """Raised when computing classification metrics fails."""

    pass
