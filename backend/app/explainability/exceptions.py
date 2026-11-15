"""Explainability specific exceptions"""


class ExplainabilityError(Exception):
    """Base class for all explainability related errors."""


class GradientError(ExplainabilityError):
    """Raised when Integrated Gradients computation fails or parameters are invalid."""


class AttentionError(ExplainabilityError):
    """Raised when attention extraction fails or the model does not provide attentions."""


class TokenImportanceError(ExplainabilityError):
    """Raised when token importance computation fails."""


class ConfidenceExplanationError(ExplainabilityError):
    """Raised when confidence explanation computation fails."""
