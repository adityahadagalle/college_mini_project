"""Custom exceptions for dataset engineering and data pipelines."""


class DatasetError(Exception):
    """Base exception for dataset pipeline operations."""

    pass


class DatasetNotFoundError(DatasetError):
    """Raised when a requested dataset path or resource is missing."""

    pass


class InvalidDatasetError(DatasetError):
    """Raised when dataset structure, schema, or labels fail validation."""

    pass


class CorruptedDatasetError(DatasetError):
    """Raised when dataset files contain unparseable bytes or records."""

    pass


class TokenizerError(DatasetError):
    """Raised when Hugging Face TensorFlow tokenization fails."""

    pass


class PreprocessingError(DatasetError):
    """Raised when text cleaning or regex normalization fails."""

    pass
