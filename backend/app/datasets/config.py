"""Dataset and data pipeline configuration settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseSettings):
    """Pydantic configuration settings for dataset pipelines."""

    batch_size: int = Field(default=64, ge=1, le=1024)
    max_sequence_length: int = Field(default=128, ge=8, le=512)
    shuffle_buffer_size: int = Field(default=10000, ge=1)

    cache_dataset: bool = Field(default=True)
    prefetch_autotune: bool = Field(default=True)

    # Preprocessing options
    lowercase: bool = Field(default=True)
    normalize_unicode: bool = Field(default=True)
    remove_html_tags: bool = Field(default=True)
    remove_urls: bool = Field(default=True)
    remove_emails: bool = Field(default=True)
    handle_emojis: bool = Field(default=True)
    normalize_whitespace: bool = Field(default=True)
    preserve_sentiment_punctuation: bool = Field(default=True)

    # Tokenizer options
    tokenizer_name: str = Field(default="distilbert-base-uncased")

    # Split Ratios
    train_ratio: float = Field(default=0.8, ge=0.5, le=0.9)
    val_ratio: float = Field(default=0.1, ge=0.05, le=0.25)
    test_ratio: float = Field(default=0.1, ge=0.05, le=0.25)

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


default_dataset_config = DatasetConfig()
