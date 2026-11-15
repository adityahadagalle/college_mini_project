"""Training configuration settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingConfig(BaseSettings):
    """Pydantic BaseSettings for TensorFlow training engine configuration."""

    epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=2e-5, gt=0.0, le=1.0)
    optimizer: str = Field(default="adamw")
    scheduler: str = Field(default="linear")
    batch_size: int = Field(default=32, ge=1, le=512)
    max_sequence_length: int = Field(default=128, ge=8, le=512)

    mixed_precision: bool = Field(default=False)
    random_seed: int = Field(default=42)
    gradient_clipping: float = Field(default=1.0, ge=0.0)

    early_stopping: bool = Field(default=True)
    patience: int = Field(default=3, ge=1)

    checkpoint_frequency: int = Field(default=1, ge=1)
    output_directory: str = Field(default="./models/checkpoints")
    export_directory: str = Field(default="./models/exported")

    model_architecture: str = Field(default="distilbert")
    num_classes: int = Field(default=3, ge=2)

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


default_training_config = TrainingConfig()
