from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Sentiment Analysis Platform"
    ENVIRONMENT: str = "development"
    API_V1_STR: str = "/api/v1"
    # Primary secret for signing JWTs – keep for backward compatibility
    SECRET_KEY: str = "PRODUCTION_SECRET_KEY_CHANGE_THIS_IN_ENV_FILE_32_BYTES_MIN"
    # New distinct secrets – never derived from each other
    ACCESS_TOKEN_SECRET: str = "ACCESS_TOKEN_SECRET_CHANGE_ME_IN_ENV"
    REFRESH_TOKEN_SECRET: str = "REFRESH_TOKEN_SECRET_CHANGE_ME_IN_ENV"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./sentiment.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # TensorFlow Model
    TF_MODEL_PATH: str = "models/tf_sentiment_model.keras"
    MODEL_METADATA_PATH: str = "models/model_metadata.json"

    # Limits & Thresholds
    MAX_SINGLE_PREDICT_LEN: int = 5000
    BATCH_PROCESSING_SIZE: int = 64
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
