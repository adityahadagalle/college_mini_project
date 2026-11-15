import uuid
from datetime import datetime

from app.core.database import Base
from app.models.auth import Role as Role
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    role = Column(Enum(Role), default=Role.USER, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    batch_jobs = relationship("BatchJob", back_populates="user", cascade="all, delete-orphan")
    settings = relationship(
        "UserSettings",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    refresh_tokens = relationship(
        "RefreshToken", back_populates="user", cascade="all, delete-orphan"
    )
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    password_history = relationship(
        "PasswordHistory", back_populates="user", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    text_content = Column(Text, nullable=False)
    sentiment_label = Column(String(20), index=True, nullable=False)
    confidence_score = Column(Float, nullable=False)
    positive_prob = Column(Float, nullable=False)
    neutral_prob = Column(Float, nullable=False)
    negative_prob = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    explainability_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    user = relationship("User", back_populates="predictions")
    aspects = relationship("ReviewAspect", back_populates="prediction", cascade="all, delete-orphan")
    emotions = relationship("ReviewEmotion", back_populates="prediction", cascade="all, delete-orphan")
    topic_assignments = relationship("TopicAssignment", back_populates="prediction")


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, default=0, nullable=False)
    processed_rows = Column(Integer, default=0, nullable=False)
    status = Column(String(50), default="PENDING", index=True, nullable=False)
    result_file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="batch_jobs")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    confidence_threshold = Column(Float, default=0.50, nullable=False)
    default_export_format = Column(String(10), default="csv", nullable=False)
    theme = Column(String(20), default="dark", nullable=False)

    user = relationship("User", back_populates="settings")


class ReviewAspect(Base):
    __tablename__ = "review_aspects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    prediction_id = Column(
        String(36),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    aspect_term = Column(String(255), index=True, nullable=False)
    aspect_category = Column(String(100), index=True, nullable=True)
    is_implicit = Column(Boolean, default=False, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    sentiment_label = Column(String(20), index=True, nullable=False)  # POSITIVE, NEUTRAL, NEGATIVE
    confidence_score = Column(Float, nullable=False)
    positive_prob = Column(Float, nullable=False)
    neutral_prob = Column(Float, nullable=False)
    negative_prob = Column(Float, nullable=False)
    evidence_text = Column(String(500), nullable=True)
    model_name = Column(String(100), nullable=False, default="DeBERTa-v3-ABSA")
    model_version = Column(String(50), nullable=False, default="v1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    __table_args__ = (
        Index("idx_review_aspects_pred_sent", "prediction_id", "sentiment_label"),
    )

    prediction = relationship("Prediction", back_populates="aspects")


class ReviewEmotion(Base):
    __tablename__ = "review_emotions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    prediction_id = Column(
        String(36),
        ForeignKey("predictions.id", ondelete="CASCADE"),
        index=True,
        nullable=True,
    )
    primary_cx_category = Column(String(100), index=True, nullable=False)
    confidence_score = Column(Float, nullable=False)
    detected_emotions_json = Column(JSON, nullable=False)
    raw_emotions_json = Column(JSON, nullable=False)
    model_name = Column(String(100), nullable=False, default="RoBERTa-GoEmotions-MultiLabel")
    model_version = Column(String(50), nullable=False, default="v1.0.0")
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    __table_args__ = (
        Index("idx_review_emotions_pred_cat", "prediction_id", "primary_cx_category"),
    )

    prediction = relationship("Prediction", back_populates="emotions")


class ReviewEmbedding(Base):
    __tablename__ = "review_embeddings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    text_hash = Column(String(64), unique=True, index=True, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    embedding_json = Column(JSON, nullable=False)
    model_name = Column(
        String(100),
        nullable=False,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    dimension = Column(Integer, nullable=False, default=384)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)


class TopicDiscoveryRun(Base):
    __tablename__ = "topic_discovery_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    status = Column(String(50), default="PENDING", index=True, nullable=False)
    corpus_size = Column(Integer, default=0, nullable=False)
    num_topics = Column(Integer, default=0, nullable=False)
    outlier_count = Column(Integer, default=0, nullable=False)
    parameters_json = Column(JSON, nullable=False)
    metrics_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    topics = relationship(
        "DiscoveredTopic",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="DiscoveredTopic.topic_index",
    )
    assignments = relationship(
        "TopicAssignment",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class DiscoveredTopic(Base):
    __tablename__ = "discovered_topics"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    run_id = Column(
        String(36),
        ForeignKey("topic_discovery_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    topic_index = Column(Integer, index=True, nullable=False)  # -1 = Outliers / Unassigned
    auto_label = Column(String(255), nullable=False)
    human_label = Column(String(255), nullable=True)
    keywords_json = Column(JSON, nullable=False)  # [{"word": str, "score": float}]
    centroid_json = Column(JSON, nullable=True)  # List[float] (384-d)
    review_count = Column(Integer, default=0, nullable=False)
    cohesion_score = Column(Float, nullable=True)
    sentiment_breakdown_json = Column(JSON, nullable=True)  # {"positive": float, ...}
    is_archived = Column(Boolean, default=False, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    run = relationship("TopicDiscoveryRun", back_populates="topics")
    assignments = relationship(
        "TopicAssignment",
        back_populates="topic",
        cascade="all, delete-orphan",
    )
    audit_logs = relationship(
        "TopicAuditLog",
        back_populates="topic",
        cascade="all, delete-orphan",
    )


class TopicAssignment(Base):
    __tablename__ = "topic_assignments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    run_id = Column(
        String(36),
        ForeignKey("topic_discovery_runs.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    topic_id = Column(
        String(36),
        ForeignKey("discovered_topics.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    prediction_id = Column(
        String(36),
        ForeignKey("predictions.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    review_text = Column(Text, nullable=False)
    distance_to_centroid = Column(Float, nullable=True)
    is_outlier = Column(Boolean, default=False, index=True, nullable=False)
    is_representative = Column(Boolean, default=False, index=True, nullable=False)
    sentiment_label = Column(String(20), nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    __table_args__ = (
        Index("idx_topic_assignments_pred_topic", "prediction_id", "topic_id"),
    )

    run = relationship("TopicDiscoveryRun", back_populates="assignments")
    topic = relationship("DiscoveredTopic", back_populates="assignments")
    prediction = relationship("Prediction", back_populates="topic_assignments")


class TopicAuditLog(Base):
    __tablename__ = "topic_audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    topic_id = Column(
        String(36),
        ForeignKey("discovered_topics.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    action = Column(String(50), index=True, nullable=False)  # RENAME, MERGE, ARCHIVE
    details_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    topic = relationship("DiscoveredTopic", back_populates="audit_logs")


class IntelligenceBatchJob(Base):
    __tablename__ = "intelligence_batch_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    status = Column(String(50), default="PENDING", index=True, nullable=False)
    progress_pct = Column(Float, default=0.0, nullable=False)
    total_records = Column(Integer, default=0, nullable=False)
    processed_records = Column(Integer, default=0, nullable=False)
    succeeded_records = Column(Integer, default=0, nullable=False)
    failed_records = Column(Integer, default=0, nullable=False)
    partial_records = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    results_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    completed_at = Column(DateTime, nullable=True)
