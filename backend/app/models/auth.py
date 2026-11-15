import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import ClassVar

from app.core.database import Base
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String
from sqlalchemy.orm import relationship


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


class Role(PyEnum):
    ADMIN = "admin"
    ANALYST = "analyst"
    USER = "user"


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash = Column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    expires_at = Column(
        DateTime,
        nullable=False,
    )
    revoked = Column(
        Boolean,
        default=False,
        nullable=False,
    )
    # Plain token is not persisted; service attaches it for client response
    plain_token: ClassVar[str]

    user = relationship("User", back_populates="refresh_tokens")


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ip_address = Column(
        String(45),
        nullable=True,
    )
    user_agent = Column(
        String(255),
        nullable=True,
    )
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )
    expires_at = Column(
        DateTime,
        nullable=False,
    )
    revoked = Column(
        Boolean,
        default=False,
        nullable=False,
    )

    user = relationship("User", back_populates="sessions")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(36), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(255), nullable=True)
    correlation_id = Column(String(36), nullable=True)
    status = Column(String(20), nullable=True)
    details = Column(JSON, nullable=True)
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    user = relationship("User", back_populates="audit_logs")


class PasswordHistory(Base):
    __tablename__ = "password_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="password_history")
