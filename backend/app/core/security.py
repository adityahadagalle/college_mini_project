from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union

from app.core.config import settings
from jose import jwt
from passlib.context import CryptContext

# Password hashing using pbkdf2_sha256
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)  # type: ignore[no-any-return]
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    res: str = str(pwd_context.hash(password))
    return res


def _create_jwt(
    subject: Union[str, Any], secret: str, expires_delta: Optional[timedelta] = None
) -> str:
    """Internal helper to create a JWT signed with the provided secret."""
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta if expires_delta else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode = {"exp": expire, "sub": str(subject)}
    return jwt.encode(to_encode, secret, algorithm=ALGORITHM)  # type: ignore[no-any-return]


def create_access_token(subject: Union[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token signed with ACCESS_TOKEN_SECRET."""
    return _create_jwt(subject, settings.ACCESS_TOKEN_SECRET, expires_delta)


def create_refresh_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a refresh token signed with REFRESH_TOKEN_SECRET."""
    return _create_jwt(subject, settings.REFRESH_TOKEN_SECRET, expires_delta)


def decode_access_token(token: str) -> Optional[str]:
    """Decode an access token; returns the subject (`sub`) or ``None`` if invalid."""
    try:
        payload = jwt.decode(token, settings.ACCESS_TOKEN_SECRET, algorithms=[ALGORITHM])
        res: Optional[str] = payload.get("sub")
        return res
    except Exception:
        return None


def decode_refresh_token(token: str) -> Optional[str]:
    """Decode a refresh token; returns the subject (`sub`) or ``None`` if invalid."""
    try:
        payload = jwt.decode(token, settings.REFRESH_TOKEN_SECRET, algorithms=[ALGORITHM])
        res: Optional[str] = payload.get("sub")
        return res
    except Exception:
        return None
