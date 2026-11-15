from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.core.security import get_password_hash, verify_password
from app.models.auth import AuditLog, PasswordHistory, RefreshToken, Session
from app.models.database import Role, User
from sqlalchemy import delete, select, update
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession


class UserRepository:
    """Data‑access layer for the ``User`` entity."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.email == email))
        user: Optional[User] = result.scalars().first()
        return user

    async def get(self, user_id: str) -> User:
        result = await self.db.execute(select(User).where(User.id == user_id))
        user: Optional[User] = result.scalars().first()
        if not user:
            raise NoResultFound(f"User {user_id} not found")
        return user

    async def create(
        self, email: str, hashed_password: str, full_name: str, role: Role = Role.USER
    ) -> User:
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            role=role,
        )
        self.db.add(user)
        await self.db.flush()  # populate PK
        return user

    async def update_password(self, user_id: str, new_hash: str) -> None:
        await self.db.execute(
            update(User).where(User.id == user_id).values(hashed_password=new_hash)
        )
        await self.db.flush()

    async def increment_failed_logins(self, user_id: str) -> None:
        await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(failed_login_attempts=User.failed_login_attempts + 1)
        )
        await self.db.flush()

    async def reset_failed_logins(self, user_id: str) -> None:
        await self.db.execute(
            update(User)
            .where(User.id == user_id)
            .values(failed_login_attempts=0, lockout_until=None)
        )
        await self.db.flush()

    async def lock_account(self, user_id: str, lock_minutes: int = 15) -> None:
        from datetime import datetime, timedelta

        lock_until = datetime.utcnow() + timedelta(minutes=lock_minutes)
        await self.db.execute(
            update(User).where(User.id == user_id).values(lockout_until=lock_until)
        )
        await self.db.flush()


class RefreshTokenRepository:
    """Repository for refresh token CRUD operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self, user_id: str, token_hash: str, expires_at: Optional[datetime] = None
    ) -> RefreshToken:
        if expires_at is None:
            expires_at = datetime.utcnow() + timedelta(days=7)
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self.db.add(token)
        await self.db.flush()
        return token

    async def get_by_hash(self, token_hash: str) -> Optional[RefreshToken]:
        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        ret: Optional[RefreshToken] = result.scalars().first()
        return ret

    async def revoke(self, token_id: str) -> None:
        await self.db.execute(
            update(RefreshToken).where(RefreshToken.id == token_id).values(revoked=True)
        )
        await self.db.flush()

    async def revoke_all_for_user(self, user_id: str) -> None:
        await self.db.execute(
            update(RefreshToken).where(RefreshToken.user_id == user_id).values(revoked=True)
        )
        await self.db.flush()

    async def get_valid(self, token_hash: str) -> Optional[RefreshToken]:
        result = await self.db.execute(
            select(RefreshToken)
            .where(RefreshToken.token_hash == token_hash)
            .where(RefreshToken.revoked.is_(False))
        )
        ret: Optional[RefreshToken] = result.scalars().first()
        return ret


class SessionRepository:
    """Repository for user sessions (login sessions)."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Session:
        if expires_at is None:
            expires_at = datetime.utcnow() + timedelta(days=1)
        session = Session(
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        result = await self.db.execute(select(Session).where(Session.id == session_id))
        ret: Optional[Session] = result.scalars().first()
        return ret

    async def revoke(self, session_id: str) -> None:
        await self.db.execute(update(Session).where(Session.id == session_id).values(revoked=True))
        await self.db.flush()

    async def revoke_all_for_user(self, user_id: str) -> None:
        await self.db.execute(
            update(Session).where(Session.user_id == user_id).values(revoked=True)
        )
        await self.db.flush()

    async def create_refresh_token(self, user_id: str) -> RefreshToken:
        """Convenience method used by ``AuthenticationService``.

        * Generates a random UUID token string.
        * Stores a **hash** of the token (so the plain token never touches the DB).
        * Returns an object that contains ``plain_token`` for the client.
        """
        from datetime import datetime, timedelta
        from uuid import uuid4

        plain = str(uuid4())
        token_hash = get_password_hash(plain)
        expires_at = datetime.utcnow() + timedelta(days=7)  # default refresh lifetime
        token = RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        # Attach a transient attribute for the caller
        setattr(token, "plain_token", plain)
        self.db.add(token)
        await self.db.flush()
        return token

    async def get_valid_refresh_token(self, plain_token: str) -> Optional[RefreshToken]:
        """Validate a presented refresh token.

        The token is hashed and compared against the stored hash.
        Returns ``None`` if not found, revoked, or expired.
        """
        # Scan all non‑revoked tokens for the user – in production this would be indexed.
        result = await self.db.execute(select(RefreshToken).where(RefreshToken.revoked.is_(False)))
        for token in result.scalars():
            if verify_password(plain_token, token.token_hash):
                if token.expires_at and token.expires_at < datetime.utcnow():
                    return None
                ret: Optional[RefreshToken] = token
                return ret
        return None

    async def revoke_refresh_token(self, token_id: str) -> None:
        await self.db.execute(
            update(RefreshToken).where(RefreshToken.id == token_id).values(revoked=True)
        )
        await self.db.flush()

    async def revoke_all_user_tokens(self, user_id: str) -> None:
        await self.db.execute(
            update(RefreshToken).where(RefreshToken.user_id == user_id).values(revoked=True)
        )
        await self.db.flush()


class AuditRepository:
    """Simple repository for persisting audit events."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def log_event(
        self,
        user_id: Optional[str],
        action: str,
        details: Dict[str, Any],
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        correlation_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> AuditLog:
        audit = AuditLog(
            user_id=user_id,
            action=action,
            details=details,
            resource_type=resource_type,
            resource_id=resource_id,
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id,
            status=status,
        )
        self.db.add(audit)
        await self.db.flush()
        return audit


class PasswordHistoryRepository:
    """Repository to store the last 5 password hashes for a user."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def add(self, user_id: str, password_hash: str) -> None:
        ph = PasswordHistory(user_id=user_id, password_hash=password_hash)
        self.db.add(ph)
        await self.db.flush()
        # Keep only the 5 most recent entries
        result = await self.db.execute(
            select(PasswordHistory.id)
            .where(PasswordHistory.user_id == user_id)
            .order_by(PasswordHistory.created_at.desc())
        )
        ids = [row[0] for row in result.fetchall()]
        if len(ids) > 5:
            # delete older entries
            await self.db.execute(delete(PasswordHistory).where(PasswordHistory.id.in_(ids[5:])))
            await self.db.flush()

    async def recent_hashes(self, user_id: str) -> List[str]:
        result = await self.db.execute(
            select(PasswordHistory.password_hash)
            .where(PasswordHistory.user_id == user_id)
            .order_by(PasswordHistory.created_at.desc())
            .limit(5)
        )
        return [row[0] for row in result.fetchall()]
