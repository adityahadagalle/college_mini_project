# backend/app/auth/service.py
"""Authentication Service implementation.

Provides methods for user registration, login, token generation, refresh, logout,
logout from all devices, password change and related utilities.

All JWT creation is confined to this service as required by the architecture.
"""

from datetime import timedelta
from typing import Any, Dict, Optional

from app.auth.exceptions import AuthenticationError, PasswordPolicyError, TokenRevokedError
from app.auth.repositories import AuditRepository, SessionRepository, UserRepository
from app.core.config import settings
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.database import Role, User


class AuthenticationService:
    """Business logic for authentication and token lifecycle.

    The service is intentionally stateless; it receives an ``AsyncSession``
    instance (or a ``UserRepository`` that already holds a session) and operates
    on it.  This supports dependency‑injection patterns used throughout the
    project.
    """

    def __init__(
        self,
        user_repo: UserRepository,
        session_repo: SessionRepository,
        audit_repo: AuditRepository,
    ) -> None:
        self.user_repo = user_repo
        self.session_repo = session_repo
        self.audit_repo = audit_repo

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _validate_password_policy(password: str) -> None:
        """Enforce a minimal password policy.

        * Minimum length 8 characters.
        * Must contain at least one uppercase, one lowercase, one digit.
        * No spaces.
        """
        if len(password) < 8:
            raise PasswordPolicyError("Password must be at least 8 characters long")
        if password.islower() or password.isupper():
            raise PasswordPolicyError("Password must contain both uppercase and lowercase letters")
        if not any(ch.isdigit() for ch in password):
            raise PasswordPolicyError("Password must contain at least one digit")
        if " " in password:
            raise PasswordPolicyError("Password cannot contain spaces")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    async def register_user(
        self,
        email: str,
        password: str,
        full_name: str,
        role: Role = Role.USER,
    ) -> User:
        """Create a new user after validating password policy.

        Raises:
            AuthenticationError: If the e‑mail already exists.
            PasswordPolicyError: If the password does not meet requirements.
        """
        self._validate_password_policy(password)
        existing = await self.user_repo.get_by_email(email)
        if existing:
            raise AuthenticationError("User with this email already exists")
        hashed = get_password_hash(password)
        user = await self.user_repo.create(
            email=email,
            hashed_password=hashed,
            full_name=full_name,
            role=role,
        )
        await self.audit_repo.log_event(
            user_id=user.id,
            action="register",
            details={"email": email, "full_name": full_name, "role": role.value},
        )
        return user

    async def authenticate_user(self, email: str, password: str) -> User:
        """Validate credentials and return the corresponding ``User``.

        Raises:
            AuthenticationError: If credentials are invalid.
        """
        user = await self.user_repo.get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid credentials")
        return user

    async def login(
        self,
        email: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform full login flow and return access/refresh tokens.

        The method creates a new session record and a refresh token that is stored
        in the database for future rotation.
        """
        user = await self.authenticate_user(email, password)
        access_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        # Create refresh token (a random UUID stored hashed)
        refresh_token_obj = await self.session_repo.create_refresh_token(user.id)
        # Audit login
        await self.audit_repo.log_event(
            user_id=user.id,
            action="login",
            details={"ip_address": ip_address, "user_agent": user_agent},
        )
        return {
            "access_token": access_token,
            "refresh_token": getattr(
                refresh_token_obj, "plain_token", ""
            ),  # plain token returned to client
            "token_type": "bearer",
        }

    async def refresh_access_token(
        self, refresh_token: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rotate a refresh token and issue a new access token.

        The incoming *refresh_token* is compared against the stored hash.  If it
        is valid and not revoked, a new access token and a new refresh token are
        generated; the old refresh token is revoked.
        """
        token_obj = await self.session_repo.get_valid_refresh_token(refresh_token)
        if not token_obj:
            raise TokenRevokedError("Refresh token is invalid or revoked")
        # Revoke old token
        await self.session_repo.revoke_refresh_token(token_obj.id)
        # Issue new tokens
        user = await self.user_repo.get(token_obj.user_id)
        access_token = create_access_token(subject=user.id)
        new_refresh = await self.session_repo.create_refresh_token(user.id)
        await self.audit_repo.log_event(
            user_id=user.id,
            action="token_refresh",
            details={"old_token_id": token_obj.id},
        )
        return {
            "access_token": access_token,
            "refresh_token": new_refresh.plain_token,
            "token_type": "bearer",
        }

    async def logout(
        self, refresh_token: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None
    ) -> None:
        """Revoke a single refresh token.

        Raises:
            TokenRevokedError: If the token cannot be found.
        """
        token_obj = await self.session_repo.get_valid_refresh_token(refresh_token)
        if not token_obj:
            raise TokenRevokedError("Refresh token not found")
        await self.session_repo.revoke_refresh_token(token_obj.id)
        await self.audit_repo.log_event(
            user_id=token_obj.user_id,
            action="logout",
            details={
                "refresh_token_id": token_obj.id,
                "ip_address": ip_address,
                "user_agent": user_agent,
            },
        )

    async def logout_all_devices(
        self, user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None
    ) -> None:
        """Revoke *all* refresh tokens for a user, effectively logging out everywhere."""
        await self.session_repo.revoke_all_user_tokens(user_id)
        await self.audit_repo.log_event(
            user_id=user_id,
            action="logout_all",
            details={"ip_address": ip_address, "user_agent": user_agent},
        )

    async def change_password(self, user_id: str, current_password: str, new_password: str) -> None:
        """Validate current password, enforce policy, and update the hash.

        All existing refresh tokens are revoked to force re‑authentication.
        """
        user = await self.user_repo.get(user_id)
        if not verify_password(current_password, user.hashed_password):
            raise AuthenticationError("Current password is incorrect")
        self._validate_password_policy(new_password)
        new_hashed = get_password_hash(new_password)
        await self.user_repo.update_password(user_id, new_hashed)
        # Invalidate all sessions after password change
        await self.session_repo.revoke_all_user_tokens(user_id)
        await self.audit_repo.log_event(
            user_id=user_id,
            action="password_change",
            details={},
        )
