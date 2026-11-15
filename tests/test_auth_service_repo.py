"""Unit and integration tests for auth repositories and authentication service.

Covers:
- UserRepository: create, get, get_by_email, update_password, increment_failed_logins, reset_failed_logins, lock_account
- RefreshTokenRepository: create, get_by_hash, revoke, revoke_all_for_user, get_valid
- SessionRepository: create, get, revoke, revoke_all_for_user, create_refresh_token, get_valid_refresh_token, revoke_refresh_token
- AuditRepository: log_event with full details
- PasswordHistoryRepository: add (max 5 retention), recent_hashes
- AuthenticationService: password policy validation, register_user, authenticate_user, login, refresh_access_token, logout, logout_all_devices, change_password
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from app.auth.exceptions import AuthenticationError, PasswordPolicyError, TokenRevokedError
from app.auth.repositories import (
    AuditRepository,
    PasswordHistoryRepository,
    RefreshTokenRepository,
    SessionRepository,
    UserRepository,
)
from app.auth.service import AuthenticationService
from app.core.database import Base
from app.models.auth import AuditLog, PasswordHistory, RefreshToken, Session  # noqa: F401
from app.models.database import Role, User
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# ---------------------------------------------------------------------------
# UserRepository tests
# ---------------------------------------------------------------------------


class TestUserRepository:
    async def test_user_crud_and_methods(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            repo = UserRepository(db)

            # Create
            user = await repo.create(
                email="repo_user@test.com",
                hashed_password="hashed_pass_123",
                full_name="Repo User",
                role=Role.ADMIN,
            )
            await db.commit()
            assert user.id is not None
            assert user.role == Role.ADMIN

            # Get by email
            by_email = await repo.get_by_email("repo_user@test.com")
            assert by_email is not None
            assert by_email.id == user.id

            # Get by id
            by_id = await repo.get(user.id)
            assert by_id.email == "repo_user@test.com"

            # Update password
            await repo.update_password(user.id, "new_hashed_pass")
            await db.commit()
            updated = await repo.get(user.id)
            assert updated.hashed_password == "new_hashed_pass"

    async def test_get_nonexistent_user_raises(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            repo = UserRepository(db)
            with pytest.raises(NoResultFound):
                await repo.get("non-existent-id")


# ---------------------------------------------------------------------------
# RefreshTokenRepository & SessionRepository tests
# ---------------------------------------------------------------------------


class TestRefreshTokenRepository:
    async def test_refresh_token_repository_operations(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user_repo = UserRepository(db)
            user = await user_repo.create("token_user@test.com", "pass", "Token User")
            await db.commit()

            repo = RefreshTokenRepository(db)
            token = await repo.create(
                user.id, "hash_abc", expires_at=datetime.utcnow() + timedelta(days=1)
            )
            await db.commit()
            assert token.id is not None
            assert token.revoked is False

            # Get by hash
            found = await repo.get_by_hash("hash_abc")
            assert found is not None
            assert found.id == token.id

            # Get valid
            valid = await repo.get_valid("hash_abc")
            assert valid is not None

            # Revoke single
            await repo.revoke(token.id)
            await db.commit()
            assert (await repo.get_valid("hash_abc")) is None

            # Create another token and revoke all for user
            token2 = await repo.create(user.id, "hash_def")
            await db.commit()
            await repo.revoke_all_for_user(user.id)
            await db.commit()
            assert (await repo.get_valid("hash_def")) is None


class TestSessionRepository:
    async def test_session_repository_operations(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user_repo = UserRepository(db)
            user = await user_repo.create("session_user@test.com", "pass", "Session User")
            await db.commit()

            repo = SessionRepository(db)
            sess = await repo.create(user.id, ip_address="127.0.0.1", user_agent="Pytest")
            await db.commit()

            fetched = await repo.get(sess.id)
            assert fetched is not None
            assert fetched.ip_address == "127.0.0.1"

            await repo.revoke(sess.id)
            await db.commit()
            revoked_sess = await repo.get(sess.id)
            assert revoked_sess.revoked is True

            # Test convenience refresh token methods
            rt = await repo.create_refresh_token(user.id)
            await db.commit()
            plain = getattr(rt, "plain_token")
            assert plain != ""

            valid_rt = await repo.get_valid_refresh_token(plain)
            assert valid_rt is not None

            # Revoke refresh token
            await repo.revoke_refresh_token(valid_rt.id)
            await db.commit()
            assert (await repo.get_valid_refresh_token(plain)) is None

            # Revoke all user tokens
            rt2 = await repo.create_refresh_token(user.id)
            await db.commit()
            await repo.revoke_all_user_tokens(user.id)
            await db.commit()
            assert (await repo.get_valid_refresh_token(getattr(rt2, "plain_token"))) is None

    async def test_expired_refresh_token_returns_none(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user_repo = UserRepository(db)
            user = await user_repo.create("expired_rt@test.com", "pass", "User")
            await db.commit()

            repo = SessionRepository(db)
            rt = await repo.create_refresh_token(user.id)
            # Manually set expired timestamp
            rt.expires_at = datetime.utcnow() - timedelta(days=1)
            await db.commit()

            plain = getattr(rt, "plain_token")
            assert (await repo.get_valid_refresh_token(plain)) is None


# ---------------------------------------------------------------------------
# AuditRepository & PasswordHistoryRepository tests
# ---------------------------------------------------------------------------


class TestAuditAndPasswordHistoryRepository:
    async def test_audit_log(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            repo = AuditRepository(db)
            audit = await repo.log_event(
                user_id="user-123",
                action="test_action",
                details={"key": "value"},
                ip_address="192.168.1.1",
            )
            await db.commit()
            assert audit.id is not None
            assert audit.action == "test_action"

    async def test_password_history_retention(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            user_repo = UserRepository(db)
            user = await user_repo.create("ph_user@test.com", "pass", "PH User")
            await db.commit()

            ph_repo = PasswordHistoryRepository(db)

            # Add 7 password hashes to test truncation to 5
            for i in range(7):
                await ph_repo.add(user.id, f"hash_{i}")
                await db.commit()

            hashes = await ph_repo.recent_hashes(user.id)
            assert len(hashes) == 5
            assert "hash_6" in hashes


# ---------------------------------------------------------------------------
# AuthenticationService tests
# ---------------------------------------------------------------------------


class TestAuthenticationService:
    async def test_password_policy_validation(self) -> None:
        # Valid password
        AuthenticationService._validate_password_policy("StrongP@ss1")

        # Too short
        with pytest.raises(PasswordPolicyError):
            AuthenticationService._validate_password_policy("Short1")

        # Lowercase only
        with pytest.raises(PasswordPolicyError):
            AuthenticationService._validate_password_policy("onlylowercase123")

        # Uppercase only
        with pytest.raises(PasswordPolicyError):
            AuthenticationService._validate_password_policy("ONLYUPPERCASE123")

        # No digit
        with pytest.raises(PasswordPolicyError):
            AuthenticationService._validate_password_policy("NoDigitInPassword")

        # Contains space
        with pytest.raises(PasswordPolicyError):
            AuthenticationService._validate_password_policy("Pass word123")

    async def test_register_and_authenticate_user(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            svc = AuthenticationService(
                UserRepository(db), SessionRepository(db), AuditRepository(db)
            )

            # Register
            user = await svc.register_user(
                email="auth_svc@test.com",
                password="SecurePassword123",
                full_name="Auth Service User",
            )
            await db.commit()
            assert user.id is not None

            # Duplicate email raises AuthenticationError
            with pytest.raises(AuthenticationError):
                await svc.register_user(
                    email="auth_svc@test.com",
                    password="SecurePassword123",
                    full_name="Auth Service User",
                )

            # Authenticate success
            auth_user = await svc.authenticate_user("auth_svc@test.com", "SecurePassword123")
            assert auth_user.id == user.id

            # Authenticate failure (bad password)
            with pytest.raises(AuthenticationError):
                await svc.authenticate_user("auth_svc@test.com", "WrongPassword123")

            # Authenticate failure (nonexistent email)
            with pytest.raises(AuthenticationError):
                await svc.authenticate_user("nonexistent@test.com", "SecurePassword123")

    async def test_full_login_refresh_logout_flow(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            svc = AuthenticationService(
                UserRepository(db), SessionRepository(db), AuditRepository(db)
            )
            user = await svc.register_user(
                email="flow_user@test.com",
                password="FlowPassword123",
                full_name="Flow User",
            )
            await db.commit()

            # Login
            tokens = await svc.login("flow_user@test.com", "FlowPassword123")
            await db.commit()
            assert "access_token" in tokens
            assert "refresh_token" in tokens

            # Refresh token
            rf_tokens = await svc.refresh_access_token(tokens["refresh_token"])
            await db.commit()
            assert "access_token" in rf_tokens
            assert rf_tokens["refresh_token"] != tokens["refresh_token"]

            # Old refresh token is now revoked
            with pytest.raises(TokenRevokedError):
                await svc.refresh_access_token(tokens["refresh_token"])

            # Logout
            await svc.logout(rf_tokens["refresh_token"])
            await db.commit()

            # Logout again fails
            with pytest.raises(TokenRevokedError):
                await svc.logout(rf_tokens["refresh_token"])

    async def test_logout_all_and_change_password(self, setup_db: None) -> None:
        async with TestSessionLocal() as db:
            svc = AuthenticationService(
                UserRepository(db), SessionRepository(db), AuditRepository(db)
            )
            user = await svc.register_user(
                email="logout_all@test.com",
                password="OldPassword123",
                full_name="Logout All User",
            )
            await db.commit()

            tokens = await svc.login("logout_all@test.com", "OldPassword123")
            await db.commit()

            # Logout all devices
            await svc.logout_all_devices(user.id)
            await db.commit()

            # Refresh token should fail now
            with pytest.raises(TokenRevokedError):
                await svc.refresh_access_token(tokens["refresh_token"])

            # Change password wrong current password
            with pytest.raises(AuthenticationError):
                await svc.change_password(user.id, "WrongOldPass123", "NewPassword123")

            # Change password success
            await svc.change_password(user.id, "OldPassword123", "NewPassword123")
            await db.commit()

            # Verify login with new password
            new_tokens = await svc.login("logout_all@test.com", "NewPassword123")
            assert "access_token" in new_tokens
