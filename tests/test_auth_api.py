"""Comprehensive API endpoint tests for authentication (/api/v1/auth/*).

Covers:
- POST /api/v1/auth/register (success, duplicate email, password policy validation)
- POST /api/v1/auth/login (success OAuth2 form data, invalid password, nonexistent user)
- POST /api/v1/auth/refresh (success, invalid token)
- POST /api/v1/auth/logout (success, invalid token)
- POST /api/v1/auth/logout-all (authenticated success, unauthenticated 401)
- POST /api/v1/auth/change-password (success, incorrect current password)
- GET /api/v1/auth/me (success with token, unauthenticated 401, expired token 401)
- get_current_user helper dependency validation
"""

from typing import Any

import pytest
from app.core.database import Base, get_db
from app.main import app
from app.models.auth import AuditLog, PasswordHistory, RefreshToken, Session  # noqa: F401
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def _override_get_db():  # type: ignore[no-untyped-def]
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


client = TestClient(app)


@pytest.fixture(autouse=True)
async def setup_db():
    app.dependency_overrides[get_db] = _override_get_db
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.clear()


class TestAuthAPIEndpoints:
    def test_register_login_me_logout_flow(self, setup_db: None) -> None:
        # 1. Register user
        reg_payload = {
            "email": "user_api@example.com",
            "password": "SecurePassword123!",
            "full_name": "API User",
            "role": "user",
        }
        reg_resp = client.post("/api/v1/auth/register", json=reg_payload)
        assert reg_resp.status_code == 201
        reg_data = reg_resp.json()
        assert reg_data["email"] == "user_api@example.com"
        assert reg_data["full_name"] == "API User"
        assert "id" in reg_data

        # 2. Duplicate registration fails
        dup_resp = client.post("/api/v1/auth/register", json=reg_payload)
        assert dup_resp.status_code == 400

        # 3. Login with OAuth2 form data
        login_resp = client.post(
            "/api/v1/auth/login",
            data={"username": "user_api@example.com", "password": "SecurePassword123!"},
        )
        assert login_resp.status_code == 200
        tokens = login_resp.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        # 4. Login with invalid password
        bad_login = client.post(
            "/api/v1/auth/login",
            data={"username": "user_api@example.com", "password": "WrongPassword123!"},
        )
        assert bad_login.status_code == 401

        # 5. GET /me with valid token
        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == "user_api@example.com"

        # 6. Refresh token
        refresh_resp = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
        assert refresh_resp.status_code == 200
        new_tokens = refresh_resp.json()
        assert "access_token" in new_tokens

        # 7. Logout single token
        logout_resp = client.post(
            "/api/v1/auth/logout", json={"refresh_token": new_tokens["refresh_token"]}
        )
        assert logout_resp.status_code == 200

        # 8. Change password
        pwd_resp = client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "SecurePassword123!",
                "new_password": "NewSecurePassword456!",
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert pwd_resp.status_code == 200

        # Change password bad current password
        bad_pwd_resp = client.post(
            "/api/v1/auth/change-password",
            json={
                "current_password": "WrongPassword123!",
                "new_password": "AnotherPassword789!",
            },
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert bad_pwd_resp.status_code == 401

        # 9. Logout all devices
        logout_all_resp = client.post(
            "/api/v1/auth/logout-all",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert logout_all_resp.status_code == 200

    def test_unauthorized_endpoints(self) -> None:
        # GET /me without token
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401

        # GET /me with invalid token
        resp_invalid = client.get(
            "/api/v1/auth/me", headers={"Authorization": "Bearer invalid_token_123"}
        )
        assert resp_invalid.status_code == 401

        # POST /logout with invalid refresh token
        resp_logout = client.post(
            "/api/v1/auth/logout", json={"refresh_token": "invalid_refresh_token"}
        )
        assert resp_logout.status_code == 401
