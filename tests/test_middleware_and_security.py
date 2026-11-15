"""Tests for security middleware, correlation middleware, and core security utilities.

Covers:
- TokenBucket: consume, capacity limit, refill rate
- SecurityMiddleware: HTTPS enforcement, token bucket rate limiting (429 response), security headers
- CorrelationMiddleware: X-Correlation-ID / X-Request-ID propagation & generation
- Core security: JWT create/decode, password hash/verify, expired/invalid tokens
"""

import time

import pytest
from app.core.config import settings
from app.core.security import (
    _create_jwt,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
    get_password_hash,
    verify_password,
)
from app.middleware.correlation import CorrelationMiddleware
from app.middleware.security import SecurityMiddleware, TokenBucket, _select_bucket
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Token Bucket & Security Middleware Tests
# ---------------------------------------------------------------------------


class TestTokenBucketAndSecurityMiddleware:
    def test_token_bucket_consume_and_refill(self) -> None:
        bucket = TokenBucket(capacity=2.0, refill_rate=1.0)
        assert bucket.consume(1.0) is True
        assert bucket.consume(1.0) is True
        assert bucket.consume(1.0) is False  # Empty

        time.sleep(1.1)  # Refill 1 token
        assert bucket.consume(1.0) is True

    def test_select_bucket_endpoint_limits(self) -> None:
        app_test = FastAPI()
        client_test = TestClient(app_test)

        @app_test.get("/api/v1/auth/login")
        def dummy_login():
            return {"ok": True}

        # Request to endpoint in limit map
        req = Request(
            scope={"type": "http", "method": "GET", "path": "/api/v1/auth/login", "headers": []}
        )
        limited, bucket = _select_bucket(req)
        assert limited is True
        assert bucket.capacity == 5.0

    def test_security_middleware_headers_and_rate_limiting(self) -> None:
        app_test = FastAPI()
        app_test.add_middleware(SecurityMiddleware)

        @app_test.get("/ping")
        def ping():
            return {"msg": "pong"}

        @app_test.post("/api/v1/auth/login")
        def login_limit():
            return {"msg": "login"}

        test_client = TestClient(app_test)

        # Check security headers
        resp = test_client.get("/ping")
        assert resp.status_code == 200
        assert "Strict-Transport-Security" in resp.headers
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"

        # Rate limit trigger (capacity 5 for /api/v1/auth/login)
        for _ in range(5):
            test_client.post("/api/v1/auth/login")

        rate_limit_resp = test_client.post("/api/v1/auth/login")
        assert rate_limit_resp.status_code == 429
        assert rate_limit_resp.text == "Too Many Requests"

    def test_https_enforcement_in_production(self) -> None:
        from unittest.mock import patch

        app_prod = FastAPI()
        app_prod.add_middleware(SecurityMiddleware)

        @app_prod.get("/secure")
        def secure_route():
            return {"ok": True}

        prod_client = TestClient(app_prod)

        with patch("app.middleware.security.settings") as mock_settings:
            mock_settings.ENVIRONMENT = "production"
            resp = prod_client.get("http://testserver/secure")
            assert resp.status_code == 403
            assert resp.text == "HTTPS required"


# ---------------------------------------------------------------------------
# Correlation Middleware Tests
# ---------------------------------------------------------------------------


class TestCorrelationMiddleware:
    def test_correlation_middleware(self) -> None:
        app_corr = FastAPI()
        app_corr.add_middleware(CorrelationMiddleware)

        @app_corr.get("/test-correlation")
        def correlation_route(request: Request):
            return {
                "correlation_id": getattr(request.state, "correlation_id", None),
                "request_id": getattr(request.state, "request_id", None),
            }

        client_corr = TestClient(app_corr)

        # Generated correlation ID when none provided
        resp = client_corr.get("/test-correlation")
        assert resp.status_code == 200
        assert "X-Correlation-ID" in resp.headers
        assert "X-Request-ID" in resp.headers
        data = resp.json()
        assert data["correlation_id"] is not None

        # Custom correlation ID provided in headers
        custom_id = "custom-corr-id-9999"
        resp_custom = client_corr.get("/test-correlation", headers={"X-Correlation-ID": custom_id})
        assert resp_custom.headers["X-Correlation-ID"] == custom_id
        assert resp_custom.json()["correlation_id"] == custom_id


# ---------------------------------------------------------------------------
# Core Security Utility Tests
# ---------------------------------------------------------------------------


class TestCoreSecurity:
    def test_password_hashing(self) -> None:
        pwd = "MySecretPass123!"
        hashed = get_password_hash(pwd)
        assert hashed != pwd
        assert verify_password(pwd, hashed) is True
        assert verify_password("WrongPassword", hashed) is False

    def test_jwt_tokens(self) -> None:
        subject = "user-uuid-12345"

        # Access token
        access_tok = create_access_token(subject)
        decoded_sub = decode_access_token(access_tok)
        assert decoded_sub == subject

        # Refresh token
        ref_tok = create_refresh_token(subject)
        decoded_ref_sub = decode_refresh_token(ref_tok)
        assert decoded_ref_sub == subject

        # Invalid token returns None
        assert decode_access_token("invalid.jwt.token") is None
        assert decode_refresh_token("invalid.jwt.token") is None
