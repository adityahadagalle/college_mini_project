# backend/app/middleware/security.py
"""Security middleware for the FastAPI application.

Features:
- Token‑bucket rate limiting (per IP for unauthenticated, per user for authenticated).
- Standard security response headers.
- Optional HTTPS enforcement.
- Simple trusted‑host validation placeholder.
- Audit logging of each request.
"""

import time
from collections import defaultdict
from typing import Awaitable, Callable, Dict, Tuple

from app.core.config import settings
from app.core.logging_config import logger
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# ---------------------------------------------------------------------------
# Token bucket implementation
# ---------------------------------------------------------------------------
class TokenBucket:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, amount: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


# ---------------------------------------------------------------------------
# Bucket registries
# ---------------------------------------------------------------------------
_ip_buckets: Dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(60, 1))
_user_buckets: Dict[str, TokenBucket] = defaultdict(lambda: TokenBucket(120, 2))

_ENDPOINT_LIMITS = {
    "/api/v1/auth/login": ("ip", 5, 60),
    "/api/v1/auth/register": ("ip", 5, 60),
    "/api/v1/predict": ("user", 100, 60),
    "/api/v1/explainability": ("user", 50, 60),
    "/api/v1/batch": ("user", 10, 60),
}


def _select_bucket(request: Request) -> Tuple[bool, TokenBucket]:
    path = request.url.path
    for prefix, (kind, cap, per) in _ENDPOINT_LIMITS.items():
        if path.startswith(prefix):
            if kind == "ip":
                ip = request.client.host if request.client else "unknown"
                bucket = _ip_buckets[ip]
                bucket.capacity = cap
                bucket.refill_rate = cap / per
                return True, bucket
            else:
                user_id = getattr(request.state, "user_id", None)
                if not user_id:
                    ip = request.client.host if request.client else "unknown"
                    bucket = _ip_buckets[ip]
                    bucket.capacity = cap
                    bucket.refill_rate = cap / per
                    return True, bucket
                bucket = _user_buckets[user_id]
                bucket.capacity = cap
                bucket.refill_rate = cap / per
                return True, bucket
    return False, TokenBucket(0, 0)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Applies rate limiting, security headers, HTTPS enforcement and logging."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # HTTPS enforcement for production
        if (
            getattr(settings, "ENVIRONMENT", "development") == "production"
            and request.url.scheme != "https"
        ):
            return Response(status_code=403, content="HTTPS required")

        limited, bucket = _select_bucket(request)
        if limited and not bucket.consume():
            return Response(status_code=429, content="Too Many Requests")

        # Audit log the request
        logger.info(
            "incoming request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None,
                "correlation_id": getattr(request.state, "correlation_id", None),
            },
        )

        response: Response = await call_next(request)

        # Add security headers
        security_headers = {
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'",
        }
        for k, v in security_headers.items():
            response.headers[k] = v

        return response


# End of security middleware
