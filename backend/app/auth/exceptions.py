class AuthenticationError(Exception):
    """Raised for authentication related failures."""


class AuthorizationError(Exception):
    """Raised when a user lacks required permissions."""


class TokenExpiredError(Exception):
    """Raised when a JWT has expired."""


class TokenRevokedError(Exception):
    """Raised when a refresh token has been revoked or blacklisted."""


class SessionError(Exception):
    """Raised for session management errors."""


class PasswordPolicyError(Exception):
    """Raised when a password does not meet the security policy."""
