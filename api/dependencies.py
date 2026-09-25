"""
API Dependencies, Authentication, and Session Verification.
"""

import os
from typing import Any, Optional
from fastapi import Request
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from config.settings import Config

SESSION_COOKIE_NAME = "p06_session"
SESSION_MAX_AGE_SECONDS = 86400  # 24 hours
SESSION_SALT = "p06-rag-session-salt-v1"


class APIError(Exception):
    """Custom API Exception for structured error handling."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Any] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def _get_serializer() -> URLSafeTimedSerializer:
    """Returns timed serializer using application access key as signing secret."""
    secret = Config.get_app_access_key() or "p06-default-internal-secret"
    return URLSafeTimedSerializer(secret_key=secret, salt=SESSION_SALT)


def create_session_token() -> str:
    """Generates a cryptographically signed, timestamped session token."""
    serializer = _get_serializer()
    return serializer.dumps({"authenticated": True})


def verify_session_token(token: str) -> bool:
    """
    Validates signature and expiration of a session token.
    Returns True if valid and unexpired; False otherwise.
    """
    if not token or not isinstance(token, str):
        return False
    serializer = _get_serializer()
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        return isinstance(data, dict) and data.get("authenticated") is True
    except (BadSignature, SignatureExpired, Exception):
        return False


def extract_session_token(request: Request) -> Optional[str]:
    """
    Extracts session token from HTTP-only cookie or Authorization Bearer header.
    """
    # 1. Primary: Cookie
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_token:
        return cookie_token

    # 2. Secondary: Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return None


def get_current_session(request: Request) -> bool:
    """
    Returns True if the current request presents a valid session token, False otherwise.
    Does not raise an exception.
    """
    token = extract_session_token(request)
    if not token:
        return False
    return verify_session_token(token)


def require_authentication(request: Request) -> bool:
    """
    FastAPI dependency enforcing valid session authentication.
    Raises structured 401 APIError if unauthenticated.
    """
    if not get_current_session(request):
        raise APIError(
            status_code=401,
            code="UNAUTHORIZED",
            message="Authentication required.",
        )
    return True
