"""
API Dependencies, Authentication, and Session Verification.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Union
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


@dataclass
class UserSession:
    """Represents an authenticated user session."""
    user_id: str = "default_user"
    role: str = "user"
    auth_type: str = "google"
    authenticated: bool = True
    name: Optional[str] = None
    email: Optional[str] = None
    picture: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        """Returns True if the session possesses administrative privileges."""
        return self.role == "admin"

    @property
    def is_google_user(self) -> bool:
        """Returns True if the session originates from verified Google OAuth."""
        return self.auth_type == "google" and self.user_id.startswith("google_")


def _get_serializer() -> URLSafeTimedSerializer:
    """Returns timed serializer using dedicated SESSION_SIGNING_KEY as secret."""
    secret = Config.get_session_signing_key()
    if not secret:
        raise APIError(
            status_code=500,
            code="AUTH_CONFIGURATION_ERROR",
            message="Session signing key (SESSION_SIGNING_KEY) is not configured.",
        )
    return URLSafeTimedSerializer(secret_key=secret, salt=SESSION_SALT)


def create_session_token(
    user_id: str = "default_user",
    role: str = "user",
    auth_type: str = "google",
    name: Optional[str] = None,
    email: Optional[str] = None,
    picture: Optional[str] = None,
) -> str:
    """Generates a cryptographically signed, timestamped session token with identity and auth_type."""
    serializer = _get_serializer()
    payload = {
        "authenticated": True,
        "user_id": user_id,
        "role": role,
        "auth_type": auth_type,
    }
    if name:
        payload["name"] = name
    if email:
        payload["email"] = email
    if picture:
        payload["picture"] = picture
    return serializer.dumps(payload)


def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validates signature and expiration of a session token.
    Returns session dict payload if valid; None otherwise.
    """
    if not token or not isinstance(token, str):
        return None
    try:
        serializer = _get_serializer()
        data = serializer.loads(token, max_age=SESSION_MAX_AGE_SECONDS)
        if isinstance(data, dict) and data.get("authenticated") is True:
            if "user_id" not in data:
                data["user_id"] = "default_user"
            if "role" not in data:
                data["role"] = "user"
            if "auth_type" not in data:
                data["auth_type"] = "admin_key" if data.get("role") == "admin" and not data.get("user_id", "").startswith("google_") else "google"
            return data
        return None
    except (BadSignature, SignatureExpired, APIError, Exception):
        return None


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


def get_current_session(request: Request) -> Optional[UserSession]:
    """
    Returns UserSession if the current request presents a valid session token, None otherwise.
    Does not raise an exception.
    """
    token = extract_session_token(request)
    if not token:
        return None
    payload = verify_session_token(token)
    if not payload:
        return None
    return UserSession(
        user_id=payload.get("user_id", "default_user"),
        role=payload.get("role", "user"),
        auth_type=payload.get("auth_type", "google"),
        authenticated=True,
        name=payload.get("name"),
        email=payload.get("email"),
        picture=payload.get("picture"),
    )


def require_authentication(request: Request) -> UserSession:
    """
    FastAPI dependency enforcing valid session authentication.
    Returns authenticated UserSession or raises structured 401 APIError.
    """
    session = get_current_session(request)
    if not session:
        raise APIError(
            status_code=401,
            code="UNAUTHORIZED",
            message="Authentication required.",
        )
    return session


def require_admin(request: Request) -> UserSession:
    """
    FastAPI dependency enforcing administrative privileges.
    Returns authenticated UserSession or raises structured 403 APIError.
    """
    session = require_authentication(request)
    if not session.is_admin:
        raise APIError(
            status_code=403,
            code="FORBIDDEN",
            message="Administrative privileges required for this operation.",
        )
    return session


def require_workspace_access(request: Request) -> UserSession:
    """
    Enforces that normal workspace RAG and document operations require a Google-authenticated
    user with active Google Drive authorization. Break-glass admin-key sessions are prohibited.
    """
    session = require_authentication(request)

    # 1. Prohibit break-glass admin-key sessions from normal document workspace & RAG
    if session.auth_type == "admin_key" or not session.user_id.startswith("google_"):
        raise APIError(
            status_code=403,
            code="ADMIN_KEY_WORKSPACE_RESTRICTED",
            message=(
                "Administrator break-glass key session is restricted to administrative operations. "
                "Document workspace and RAG operations require Google login with Google Drive authorization."
            ),
        )

    # 2. Check that the user has authorized Google Drive
    from rag.storage.metadata_db import get_metadata_repo
    repo = get_metadata_repo()
    tokens = repo.get_oauth_tokens(session.user_id, "google")
    if not tokens or not (tokens.get("access_token") or tokens.get("refresh_token")):
        raise APIError(
            status_code=403,
            code="DRIVE_AUTHORIZATION_REQUIRED",
            message="Google Drive authorization is required to access the P06 workspace. Please sign in with Google and grant Google Drive permissions.",
        )

    return session


# -----------------------------------------------------------------------------
# RAG Pipeline Dependency Management
# -----------------------------------------------------------------------------

_rag_pipeline_instance: Optional[Any] = None


def get_rag_pipeline() -> Any:
    """
    FastAPI dependency returning singleton RAGPipeline instance.
    Enforces shared in-memory vector store lifecycle across requests.
    """
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        from rag.pipeline import RAGPipeline
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance


def set_rag_pipeline(pipeline: Optional[Any]) -> None:
    """
    Injects or resets the active RAGPipeline instance (used for test isolation).
    """
    global _rag_pipeline_instance
    _rag_pipeline_instance = pipeline
