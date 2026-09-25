"""
Authentication routes for application access key gate.
"""

import os
from fastapi import APIRouter, Request, Response
from api.dependencies import (
    APIError,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    create_session_token,
    get_current_session,
)
from api.schemas import (
    AuthStatusResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
)
from config.settings import Config
from utils.security import verify_access_key

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate with access key",
    description="Validates the application access key and issues an HTTP-only session cookie.",
)
async def login(req: LoginRequest, response: Response) -> LoginResponse:
    """Authenticates access key and sets secure session cookie."""
    expected_key = Config.get_app_access_key()

    if not verify_access_key(req.access_key, expected_key):
        raise APIError(
            status_code=401,
            code="AUTHENTICATION_FAILED",
            message="Invalid access key.",
        )

    session_token = create_session_token()
    is_secure = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1", "yes")

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
    )

    return LoginResponse(
        authenticated=True,
        message="Authentication successful.",
    )


@router.get(
    "/status",
    response_model=AuthStatusResponse,
    summary="Check session authentication status",
    description="Returns whether the caller has an active authenticated session.",
)
async def auth_status(request: Request) -> AuthStatusResponse:
    """Checks session validity from cookie or Authorization header."""
    is_auth = get_current_session(request)
    return AuthStatusResponse(authenticated=is_auth)


@router.post(
    "/logout",
    response_model=LogoutResponse,
    summary="Log out of current session",
    description="Clears the HTTP-only session cookie.",
)
async def logout(response: Response) -> LogoutResponse:
    """Invalidates the active session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
    )
    return LogoutResponse(
        authenticated=False,
        message="Logged out successfully.",
    )
