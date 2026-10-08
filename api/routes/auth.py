import hmac
import os
import re
import secrets
from urllib.parse import quote
from typing import Optional
import httpx
from fastapi import APIRouter, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from api.dependencies import (
    APIError,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE_SECONDS,
    create_session_token,
    get_current_session,
)
from api.schemas import (
    AuthStatusResponse,
    GoogleAuthConfigResponse,
    GoogleAuthUrlResponse,
    GoogleOAuthCallbackRequest,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
)
from config.settings import Config
from rag.storage.metadata_db import get_metadata_repo
from utils.logging import logger
from utils.security import verify_access_key

router = APIRouter(prefix="/auth", tags=["Authentication"])

OAUTH_STATE_COOKIE_NAME = "p06_oauth_state"
OAUTH_STATE_MAX_AGE_SECONDS = 600  # 10 minutes
OAUTH_STATE_SALT = "p06-oauth-state-v1"


def _get_oauth_state_serializer() -> URLSafeTimedSerializer:
    """Returns timed serializer using dedicated SESSION_SIGNING_KEY."""
    secret = Config.get_session_signing_key()
    if not secret:
        raise APIError(
            status_code=500,
            code="AUTH_CONFIGURATION_ERROR",
            message="Session signing key (SESSION_SIGNING_KEY) is not configured.",
        )
    return URLSafeTimedSerializer(secret_key=secret, salt=OAUTH_STATE_SALT)


def create_oauth_state_token(state: str) -> str:
    """Signs an OAuth state token with timestamp and HMAC."""
    return _get_oauth_state_serializer().dumps(state)


def verify_oauth_state_token(signed_token: str, max_age: int = OAUTH_STATE_MAX_AGE_SECONDS) -> Optional[str]:
    """Verifies signature and expiration of an OAuth state token."""
    if not signed_token or not isinstance(signed_token, str):
        return None
    try:
        data = _get_oauth_state_serializer().loads(signed_token, max_age=max_age)
        return str(data) if data else None
    except (BadSignature, SignatureExpired, Exception):
        return None


def _sanitize_username(username: str) -> str:
    """Sanitizes username into alphanumeric safe identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_\-\.]", "", username.strip().lower())
    return cleaned[:64] if cleaned else "default_user"


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Authenticate with administrator break-glass key",
    description="Validates the administrator access key for emergency break-glass console access. Normal users must authenticate using Google OAuth.",
)
async def login(req: LoginRequest, response: Response) -> LoginResponse:
    """Authenticates administrator break-glass access key and sets secure session cookie."""
    admin_key = Config.get_admin_access_key()

    if not admin_key:
        raise APIError(
            status_code=500,
            code="AUTH_CONFIGURATION_ERROR",
            message="Administrator access key (ADMIN_ACCESS_KEY) is not configured.",
        )

    # Validate administrator access key strictly
    is_admin = bool(verify_access_key(req.access_key, admin_key))
    if not is_admin:
        raise APIError(
            status_code=401,
            code="AUTHENTICATION_FAILED",
            message="Invalid administrator access key. Normal users must sign in with Google.",
        )

    user_id = "admin"
    role = "admin"
    auth_type = "admin_key"

    session_token = create_session_token(user_id=user_id, role=role, auth_type=auth_type, name="admin")
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
        user_id=user_id,
        role=role,
        auth_type=auth_type,
        drive_authorized=False,
        name="admin",
        message="Administrator break-glass authentication successful.",
    )


@router.get(
    "/google/config",
    response_model=GoogleAuthConfigResponse,
    summary="Google OAuth Configuration",
    description="Returns public client configuration for Google OAuth integration.",
)
async def get_google_auth_config() -> GoogleAuthConfigResponse:
    """Returns Google OAuth client setup status."""
    configured = Config.is_google_oauth_configured()
    return GoogleAuthConfigResponse(
        configured=configured,
        client_id=Config.get_google_client_id() if configured else None,
        redirect_uri=Config.get_google_redirect_uri() if configured else None,
    )


@router.get(
    "/google/url",
    response_model=GoogleAuthUrlResponse,
    summary="Generate Google OAuth Authorization URL",
    description="Generates an OAuth authorization URL with a cryptographically signed anti-CSRF state token and sets a secure HttpOnly cookie.",
)
async def get_google_auth_url(response: Response) -> GoogleAuthUrlResponse:
    """Generates Google OAuth URL and binds cryptographic anti-CSRF state in a secure cookie."""
    if not Config.is_google_oauth_configured():
        raise APIError(
            status_code=400,
            code="OAUTH_NOT_CONFIGURED",
            message="Google OAuth is not configured on this server.",
        )

    client_id = Config.get_google_client_id()
    redirect_uri = Config.get_google_redirect_uri()
    raw_state = secrets.token_urlsafe(32)
    signed_state = create_oauth_state_token(raw_state)

    is_secure = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1", "yes")
    response.set_cookie(
        key=OAUTH_STATE_COOKIE_NAME,
        value=signed_state,
        max_age=OAUTH_STATE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=is_secure,
        path="/",
    )

    scope = "openid email profile https://www.googleapis.com/auth/drive.file"
    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={quote(client_id)}"
        f"&redirect_uri={quote(redirect_uri)}"
        f"&response_type=code"
        f"&scope={quote(scope)}"
        f"&state={quote(raw_state)}"
        f"&access_type=offline"
        f"&prompt=consent"
    )

    return GoogleAuthUrlResponse(url=auth_url, state=raw_state)


async def _process_google_oauth_exchange(
    request: Request,
    code: str,
    state: Optional[str],
    response: Response,
) -> LoginResponse:
    """Shared internal helper for exchanging Google authorization code, validating anti-CSRF state,
    persisting tokens, and provisioning authenticated session.
    """
    if not Config.is_google_oauth_configured():
        raise APIError(
            status_code=400,
            code="OAUTH_NOT_CONFIGURED",
            message="Google OAuth is not configured on this server.",
        )

    # 1. Anti-CSRF server-bound state validation
    state_cookie = request.cookies.get(OAUTH_STATE_COOKIE_NAME)
    if not state_cookie or not state:
        response.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")
        raise APIError(
            status_code=400,
            code="INVALID_OAUTH_STATE",
            message="OAuth anti-CSRF state token is missing.",
        )

    verified_raw_state = verify_oauth_state_token(state_cookie)
    if not verified_raw_state:
        response.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")
        raise APIError(
            status_code=400,
            code="EXPIRED_OAUTH_STATE",
            message="OAuth anti-CSRF state token is invalid or expired.",
        )

    if not hmac.compare_digest(verified_raw_state, state):
        response.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")
        raise APIError(
            status_code=400,
            code="OAUTH_STATE_MISMATCH",
            message="OAuth anti-CSRF state token verification failed.",
        )

    # Invalidate OAuth state cookie immediately after verification to prevent replay
    response.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")

    client_id = Config.get_google_client_id()
    client_secret = Config.get_google_client_secret()
    redirect_uri = Config.get_google_redirect_uri()

    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            # 2. Exchange auth code for tokens
            token_resp = await http_client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code",
                },
            )

            if token_resp.status_code != 200:
                logger.error(f"Google token exchange failed: {token_resp.text}")
                raise APIError(
                    status_code=401,
                    code="OAUTH_EXCHANGE_FAILED",
                    message="Failed to exchange authorization code with Google.",
                )

            token_data = token_resp.json()
            access_token = token_data.get("access_token")

            # 3. Retrieve verified user identity
            userinfo_resp = await http_client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            if userinfo_resp.status_code != 200:
                logger.error(f"Google userinfo lookup failed: {userinfo_resp.text}")
                raise APIError(
                    status_code=401,
                    code="OAUTH_USERINFO_FAILED",
                    message="Failed to retrieve user profile from Google.",
                )

            userinfo = userinfo_resp.json()
            sub = userinfo.get("sub")
            email = userinfo.get("email", "").lower()
            email_verified = userinfo.get("email_verified") is True
            raw_name = (userinfo.get("name") or "").strip()
            display_name = raw_name if raw_name else "Google User"
            picture = userinfo.get("picture") or None

            if not sub:
                raise APIError(
                    status_code=400,
                    code="INVALID_OAUTH_PAYLOAD",
                    message="Google identity sub claim is missing.",
                )

            # Canonical multi-user identifier for OAuth
            user_id = f"google_{sub}"

            # 4. Check role escalation allowlist with strictly verified email and dedicated subject-ID allowlist
            admin_emails = [e.strip().lower() for e in Config.get_google_admin_emails() if e.strip()]
            admin_subs = [s.strip() for s in Config.get_google_admin_subs() if s.strip()]

            is_admin_by_email = bool(email_verified and email and email in admin_emails)
            is_admin_by_sub = bool(sub and sub in admin_subs)
            is_admin = is_admin_by_email or is_admin_by_sub
            role = "admin" if is_admin else "user"

            # 5. Save encrypted tokens for Drive storage integration
            repo = get_metadata_repo()
            repo.save_oauth_tokens(user_id, "google", token_data)

            # 6. Issue session cookie
            session_token = create_session_token(
                user_id=user_id,
                role=role,
                auth_type="google",
                name=display_name,
                email=email,
                picture=picture,
            )
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

            logger.info(f"Google OAuth login success for user '{user_id}' (display_name: '{display_name}', email: '{email}', email_verified: {email_verified}, role: '{role}')")

            return LoginResponse(
                authenticated=True,
                user_id=user_id,
                role=role,
                auth_type="google",
                drive_authorized=True,
                name=display_name,
                email=email,
                picture=picture,
                message=f"Google authentication successful for {display_name or email or user_id}.",
            )

    except APIError:
        response.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")
        raise
    except Exception as e:
        response.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")
        logger.error(f"Unexpected OAuth callback error: {e}")
        raise APIError(
            status_code=500,
            code="OAUTH_INTERNAL_ERROR",
            message="An error occurred while completing Google authentication.",
        )


@router.get(
    "/google/callback",
    summary="Google OAuth Browser Callback",
    description="Handles browser redirect from Google OAuth, exchanges authorization code for tokens, sets session cookie, and redirects user to application root.",
    response_class=RedirectResponse,
    status_code=status.HTTP_303_SEE_OTHER,
)
async def google_oauth_callback_get(
    request: Request,
    code: Optional[str] = Query(None, description="Google OAuth authorization code"),
    state: Optional[str] = Query(None, description="OAuth anti-CSRF state token"),
    error: Optional[str] = Query(None, description="Google OAuth error code if authorization failed"),
    error_description: Optional[str] = Query(None, description="Description of authorization error"),
) -> RedirectResponse:
    """Handles browser redirect from Google OAuth, validates CSRF state, and redirects to SPA with session."""
    # Handle Google OAuth errors safely without token exchange or open redirect
    if error:
        safe_error = re.sub(r"[^a-zA-Z0-9_\-]", "", error)[:64] or "oauth_error"
        logger.warning(f"Google OAuth callback received error '{safe_error}': {error_description or ''}")
        redirect_res = RedirectResponse(url=f"/?error={safe_error}", status_code=status.HTTP_303_SEE_OTHER)
        redirect_res.delete_cookie(key=OAUTH_STATE_COOKIE_NAME, path="/")
        return redirect_res

    if not code or not code.strip():
        raise APIError(
            status_code=400,
            code="INVALID_OAUTH_CODE",
            message="OAuth authorization code is missing.",
        )

    redirect_res = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    await _process_google_oauth_exchange(
        request=request,
        code=code.strip(),
        state=state,
        response=redirect_res,
    )
    return redirect_res


@router.post(
    "/google/callback",
    response_model=LoginResponse,
    summary="Google OAuth Callback & Token Exchange",
    description="Exchanges Google authorization code for tokens, validates server-bound anti-CSRF state, verifies identity and email status, and issues a session cookie.",
)
async def google_oauth_callback(
    request: Request, payload: GoogleOAuthCallbackRequest, response: Response
) -> LoginResponse:
    """Exchanges Google authorization code, validates CSRF state, and provisions authenticated session."""
    return await _process_google_oauth_exchange(
        request=request,
        code=payload.code,
        state=payload.state,
        response=response,
    )


@router.get(
    "/status",
    response_model=AuthStatusResponse,
    summary="Check session authentication and Google Drive authorization status",
    description="Returns whether the caller has an active authenticated session, user identity, and valid Drive authorization.",
)
async def auth_status(request: Request) -> AuthStatusResponse:
    """Checks session validity and identity from cookie or Authorization header."""
    session = get_current_session(request)
    if not session:
        return AuthStatusResponse(
            authenticated=False,
            user_id=None,
            role=None,
            auth_type=None,
            drive_authorized=False,
        )

    drive_authorized = False
    if session.user_id.startswith("google_"):
        repo = get_metadata_repo()
        tokens = repo.get_oauth_tokens(session.user_id, "google")
        drive_authorized = bool(tokens and (tokens.get("access_token") or tokens.get("refresh_token")))

    display_name = session.name
    if not display_name and session.is_google_user:
        display_name = "Google User"

    return AuthStatusResponse(
        authenticated=True,
        user_id=session.user_id,
        role=session.role,
        auth_type=session.auth_type,
        drive_authorized=drive_authorized,
        name=display_name,
        email=session.email,
        picture=session.picture,
    )


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
