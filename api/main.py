"""
Unified Enterprise RAG System — FastAPI Application.

Production-grade REST API service providing multi-provider AI abstraction,
health probes, session authentication, and protected OpenAPI documentation.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, Depends, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from api.dependencies import APIError, require_authentication
from api.routes import auth, health, providers
from utils.security import sanitize_error_message


# -----------------------------------------------------------------------------
# FastAPI Application Initialization
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Unified Enterprise RAG System API",
    version="1.0.0",
    description=(
        "Enterprise Retrieval-Augmented Generation REST API with unified "
        "AI provider abstraction (Google Gemini, NVIDIA NIM, OpenAI)."
    ),
    docs_url=None,       # Disabled public Swagger UI (protected via custom route)
    openapi_url=None,    # Disabled public OpenAPI schema (protected via custom route)
    redoc_url=None,      # Disabled ReDoc
)


# -----------------------------------------------------------------------------
# Security Headers Middleware
# -----------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Injects baseline HTTP security headers onto every response."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


app.add_middleware(SecurityHeadersMiddleware)


# -----------------------------------------------------------------------------
# CORS Middleware Configuration
# -----------------------------------------------------------------------------

allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
if allowed_origins_env:
    allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
else:
    # Explicit trusted origins for local frontend development (no wildcard)
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


# -----------------------------------------------------------------------------
# Structured Exception Handlers
# -----------------------------------------------------------------------------

@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handles controlled API errors with structured payload."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handles request schema validation failures."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Request validation failed.",
                "details": exc.errors(),
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Converts Starlette/FastAPI HTTPExceptions into standardized format."""
    code_map = {
        400: "INVALID_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
    }
    error_code = code_map.get(exc.status_code, "HTTP_ERROR")
    message = str(exc.detail) if exc.detail else "An HTTP error occurred."
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": None,
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled server errors and redacts sensitive information."""
    sanitized_msg = sanitize_error_message(exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": sanitized_msg or "An internal server error occurred.",
                "details": None,
            }
        },
    )


# -----------------------------------------------------------------------------
# API Version 1 Routers
# -----------------------------------------------------------------------------

app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(providers.router, prefix="/api/v1")


# -----------------------------------------------------------------------------
# Protected OpenAPI Schema and Swagger UI Routes
# -----------------------------------------------------------------------------

def _generate_custom_openapi():
    """Generates and caches OpenAPI schema for the application."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get(
    "/openapi.json",
    include_in_schema=False,
    summary="Protected OpenAPI schema",
)
async def get_protected_openapi(authenticated: bool = Depends(require_authentication)):
    """Returns OpenAPI schema only for authenticated sessions."""
    return JSONResponse(content=_generate_custom_openapi())


@app.get(
    "/docs",
    include_in_schema=False,
    summary="Protected Swagger UI",
)
async def get_protected_swagger_ui(authenticated: bool = Depends(require_authentication)):
    """Renders interactive Swagger UI documentation for authenticated sessions."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} — API Documentation",
    )


@app.get(
    "/api/v1/openapi.json",
    include_in_schema=False,
    summary="Protected OpenAPI schema (versioned alias)",
)
async def get_protected_openapi_v1(authenticated: bool = Depends(require_authentication)):
    """Versioned alias for protected OpenAPI schema."""
    return JSONResponse(content=_generate_custom_openapi())


@app.get(
    "/api/v1/docs",
    include_in_schema=False,
    summary="Protected Swagger UI (versioned alias)",
)
async def get_protected_swagger_ui_v1(authenticated: bool = Depends(require_authentication)):
    """Versioned alias for protected Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url="/api/v1/openapi.json",
        title=f"{app.title} — API Documentation",
    )
