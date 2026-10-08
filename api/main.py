"""
Unified Enterprise RAG System — FastAPI Application.

Production-grade REST API service providing multi-provider AI abstraction,
health probes, session authentication, and protected OpenAPI documentation.
"""

import os
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, Depends, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from api.dependencies import APIError, require_authentication
from api.routes import admin, auth, documents, health, providers, rag
from api.schemas import (
    DocumentDeleteResponse,
    DocumentIngestResponse,
    DocumentItem,
    DocumentListResponse,
    ErrorDetail,
    ErrorResponse,
    UserClearResponse,
)
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
    docs_url=None,       # Handled explicitly with OpenAPI customization
    openapi_url=None,    # Handled explicitly with OpenAPI customization
    redoc_url=None,      # Handled explicitly with OpenAPI customization
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
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )
    if "p06_oauth_state" in request.cookies:
        response.delete_cookie(key="p06_oauth_state", path="/")
    return response


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

app.include_router(health.router)
app.include_router(health.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(providers.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(admin.router)


# -----------------------------------------------------------------------------
# OpenAPI Schema, Swagger UI, and ReDoc Documentation Routes
# -----------------------------------------------------------------------------

def _generate_custom_openapi():
    """Generates and caches OpenAPI schema for the application with security schemes."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    components = openapi_schema.setdefault("components", {})
    schemas = components.setdefault("schemas", {})

    # Ensure Error models are registered in schemas
    if "ErrorDetail" not in schemas:
        schemas["ErrorDetail"] = ErrorDetail.model_json_schema()
    if "ErrorResponse" not in schemas:
        schemas["ErrorResponse"] = ErrorResponse.model_json_schema()

    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["CookieAuth"] = {
        "type": "apiKey",
        "in": "cookie",
        "name": "p06_session",
        "description": (
            "Cryptographically signed HttpOnly session cookie set upon successful login at /api/v1/auth/login. "
            "Used automatically by browser clients."
        ),
    }
    security_schemes["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "Session token passed in the Authorization header: `Authorization: Bearer <session_token>`."
        ),
    }

    # Annotate protected operational routes with security requirements and 401 error response
    paths = openapi_schema.get("paths", {})
    for path, path_item in paths.items():
        if (
            path.startswith("/api/v1/providers")
            or path.startswith("/api/v1/documents")
            or path.startswith("/api/v1/rag")
            or path.startswith("/api/v1/admin")
        ):
            for method, operation in path_item.items():
                if isinstance(operation, dict):
                    operation.setdefault("security", [{"CookieAuth": []}, {"BearerAuth": []}])
                    responses = operation.setdefault("responses", {})
                    if "401" not in responses:
                        responses["401"] = {
                            "description": "Unauthorized — Valid `p06_session` cookie or `Bearer` token required.",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                                }
                            }
                        }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


@app.get(
    "/openapi.json",
    include_in_schema=False,
    summary="OpenAPI Schema",
    dependencies=[Depends(require_authentication)],
)
async def get_openapi_schema():
    """Returns OpenAPI schema."""
    return JSONResponse(content=_generate_custom_openapi())


@app.get(
    "/docs",
    include_in_schema=False,
    summary="Swagger UI",
    dependencies=[Depends(require_authentication)],
)
async def get_swagger_ui():
    """Renders interactive Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} — API Documentation",
        swagger_ui_parameters={"withCredentials": True},
    )


@app.get(
    "/redoc",
    include_in_schema=False,
    summary="ReDoc Documentation",
    dependencies=[Depends(require_authentication)],
)
async def get_redoc():
    """Renders ReDoc API documentation."""
    return get_redoc_html(
        openapi_url="/openapi.json",
        title=f"{app.title} — ReDoc",
    )


@app.get(
    "/api/v1/openapi.json",
    include_in_schema=False,
    summary="OpenAPI Schema (versioned alias)",
    dependencies=[Depends(require_authentication)],
)
async def get_openapi_schema_v1():
    """Versioned alias for OpenAPI schema."""
    return JSONResponse(content=_generate_custom_openapi())


@app.get(
    "/api/v1/docs",
    include_in_schema=False,
    summary="Swagger UI (versioned alias)",
    dependencies=[Depends(require_authentication)],
)
async def get_swagger_ui_v1():
    """Versioned alias for Swagger UI documentation."""
    return get_swagger_ui_html(
        openapi_url="/api/v1/openapi.json",
        title=f"{app.title} — API Documentation",
        swagger_ui_parameters={"withCredentials": True},
    )


@app.get(
    "/api/v1/redoc",
    include_in_schema=False,
    summary="ReDoc Documentation (versioned alias)",
    dependencies=[Depends(require_authentication)],
)
async def get_redoc_v1():
    """Versioned alias for ReDoc documentation."""
    return get_redoc_html(
        openapi_url="/api/v1/openapi.json",
        title=f"{app.title} — ReDoc",
    )


# -----------------------------------------------------------------------------
# Frontend Static Asset Serving and SPA Routing
# -----------------------------------------------------------------------------

_FRONTEND_DIST_ENV = os.getenv("FRONTEND_DIST_DIR", "")
if _FRONTEND_DIST_ENV:
    FRONTEND_DIST_DIR = Path(_FRONTEND_DIST_ENV).resolve()
else:
    FRONTEND_DIST_DIR = (Path(__file__).resolve().parent.parent / "frontend" / "dist").resolve()

_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")


@app.get("/", include_in_schema=False)
async def serve_root():
    """Serves the built React SPA index.html or baseline status info."""
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return JSONResponse(
        status_code=200,
        content={
            "service": "Unified Enterprise RAG System API",
            "status": "ONLINE",
            "version": "1.0.0",
            "api_docs": "/docs",
            "health": "/api/v1/health",
        },
    )


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa_fallback(full_path: str):
    """
    Serves static files or falls back to index.html for client-side routing.
    Does NOT swallow /api/*, /docs, or /openapi.json routes.
    """
    # Guard against intercepting non-existent API or system routes
    if (
        full_path.startswith("api/")
        or full_path.startswith("docs")
        or full_path.startswith("openapi.json")
        or full_path.startswith("redoc")
    ):
        raise StarletteHTTPException(
            status_code=404,
            detail=f"Endpoint /{full_path} not found.",
        )

    # Check if a specific static file exists in dist (e.g. favicon.ico, vite.svg)
    file_path = FRONTEND_DIST_DIR / full_path
    if file_path.is_file():
        return FileResponse(str(file_path))

    # Fallback to SPA index.html
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))

    raise StarletteHTTPException(
        status_code=404,
        detail=f"Resource /{full_path} not found.",
    )
