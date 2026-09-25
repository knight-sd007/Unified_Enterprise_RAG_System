"""
Pydantic Schemas and Data Models for FastAPI API.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Structured Error Models
# -----------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    """Standardized error payload detail."""
    code: str = Field(..., description="Machine-readable error classification code.")
    message: str = Field(..., description="Human-readable error explanation.")
    details: Optional[Any] = Field(None, description="Optional error context or field diagnostics.")


class ErrorResponse(BaseModel):
    """Standardized top-level API error response envelope."""
    error: ErrorDetail


# -----------------------------------------------------------------------------
# Authentication Models
# -----------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Application authentication login request."""
    access_key: str = Field(..., description="Application access key for authentication.", min_length=1)


class LoginResponse(BaseModel):
    """Successful login response."""
    authenticated: bool = Field(True, description="Indicates active authenticated session.")
    message: str = Field("Authentication successful.", description="Status message.")


class AuthStatusResponse(BaseModel):
    """Session authentication status check response."""
    authenticated: bool = Field(..., description="Whether current session is authenticated.")


class LogoutResponse(BaseModel):
    """Successful logout response."""
    authenticated: bool = Field(False, description="Indicates cleared session state.")
    message: str = Field("Logged out successfully.", description="Status message.")


# -----------------------------------------------------------------------------
# Provider Metadata Models
# -----------------------------------------------------------------------------

class ProviderMetadata(BaseModel):
    """Safe, non-sensitive metadata for an AI provider."""
    provider_id: str = Field(..., description="Canonical provider identifier.")
    name: str = Field(..., description="Human-readable provider display name.")
    configured: bool = Field(..., description="Whether necessary API credentials are configured.")
    chat_model: str = Field(..., description="Configured chat completion model.")
    embedding_model: str = Field(..., description="Configured vector embedding model.")
    dimension: int = Field(..., description="High-dimensional embedding vector dimension.")


class ProvidersListResponse(BaseModel):
    """Collection of available AI providers and their status."""
    providers: List[ProviderMetadata] = Field(..., description="List of supported AI providers.")


# -----------------------------------------------------------------------------
# Health Check Models
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Service health probe response."""
    status: str = Field("healthy", description="Operational status indicator.")
    service: str = Field("p06-enterprise-rag", description="Service identifier.")
    version: str = Field("1.0.0", description="Service semantic version.")
