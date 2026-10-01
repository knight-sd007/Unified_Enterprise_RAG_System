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
    default_provider: str = Field(..., description="Configured default AI provider identifier.")


# -----------------------------------------------------------------------------
# Health Check Models
# -----------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Service health probe response."""
    status: str = Field("healthy", description="Operational status indicator.")
    service: str = Field("p06-enterprise-rag", description="Service identifier.")
    version: str = Field("1.0.0", description="Service semantic version.")


# -----------------------------------------------------------------------------
# Document Ingestion Models
# -----------------------------------------------------------------------------

class DocumentIngestResponse(BaseModel):
    """Document ingestion outcome response."""
    status: str = Field(..., description="Ingestion execution status ('success', 'warning', 'error').")
    document_count: int = Field(..., description="Total documents processed.")
    chunk_count: int = Field(..., description="Total semantic chunks indexed.")
    provider: str = Field(..., description="AI provider display name used for embeddings.")
    provider_id: str = Field(..., description="Canonical AI provider ID.")
    embedding_model: str = Field(..., description="Embedding model name used.")
    vector_dimension: Optional[int] = Field(None, description="Vector dimension of embeddings.")
    collection_name: Optional[str] = Field(None, description="Target Qdrant collection or in-memory descriptor.")
    message: Optional[str] = Field(None, description="Optional diagnostic or warning message.")
    filenames: List[str] = Field(default_factory=list, description="List of ingested document filenames.")


# -----------------------------------------------------------------------------
# RAG Query & Citation Models
# -----------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """RAG semantic search and question-answering request."""
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language question or search query.")
    provider_id: Optional[str] = Field(None, description="Optional provider identifier ('openai', 'gemini', 'nvidia_nim'). Defaults to configured system provider.")
    top_k: int = Field(4, ge=1, le=20, description="Number of top context chunks to retrieve.")
    similarity_threshold: float = Field(0.25, ge=0.0, le=1.0, description="Cosine similarity score cutoff threshold.")


class SourceItem(BaseModel):
    """Individual retrieved context chunk with citation metadata."""
    chunk_id: str = Field(..., description="Deterministic chunk identifier.")
    content: str = Field(..., description="Retrieved chunk text content.")
    score: float = Field(..., description="Cosine similarity relevance score.")
    metadata: dict = Field(default_factory=dict, description="Document metadata including filename and chunk index.")


class QueryResponse(BaseModel):
    """RAG question-answering answer and citation back-references."""
    answer: str = Field(..., description="Grounded natural language answer.")
    sources: List[SourceItem] = Field(default_factory=list, description="List of retrieved context reference citations.")
    provider: str = Field(..., description="AI provider display name used for generation.")
    provider_id: str = Field(..., description="Canonical AI provider ID.")
    chat_model: str = Field(..., description="Chat completion model name used.")
    retrieved_count: int = Field(0, description="Number of matching context chunks retrieved.")


# -----------------------------------------------------------------------------
# RAG Telemetry & Index Management Models
# -----------------------------------------------------------------------------

class RAGStatsResponse(BaseModel):
    """RAG pipeline vector store and provider telemetry."""
    provider_id: str = Field(..., description="Canonical AI provider ID.")
    provider_name: str = Field(..., description="AI provider display name.")
    count: int = Field(..., description="Number of vectors currently indexed in active store.")
    dimension: Any = Field(..., description="Embedding vector dimension.")
    store_type: str = Field(..., description="Vector store backend descriptor.")
    collection_name: Optional[str] = Field(None, description="Qdrant collection name if cloud-backed.")
    status: str = Field(..., description="Vector store operational status.")
    configured: bool = Field(..., description="Whether provider credentials are configured.")


class ClearIndexResponse(BaseModel):
    """Vector index clearing outcome response."""
    status: str = Field("success", description="Clear execution status.")
    message: str = Field(..., description="Human-readable outcome message.")
    provider_id: Optional[str] = Field(None, description="Provider ID whose index was cleared.")
