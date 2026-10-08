"""
Pydantic Schemas and Data Models for FastAPI API.
"""

from typing import Any, Dict, List, Optional
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
    username: Optional[str] = Field(None, description="Optional user identifier or username for multi-user session scoping.", max_length=64)


class LoginResponse(BaseModel):
    """Successful login response."""
    authenticated: bool = Field(True, description="Indicates active authenticated session.")
    user_id: str = Field("default_user", description="Authenticated user identifier.")
    role: str = Field("user", description="Session role ('user' or 'admin').")
    auth_type: Optional[str] = Field("google", description="Authentication mechanism ('google' or 'admin_key').")
    drive_authorized: bool = Field(False, description="Whether Google Drive authorization is active.")
    message: str = Field("Authentication successful.", description="Status message.")
    name: Optional[str] = Field(None, description="User display name from Google OAuth profile.")
    email: Optional[str] = Field(None, description="User email from Google OAuth profile.")
    picture: Optional[str] = Field(None, description="User avatar image URL from Google OAuth profile.")


class AuthStatusResponse(BaseModel):
    """Session authentication status check response."""
    authenticated: bool = Field(..., description="Whether current session is authenticated.")
    user_id: Optional[str] = Field(None, description="Authenticated user identifier.")
    role: Optional[str] = Field(None, description="Session role ('user' or 'admin').")
    auth_type: Optional[str] = Field(None, description="Authentication mechanism ('google' or 'admin_key').")
    drive_authorized: bool = Field(False, description="Whether Google Drive authorization is active.")
    name: Optional[str] = Field(None, description="User display name from Google OAuth profile.")
    email: Optional[str] = Field(None, description="User email from Google OAuth profile.")
    picture: Optional[str] = Field(None, description="User avatar image URL from Google OAuth profile.")


class LogoutResponse(BaseModel):
    """Successful logout response."""
    authenticated: bool = Field(False, description="Indicates cleared session state.")
    message: str = Field("Logged out successfully.", description="Status message.")


class GoogleAuthConfigResponse(BaseModel):
    """Google OAuth configuration status for frontend."""
    configured: bool = Field(..., description="Whether Google OAuth credentials are set.")
    client_id: Optional[str] = Field(None, description="Public Google OAuth client ID.")
    redirect_uri: Optional[str] = Field(None, description="OAuth redirect URI.")


class GoogleAuthUrlResponse(BaseModel):
    """Google OAuth initiation authorization URL and anti-CSRF state token."""
    url: str = Field(..., description="Google OAuth authorization URL with signed anti-CSRF state token.")
    state: str = Field(..., description="Cryptographic anti-CSRF state token.")


class GoogleOAuthCallbackRequest(BaseModel):
    """Payload sent by client when exchanging OAuth authorization code."""
    code: str = Field(..., description="Authorization code returned by Google OAuth.", min_length=1)
    state: Optional[str] = Field(None, description="Anti-CSRF state token.")


# -----------------------------------------------------------------------------
# Provider Metadata Models
# -----------------------------------------------------------------------------

class ProviderMetadata(BaseModel):
    """Safe, non-sensitive metadata for an AI provider."""
    provider_id: str = Field(..., description="Canonical provider identifier.")
    name: str = Field(..., description="Human-readable provider display name.")
    configured: bool = Field(..., description="Whether necessary API credentials are configured.")
    chat_model: str = Field(..., description="Configured default chat completion model.")
    embedding_model: str = Field(..., description="Configured default vector embedding model.")
    dimension: int = Field(..., description="High-dimensional embedding vector dimension.")
    supported_chat_models: List[str] = Field(default_factory=list, description="Supported chat completion models.")
    supported_embedding_models: List[Dict[str, Any]] = Field(default_factory=list, description="Supported embedding models.")
    connectivity_status: Optional[str] = Field(None, description="Live connectivity status ('connected', 'not_configured', 'unreachable').")


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


class ProviderHealthItem(BaseModel):
    """Health and live connectivity probe for a single AI provider."""
    provider_id: str = Field(..., description="Provider identifier.")
    name: str = Field(..., description="Display name.")
    configured: bool = Field(..., description="Whether credentials are set.")
    status: str = Field(..., description="Health status ('connected', 'not_configured', 'unreachable').")
    message: str = Field(..., description="Diagnostic message.")
    latency_ms: Optional[float] = Field(None, description="Round-trip latency in milliseconds if tested.")


class ProvidersHealthResponse(BaseModel):
    """Comprehensive provider health diagnostics."""
    status: str = Field("healthy", description="Overall health indicator.")
    providers: List[ProviderHealthItem] = Field(..., description="Health details for each provider.")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp.")


# -----------------------------------------------------------------------------
# Document Ingestion & Lifecycle Models
# -----------------------------------------------------------------------------

class DocumentItem(BaseModel):
    """Metadata summary of an ingested document."""
    doc_id: str = Field(..., description="Unique document identifier.")
    filename: str = Field(..., description="Original document filename.")
    owner_id: str = Field(..., description="Authenticated owner user identifier.")
    provider_id: str = Field(..., description="AI provider used for embedding generation.")
    embedding_model: Optional[str] = Field(None, description="Embedding model name used.")
    chunk_count: int = Field(..., description="Number of vector chunks associated with document.")
    created_at: str = Field(..., description="ISO 8601 upload timestamp.")
    char_count: Optional[int] = Field(None, description="Extracted character count.")
    file_size: Optional[int] = Field(None, description="Original file size in bytes.")
    drive_file_id: Optional[str] = Field(None, description="Google Drive file ID if synced.")
    status: Optional[str] = Field("indexed", description="Document index status.")


class DocumentListResponse(BaseModel):
    """List of documents owned by the authenticated user."""
    documents: List[DocumentItem] = Field(default_factory=list, description="List of active documents.")
    total_documents: int = Field(..., description="Total documents count.")
    owner_id: str = Field(..., description="Scope owner identifier.")
    user_id: Optional[str] = Field(None, description="Alias for owner identifier.")


class DocumentDeleteResponse(BaseModel):
    """Document deletion outcome response."""
    status: str = Field("success", description="Deletion execution status.")
    message: str = Field(..., description="Outcome message.")
    doc_id: str = Field(..., description="Deleted document ID.")
    deleted_chunks: int = Field(..., description="Count of purged vector chunks.")


class UserClearResponse(BaseModel):
    """User-scoped document clear outcome response."""
    status: str = Field("success", description="Clear execution status.")
    message: str = Field(..., description="Outcome message.")
    owner_id: str = Field(..., description="Owner ID whose documents were purged.")
    user_id: Optional[str] = Field(None, description="Alias for owner identifier.")
    deleted_documents: int = Field(..., description="Count of purged documents.")
    deleted_chunks: int = Field(..., description="Count of purged vector chunks.")


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
    doc_ids: List[str] = Field(default_factory=list, description="List of assigned unique document IDs.")


# -----------------------------------------------------------------------------
# RAG Query & Citation Models
# -----------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """RAG semantic search and question-answering request."""
    query: str = Field(..., min_length=1, max_length=4000, description="Natural language question or search query.")
    provider_id: Optional[str] = Field(None, description="Optional legacy provider identifier ('openai', 'gemini', 'nvidia_nim').")
    chat_provider_id: Optional[str] = Field(None, description="Dedicated chat model provider identifier.")
    chat_model: Optional[str] = Field(None, description="Specific chat model name.")
    embedding_provider_id: Optional[str] = Field(None, description="Dedicated embedding model provider identifier.")
    embedding_model: Optional[str] = Field(None, description="Specific embedding model name.")
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
    provider: str = Field(..., description="Primary AI provider display name.")
    provider_id: str = Field(..., description="Primary AI provider ID.")
    chat_model: str = Field(..., description="Chat completion model name used.")
    chat_provider: Optional[str] = Field(None, description="Chat generation provider display name.")
    chat_provider_id: Optional[str] = Field(None, description="Chat generation provider ID.")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider display name.")
    embedding_provider_id: Optional[str] = Field(None, description="Embedding provider ID.")
    embedding_model: Optional[str] = Field(None, description="Embedding model name used.")
    retrieved_count: int = Field(0, description="Number of matching context chunks retrieved.")


# -----------------------------------------------------------------------------
# RAG Telemetry & Index Management Models
# -----------------------------------------------------------------------------

class RAGStatsResponse(BaseModel):
    """RAG pipeline vector store and provider telemetry."""
    provider_id: str = Field(..., description="Canonical AI provider ID.")
    provider_name: str = Field(..., description="AI provider display name.")
    count: int = Field(..., description="Number of vectors currently indexed in active store for the authenticated user.")
    dimension: Any = Field(..., description="Embedding vector dimension.")
    store_type: str = Field(..., description="Vector store backend descriptor.")
    collection_name: Optional[str] = Field(None, description="Qdrant collection name if cloud-backed.")
    status: str = Field(..., description="Vector store operational status.")
    configured: bool = Field(..., description="Whether provider credentials are configured.")
    user_id: Optional[str] = Field(None, description="User scope for the telemetry count.")


class ClearIndexResponse(BaseModel):
    """Vector index clearing outcome response (Administrative)."""
    status: str = Field("success", description="Clear execution status.")
    message: str = Field(..., description="Human-readable outcome message.")
    provider_id: Optional[str] = Field(None, description="Provider ID whose index was cleared.")


# -----------------------------------------------------------------------------
# Admin Console Models
# -----------------------------------------------------------------------------

class AdminOverviewResponse(BaseModel):
    """Aggregated administrative metrics."""
    total_documents: int = Field(..., description="Total indexed documents across all users.")
    total_chunks: int = Field(..., description="Total vector chunks in database.")
    total_bytes: int = Field(..., description="Total bytes of documents stored.")
    total_users: int = Field(..., description="Total distinct active users.")
    vector_store_backend: str = Field(..., description="Vector store backend descriptor.")
    providers: List[Dict[str, Any]] = Field(default_factory=list, description="Provider storage distribution breakdown.")


class AdminDiagnosticsResponse(BaseModel):
    """Admin diagnostic information (safe, non-sensitive)."""
    app_status: str = Field(..., description="Application status.")
    environment: str = Field(..., description="Runtime environment.")
    qdrant_configured: bool = Field(..., description="Whether Qdrant is configured.")
    qdrant_host: Optional[str] = Field(None, description="Configured Qdrant host.")
    oauth_configured: bool = Field(..., description="Whether Google OAuth is configured.")
    oauth_admin_emails: List[str] = Field(default_factory=list, description="Configured Google admin emails.")
    oauth_admin_subs: List[str] = Field(default_factory=list, description="Configured Google admin subject IDs.")
    metadata_db_path: str = Field(..., description="SQLite metadata database path.")
    providers: List[ProviderHealthItem] = Field(default_factory=list, description="Live provider connectivity check results.")


class AdminVectorClearRequest(BaseModel):
    """Privileged admin global vector purge request."""
    confirmation: str = Field(..., description="Must exactly match 'CONFIRM_ADMIN_GLOBAL_PURGE'.")
    provider_id: Optional[str] = Field(None, description="Optional provider identifier to clear specifically.")


class AdminVectorClearResponse(BaseModel):
    """Admin global vector clear outcome."""
    status: str = Field("success", description="Operation status.")
    message: str = Field(..., description="Outcome explanation.")
    purged_documents: int = Field(..., description="Count of purged documents.")
    purged_chunks: int = Field(..., description="Count of purged vector chunks.")
    provider_id: Optional[str] = Field(None, description="Provider affected if scoped.")
