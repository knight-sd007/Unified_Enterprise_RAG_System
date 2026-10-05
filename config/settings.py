"""
Centralized Configuration Manager for Unified Enterprise RAG System.

Resolves settings dynamically from:
1. Local .env file for local development.
2. System Environment Variables (os.environ).
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class ProviderVectorSpec:
    """Deterministic vector specification for an AI provider and its Qdrant collection."""
    provider_id: str
    embedding_model: str
    dimension: int
    collection_name: str
    distance_metric: str = "Cosine"


# Deterministic Provider-to-Collection Vector Space Specifications
PROVIDER_VECTOR_SPECS: Dict[str, ProviderVectorSpec] = {
    "gemini": ProviderVectorSpec(
        provider_id="gemini",
        embedding_model="gemini-embedding-2",
        dimension=768,
        collection_name="p06_gemini_embedding_2_768",
        distance_metric="Cosine",
    ),
    "nvidia_nim": ProviderVectorSpec(
        provider_id="nvidia_nim",
        embedding_model="nvidia/llama-nemotron-embed-1b-v2",
        dimension=2048,
        collection_name="p06_nvidia_llama_nemotron_embed_1b_v2_2048",
        distance_metric="Cosine",
    ),
    "openai": ProviderVectorSpec(
        provider_id="openai",
        embedding_model="text-embedding-3-small",
        dimension=1536,
        collection_name="p06_openai_text_embedding_3_small",
        distance_metric="Cosine",
    ),
}


# Supported Chat and Embedding Model Catalogs
SUPPORTED_CHAT_MODELS: Dict[str, List[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "gemini": ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    "nvidia_nim": ["nvidia/nemotron-3-super-120b-a12b", "meta/llama-3.1-8b-instruct", "meta/llama-3.1-70b-instruct"],
}

SUPPORTED_EMBEDDING_MODELS: Dict[str, List[Dict[str, Any]]] = {
    "gemini": [
        {"model": "gemini-embedding-2", "dimension": 768},
        {"model": "text-embedding-004", "dimension": 768},
    ],
    "nvidia_nim": [
        {"model": "nvidia/llama-nemotron-embed-1b-v2", "dimension": 2048},
    ],
    "openai": [
        {"model": "text-embedding-3-small", "dimension": 1536},
        {"model": "text-embedding-ada-002", "dimension": 1536},
    ],
}


class Config:
    """Centralized configuration manager providing type-safe settings retrieval."""

    @staticmethod
    def _get_val(key: str, default: str = "") -> str:
        """Helper to retrieve environment variable with optional default."""
        val = os.getenv(key, default)
        return str(val).strip() if val is not None else default

    @classmethod
    def get_environment(cls) -> str:
        """Returns runtime environment mode ('production', 'development', etc.)."""
        return cls._get_val("ENVIRONMENT", "production")

    @classmethod
    def get_app_access_key(cls) -> str:
        """Returns application access key."""
        return cls._get_val("APP_ACCESS_KEY", "")

    @classmethod
    def get_admin_access_key(cls) -> str:
        """Returns administrative access key."""
        return cls._get_val("ADMIN_ACCESS_KEY", "")

    @classmethod
    def get_session_signing_key(cls) -> str:
        """
        Returns dedicated session signing key used for cryptographic session token serialization.
        Must be a high-entropy secret distinct from APP_ACCESS_KEY and ADMIN_ACCESS_KEY.
        """
        return cls._get_val("SESSION_SIGNING_KEY", "")

    @classmethod
    def get_ai_provider(cls) -> str:
        """Returns default AI provider selection (openai, gemini, nvidia_nim)."""
        return cls._get_val("AI_PROVIDER", "openai")

    # OpenAI Configuration
    @classmethod
    def get_openai_api_key(cls) -> str:
        """Returns OpenAI API key."""
        return cls._get_val("OPENAI_API_KEY", "")

    @classmethod
    def get_openai_chat_model(cls) -> str:
        """Returns OpenAI chat model name."""
        return cls._get_val("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    @classmethod
    def get_openai_embedding_model(cls) -> str:
        """Returns OpenAI embedding model name."""
        return cls._get_val("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Google Gemini Configuration
    @classmethod
    def get_gemini_api_key(cls) -> str:
        """Returns Google Gemini API key."""
        return cls._get_val("GEMINI_API_KEY", "")

    @classmethod
    def get_gemini_chat_model(cls) -> str:
        """Returns Gemini chat model name."""
        return cls._get_val("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

    @classmethod
    def get_gemini_embedding_model(cls) -> str:
        """Returns Gemini embedding model name."""
        return cls._get_val("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")

    # NVIDIA NIM Configuration
    @classmethod
    def get_nvidia_api_key(cls) -> str:
        """Returns NVIDIA NIM API key."""
        return cls._get_val("NVIDIA_API_KEY", "")

    @classmethod
    def get_nvidia_base_url(cls) -> str:
        """Returns NVIDIA NIM base URL endpoint."""
        return cls._get_val("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

    @classmethod
    def get_nvidia_chat_model(cls) -> str:
        """Returns NVIDIA NIM chat model name."""
        return cls._get_val("NVIDIA_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b")

    @classmethod
    def get_nvidia_embedding_model(cls) -> str:
        """Returns NVIDIA NIM embedding model name."""
        return cls._get_val("NVIDIA_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-1b-v2")

    # Qdrant Cloud Configuration
    @classmethod
    def get_qdrant_url(cls) -> str:
        """Returns Qdrant Cloud cluster endpoint URL."""
        return cls._get_val("QDRANT_URL", "")

    @classmethod
    def get_qdrant_api_key(cls) -> str:
        """Returns Qdrant Cloud API key."""
        return cls._get_val("QDRANT_API_KEY", "")

    @classmethod
    def is_qdrant_configured(cls) -> bool:
        """Checks if Qdrant URL and API key are configured."""
        return bool(cls.get_qdrant_url() and cls.get_qdrant_api_key())

    @classmethod
    def get_qdrant_host(cls) -> Optional[str]:
        """Returns sanitized Qdrant host without credentials."""
        url = cls.get_qdrant_url()
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or parsed.path
        except Exception:
            return "configured"

    # Provider Readiness Helpers
    @classmethod
    def is_openai_configured(cls) -> bool:
        """Checks if OpenAI API key is present."""
        return bool(cls.get_openai_api_key())

    @classmethod
    def is_gemini_configured(cls) -> bool:
        """Checks if Google Gemini API key is present."""
        return bool(cls.get_gemini_api_key())

    @classmethod
    def is_nvidia_configured(cls) -> bool:
        """Checks if NVIDIA NIM API key is present."""
        return bool(cls.get_nvidia_api_key())

    @classmethod
    def get_provider_spec(cls, provider_id: str) -> Optional[ProviderVectorSpec]:
        """Returns vector specification for a given provider ID."""
        return PROVIDER_VECTOR_SPECS.get(provider_id)

    @classmethod
    def get_all_provider_specs(cls) -> Dict[str, ProviderVectorSpec]:
        """Returns all registered provider vector specifications."""
        return dict(PROVIDER_VECTOR_SPECS)

    # Google OAuth 2.0 / OpenID Connect Settings
    @classmethod
    def get_google_client_id(cls) -> str:
        """Returns Google OAuth Client ID."""
        return cls._get_val("GOOGLE_CLIENT_ID", "")

    @classmethod
    def get_google_client_secret(cls) -> str:
        """Returns Google OAuth Client Secret."""
        return cls._get_val("GOOGLE_CLIENT_SECRET", "")

    @classmethod
    def get_google_redirect_uri(cls) -> str:
        """Returns Google OAuth Redirect Callback URI."""
        return cls._get_val("GOOGLE_REDIRECT_URI", "https://rag.vaikuntrix.in/api/v1/auth/google/callback")

    @classmethod
    def get_google_admin_emails(cls) -> list[str]:
        """Returns list of trusted Google account emails with administrator privileges."""
        raw = cls._get_val("GOOGLE_ADMIN_EMAILS", "")
        return [e.strip().lower() for e in raw.split(",") if e.strip()]

    @classmethod
    def get_google_admin_subs(cls) -> list[str]:
        """Returns list of trusted Google account subject IDs (sub) with administrator privileges."""
        raw = cls._get_val("GOOGLE_ADMIN_SUBS", "")
        return [s.strip() for s in raw.split(",") if s.strip()]

    @classmethod
    def is_google_oauth_configured(cls) -> bool:
        """Checks if Google OAuth is configured."""
        return bool(cls.get_google_client_id() and cls.get_google_client_secret())

    # Metadata & Token Persistence Settings
    @classmethod
    def get_metadata_db_path(cls) -> str:
        """Returns SQLite database path for persistent document metadata & encrypted tokens."""
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "p06_metadata.db")
        return cls._get_val("METADATA_DB_PATH", default_path)

    @classmethod
    def get_supported_chat_models(cls, provider_id: str) -> list[str]:
        """Returns approved chat completion models for provider."""
        return SUPPORTED_CHAT_MODELS.get(provider_id.strip().lower(), [])

    @classmethod
    def get_supported_embedding_models(cls, provider_id: str) -> list[dict[str, Any]]:
        """Returns approved embedding models and specs for provider."""
        return SUPPORTED_EMBEDDING_MODELS.get(provider_id.strip().lower(), [])
