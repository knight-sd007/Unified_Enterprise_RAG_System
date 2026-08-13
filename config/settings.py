"""
Centralized Configuration Manager for Unified Enterprise RAG System.

Resolves settings dynamically from:
1. Streamlit Secrets (st.secrets) when hosted in cloud environment.
2. Local .env file for local development.
3. System Environment Variables (os.environ).
"""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    """Centralized configuration manager providing type-safe settings retrieval."""

    @staticmethod
    def _get_val(key: str, default: str = "") -> str:
        """Helper to retrieve environment key from st.secrets or os.getenv."""
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                val = st.secrets[key]
                if val is not None:
                    return str(val).strip()
        except Exception:
            pass

        val = os.getenv(key, default)
        return str(val).strip() if val is not None else default

    @classmethod
    def get_app_access_key(cls) -> str:
        """Returns application access key."""
        return cls._get_val("APP_ACCESS_KEY", "admin123")

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
        return cls._get_val("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

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
        return cls._get_val("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")

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
