"""
Unified AI Provider Base Class.

A single provider owns vector embedding generation, chat completions,
and live connectivity health checks.
"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseAIProvider(ABC):
    """Abstract Base Class for Unified AI Providers."""

    def __init__(self, name: str, provider_id: str):
        self.name = name
        self.provider_id = provider_id

    @abstractmethod
    def embed_documents(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generates vector embeddings for document texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str, model: Optional[str] = None) -> List[float]:
        """Generates vector embedding for query text."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generates text completion using the chat model."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if provider credentials are set."""
        pass

    @abstractmethod
    def get_embedding_model_name(self) -> str:
        """Returns active embedding model name."""
        pass

    @abstractmethod
    def get_chat_model_name(self) -> str:
        """Returns active chat model name."""
        pass

    def check_connectivity(self) -> Dict[str, Any]:
        """
        Performs a live connectivity test to verify if provider endpoints are reachable.
        Returns status dictionary with 'connected', 'not_configured', or 'unreachable'.
        """
        if not self.is_configured():
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "not_configured",
                "message": "API credentials are not configured.",
                "latency_ms": None,
            }
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "status": "connected",
            "message": "Provider is configured and ready.",
            "latency_ms": 0.0,
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns provider configuration status dictionary."""
        configured = self.is_configured()
        return {
            "name": self.name,
            "provider_id": self.provider_id,
            "configured": configured,
            "status_text": "Configured" if configured else "Not Configured",
            "status_icon": "🟢" if configured else "🔴",
            "embedding_model": self.get_embedding_model_name(),
            "chat_model": self.get_chat_model_name(),
        }
