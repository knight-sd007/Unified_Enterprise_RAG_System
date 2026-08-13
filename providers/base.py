"""
Unified AI Provider Base Class.

A single provider owns both vector embedding generation and chat completions.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any


class BaseAIProvider(ABC):
    """Abstract Base Class for Unified AI Providers."""

    def __init__(self, name: str, provider_id: str):
        self.name = name
        self.provider_id = provider_id

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates vector embeddings for document texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generates vector embedding for query text."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
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

    def get_status(self) -> Dict[str, Any]:
        """Returns provider status dictionary."""
        configured = self.is_configured()
        return {
            "name": self.name,
            "provider_id": self.provider_id,
            "configured": configured,
            "status_text": "Configured" if configured else "Not Configured",
            "status_icon": "🟢" if configured else "🔴",
            "embedding_model": self.get_embedding_model_name(),
            "chat_model": self.get_chat_model_name()
        }
