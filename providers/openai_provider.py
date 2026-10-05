"""
Unified OpenAI Provider Implementation.
"""

import time
from typing import List, Optional, Dict, Any
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger
from utils.security import sanitize_error_message


class OpenAIProvider(BaseAIProvider):
    """Unified OpenAI Provider handling Embeddings and Chat."""

    def __init__(self):
        super().__init__(name="OpenAI", provider_id="openai")

    def is_configured(self) -> bool:
        """Checks if OpenAI API key is present."""
        return Config.is_openai_configured()

    def get_embedding_model_name(self) -> str:
        """Returns embedding model name."""
        return Config.get_openai_embedding_model()

    def get_chat_model_name(self) -> str:
        """Returns chat model name."""
        return Config.get_openai_chat_model()

    def _get_client(self):
        """Helper to create OpenAI API client."""
        import openai
        api_key = Config.get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in environment or settings.")
        return openai.OpenAI(api_key=api_key)

    def check_connectivity(self) -> Dict[str, Any]:
        """Performs live connectivity check against OpenAI API."""
        if not self.is_configured():
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "not_configured",
                "message": "OPENAI_API_KEY is not configured.",
                "latency_ms": None,
            }

        start = time.perf_counter()
        try:
            client = self._get_client()
            # Lightweight models list check
            client.models.list()
            latency = round((time.perf_counter() - start) * 1000, 2)
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "connected",
                "message": "Successfully connected to OpenAI API.",
                "latency_ms": latency,
            }
        except Exception as e:
            clean_err = sanitize_error_message(e)
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "unreachable",
                "message": f"OpenAI connectivity error: {clean_err}",
                "latency_ms": None,
            }

    def embed_documents(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generates document embeddings using OpenAI API."""
        if not texts:
            return []
        if not self.is_configured():
            raise ValueError("OpenAI provider is not configured.")

        model = model or self.get_embedding_model_name()
        try:
            client = self._get_client()
            logger.info(f"Generating OpenAI embeddings ({len(texts)} texts, model: '{model}')")
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"OpenAI embedding error: {clean_err}")
            raise RuntimeError(f"OpenAI Embedding Error: {clean_err}")

    def embed_query(self, query: str, model: Optional[str] = None) -> List[float]:
        """Generates query embedding using OpenAI API."""
        results = self.embed_documents([query], model=model)
        return results[0] if results else []

    def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generates completion using OpenAI API."""
        if not self.is_configured():
            raise ValueError("OpenAI provider is not configured.")

        model = model or self.get_chat_model_name()
        try:
            client = self._get_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.info(f"Generating OpenAI completion (model: '{model}')")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"OpenAI generation error: {clean_err}")
            raise RuntimeError(f"OpenAI Generation Error: {clean_err}")
