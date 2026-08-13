"""
Unified NVIDIA NIM Provider Implementation.
"""

from typing import List, Optional
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger
from utils.security import sanitize_error_message


class NvidiaNimProvider(BaseAIProvider):
    """Unified NVIDIA NIM Provider implementing Embeddings and Chat via OpenAI-compatible API."""

    def __init__(self):
        super().__init__(name="NVIDIA NIM", provider_id="nvidia_nim")

    def is_configured(self) -> bool:
        """Checks if NVIDIA NIM provider is configured."""
        return Config.is_nvidia_configured()

    def get_embedding_model_name(self) -> str:
        """Returns NVIDIA embedding model."""
        return Config.get_nvidia_embedding_model()

    def get_chat_model_name(self) -> str:
        """Returns NVIDIA chat model."""
        return Config.get_nvidia_chat_model()

    def _get_client(self):
        """Helper to create OpenAI-compatible client for NVIDIA NIM."""
        import openai
        api_key = Config.get_nvidia_api_key()
        base_url = Config.get_nvidia_base_url()
        if not api_key:
            raise ValueError("NVIDIA API key is missing. Set NVIDIA_API_KEY in environment or settings.")
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates document embeddings using NVIDIA NIM API."""
        if not texts:
            return []
        if not self.is_configured():
            raise ValueError("NVIDIA NIM provider is not configured.")

        model = self.get_embedding_model_name()
        try:
            client = self._get_client()
            logger.info(f"Generating NVIDIA NIM embeddings ({len(texts)} texts, model: '{model}')")
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"NVIDIA NIM embedding error: {clean_err}")
            raise RuntimeError(f"NVIDIA NIM Embedding Error: {clean_err}")

    def embed_query(self, query: str) -> List[float]:
        """Generates query embedding using NVIDIA NIM API."""
        results = self.embed_documents([query])
        return results[0] if results else []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates chat completion using NVIDIA NIM API."""
        if not self.is_configured():
            raise ValueError("NVIDIA NIM provider is not configured.")

        model = self.get_chat_model_name()
        try:
            client = self._get_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.info(f"Generating NVIDIA NIM completion (model: '{model}')")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"NVIDIA NIM generation error: {clean_err}")
            raise RuntimeError(f"NVIDIA NIM Generation Error: {clean_err}")
