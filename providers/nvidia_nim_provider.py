"""
Unified NVIDIA NIM Provider Implementation.
"""

import math
from typing import List, Optional, Any
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger
from utils.security import sanitize_error_message

EXPECTED_NVIDIA_EMBEDDING_DIMENSION = 2048


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

    @staticmethod
    def _validate_vector(vec: Any, expected_dim: int = EXPECTED_NVIDIA_EMBEDDING_DIMENSION) -> List[float]:
        """
        Validates that an embedding vector is non-empty, numeric, finite, and matches expected dimension.
        Rejects empty, malformed, or inconsistent vectors.
        """
        if not isinstance(vec, (list, tuple)):
            raise RuntimeError("NVIDIA NIM Embedding Error: Embedding vector is not a list or sequence.")
        if len(vec) == 0:
            raise RuntimeError("NVIDIA NIM Embedding Error: Empty vector received from NVIDIA NIM API.")
        if len(vec) != expected_dim:
            raise RuntimeError(
                f"NVIDIA NIM Embedding Error: Vector dimension mismatch. Expected {expected_dim}, received {len(vec)}."
            )

        float_vec = []
        for idx, val in enumerate(vec):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise RuntimeError(
                    f"NVIDIA NIM Embedding Error: Non-numeric value at index {idx} in vector."
                )
            f_val = float(val)
            if not math.isfinite(f_val):
                raise RuntimeError(
                    f"NVIDIA NIM Embedding Error: Non-finite value (NaN/Inf) at index {idx} in vector."
                )
            float_vec.append(f_val)
        return float_vec

    def _get_client(self):
        """Helper to create OpenAI-compatible client for NVIDIA NIM."""
        import openai
        api_key = Config.get_nvidia_api_key()
        base_url = Config.get_nvidia_base_url()
        if not api_key:
            raise ValueError("NVIDIA API key is missing. Set NVIDIA_API_KEY in environment or settings.")
        return openai.OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates document embeddings using NVIDIA NIM API with strict validation."""
        if not texts:
            return []
        if not self.is_configured():
            raise ValueError("NVIDIA NIM provider is not configured.")

        model = self.get_embedding_model_name()
        spec = Config.get_provider_spec("nvidia_nim")
        expected_dim = spec.dimension if spec else EXPECTED_NVIDIA_EMBEDDING_DIMENSION

        try:
            client = self._get_client()
            logger.info(f"Generating NVIDIA NIM embeddings ({len(texts)} texts, model: '{model}')")
            response = client.embeddings.create(
                model=model,
                input=texts,
                extra_body={
                    "input_type": "passage",
                    "truncate": "NONE"
                }
            )
            embeddings: List[List[float]] = []
            for item in response.data:
                valid_vec = self._validate_vector(list(item.embedding), expected_dim=expected_dim)
                embeddings.append(valid_vec)
            return embeddings
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"NVIDIA NIM embedding error: {clean_err}")
            if str(clean_err).startswith("NVIDIA NIM Embedding Error:"):
                raise RuntimeError(clean_err)
            raise RuntimeError(f"NVIDIA NIM Embedding Error: {clean_err}")

    def embed_query(self, query: str) -> List[float]:
        """Generates query embedding using NVIDIA NIM API with strict validation and query input_type."""
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty for embedding generation.")
        if not self.is_configured():
            raise ValueError("NVIDIA NIM provider is not configured.")

        model = self.get_embedding_model_name()
        spec = Config.get_provider_spec("nvidia_nim")
        expected_dim = spec.dimension if spec else EXPECTED_NVIDIA_EMBEDDING_DIMENSION

        try:
            client = self._get_client()
            logger.info(f"Generating NVIDIA NIM query embedding (model: '{model}')")
            response = client.embeddings.create(
                model=model,
                input=[query.strip()],
                extra_body={
                    "input_type": "query",
                    "truncate": "NONE"
                }
            )
            if not response.data or len(response.data) == 0:
                raise RuntimeError("NVIDIA NIM Embedding Error: Empty data list returned from NVIDIA NIM API.")
            return self._validate_vector(list(response.data[0].embedding), expected_dim=expected_dim)
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"NVIDIA NIM query embedding error: {clean_err}")
            if str(clean_err).startswith("NVIDIA NIM Embedding Error:"):
                raise RuntimeError(clean_err)
            raise RuntimeError(f"NVIDIA NIM Embedding Error: {clean_err}")

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
