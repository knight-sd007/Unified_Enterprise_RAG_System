"""
Unified Google Gemini Provider Implementation.
"""

import math
import time
from typing import List, Optional, Any, Dict
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger
from utils.security import sanitize_error_message

EXPECTED_GEMINI_EMBEDDING_DIMENSION = 768


class GeminiProvider(BaseAIProvider):
    """Unified Google Gemini Provider handling Embeddings and Chat."""

    def __init__(self):
        super().__init__(name="Google Gemini", provider_id="gemini")

    def is_configured(self) -> bool:
        """Checks if Gemini API key is present."""
        return Config.is_gemini_configured()

    def get_embedding_model_name(self) -> str:
        """Returns Gemini embedding model."""
        return Config.get_gemini_embedding_model()

    def get_chat_model_name(self) -> str:
        """Returns Gemini chat model."""
        return Config.get_gemini_chat_model()

    def check_connectivity(self) -> Dict[str, Any]:
        """Performs live connectivity check against Google Gemini API."""
        if not self.is_configured():
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "not_configured",
                "message": "GEMINI_API_KEY is not configured.",
                "latency_ms": None,
            }

        start = time.perf_counter()
        try:
            api_key = Config.get_gemini_api_key()
            from google import genai
            client = genai.Client(api_key=api_key)
            # Lightweight dynamic model lookup using configured chat model
            configured_model = self.get_chat_model_name()
            probe_model = configured_model if configured_model.startswith("models/") else f"models/{configured_model}"
            client.models.get(model=probe_model)
            latency = round((time.perf_counter() - start) * 1000, 2)
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "connected",
                "message": f"Successfully connected to Google Gemini API (probed '{configured_model}').",
                "latency_ms": latency,
            }
        except Exception as e:
            clean_err = sanitize_error_message(e)
            return {
                "provider_id": self.provider_id,
                "name": self.name,
                "status": "unreachable",
                "message": f"Gemini connectivity error: {clean_err}",
                "latency_ms": None,
            }

    @staticmethod
    def _validate_vector(vec: Any, expected_dim: int = EXPECTED_GEMINI_EMBEDDING_DIMENSION) -> List[float]:
        """
        Validates that an embedding vector is non-empty, numeric, finite, and matches expected dimension.
        Rejects empty, malformed, or inconsistent vectors.
        """
        if not isinstance(vec, (list, tuple)):
            raise RuntimeError("Gemini Embedding Error: Embedding vector is not a list or sequence.")
        if len(vec) == 0:
            raise RuntimeError("Gemini Embedding Error: Empty vector received from Gemini API.")
        if len(vec) != expected_dim:
            raise RuntimeError(
                f"Gemini Embedding Error: Vector dimension mismatch. Expected {expected_dim}, received {len(vec)}."
            )

        float_vec = []
        for idx, val in enumerate(vec):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise RuntimeError(
                    f"Gemini Embedding Error: Non-numeric value at index {idx} in vector."
                )
            f_val = float(val)
            if not math.isfinite(f_val):
                raise RuntimeError(
                    f"Gemini Embedding Error: Non-finite value (NaN/Inf) at index {idx} in vector."
                )
            float_vec.append(f_val)
        return float_vec

    def embed_documents(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generates document embeddings using Gemini API with strict validation."""
        if not texts:
            return []
        if not self.is_configured():
            raise ValueError("Google Gemini provider is not configured.")

        api_key = Config.get_gemini_api_key()
        model_name = model or self.get_embedding_model_name()
        spec = Config.get_provider_spec("gemini")
        expected_dim = spec.dimension if spec else EXPECTED_GEMINI_EMBEDDING_DIMENSION

        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            logger.info(f"Generating Gemini document embeddings ({len(texts)} texts, model: '{model_name}')")

            config = types.EmbedContentConfig(
                output_dimensionality=expected_dim
            )

            embeddings: List[List[float]] = []
            for text in texts:
                formatted_text = f"title: none | text: {text}"
                res = client.models.embed_content(
                    model=model_name,
                    contents=formatted_text,
                    config=config
                )
                raw_values = None
                if hasattr(res, 'embedding') and res.embedding is not None and hasattr(res.embedding, 'values'):
                    raw_values = res.embedding.values
                elif hasattr(res, 'embeddings') and res.embeddings and len(res.embeddings) > 0:
                    raw_values = res.embeddings[0].values

                if raw_values is None:
                    raise RuntimeError("Gemini Embedding Error: No embedding values found in Gemini API response.")

                valid_vec = self._validate_vector(list(raw_values), expected_dim=expected_dim)
                embeddings.append(valid_vec)
            return embeddings
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Gemini embedding error: {clean_err}")
            raise RuntimeError(f"Gemini Embedding Error: {clean_err}")

    def embed_query(self, query: str, model: Optional[str] = None) -> List[float]:
        """Generates query embedding using Gemini API with strict validation."""
        if not self.is_configured():
            raise ValueError("Google Gemini provider is not configured.")

        api_key = Config.get_gemini_api_key()
        model_name = model or self.get_embedding_model_name()
        spec = Config.get_provider_spec("gemini")
        expected_dim = spec.dimension if spec else EXPECTED_GEMINI_EMBEDDING_DIMENSION

        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            logger.info(f"Generating Gemini query embedding (model: '{model_name}')")

            config = types.EmbedContentConfig(
                output_dimensionality=expected_dim
            )

            res = client.models.embed_content(
                model=model_name,
                contents=f"task: question answering | query: {query}",
                config=config
            )
            raw_values = None
            if hasattr(res, 'embedding') and res.embedding is not None and hasattr(res.embedding, 'values'):
                raw_values = res.embedding.values
            elif hasattr(res, 'embeddings') and res.embeddings and len(res.embeddings) > 0:
                raw_values = res.embeddings[0].values

            if raw_values is None:
                raise RuntimeError("Gemini Embedding Error: No embedding values found in Gemini API response.")

            return self._validate_vector(list(raw_values), expected_dim=expected_dim)
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Gemini query embedding error: {clean_err}")
            raise RuntimeError(f"Gemini Embedding Error: {clean_err}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """Generates completion using Gemini API."""
        if not self.is_configured():
            raise ValueError("Google Gemini provider is not configured.")

        api_key = Config.get_gemini_api_key()
        model_name = model or self.get_chat_model_name()

        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            logger.info(f"Generating Gemini completion (model: '{model_name}')")

            config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=0.2
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            return response.text or ""
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Gemini generation error: {clean_err}")
            raise RuntimeError(f"Gemini Generation Error: {clean_err}")
