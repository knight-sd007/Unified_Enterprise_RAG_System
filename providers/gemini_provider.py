"""
Unified Google Gemini Provider Implementation.
"""

from typing import List, Optional
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger
from utils.security import sanitize_error_message


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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates document embeddings using Gemini API."""
        if not texts:
            return []
        if not self.is_configured():
            raise ValueError("Google Gemini provider is not configured.")

        api_key = Config.get_gemini_api_key()
        model_name = self.get_embedding_model_name()

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            logger.info(f"Generating Gemini document embeddings ({len(texts)} texts, model: '{model_name}')")

            embeddings = []
            for text in texts:
                res = client.models.embed_content(
                    model=model_name,
                    contents=text
                )
                if hasattr(res, 'embedding') and hasattr(res.embedding, 'values'):
                    embeddings.append(list(res.embedding.values))
                elif hasattr(res, 'embeddings') and len(res.embeddings) > 0:
                    embeddings.append(list(res.embeddings[0].values))
                else:
                    embeddings.append([])
            return embeddings
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Gemini embedding error: {clean_err}")
            raise RuntimeError(f"Gemini Embedding Error: {clean_err}")

    def embed_query(self, query: str) -> List[float]:
        """Generates query embedding using Gemini API."""
        results = self.embed_documents([query])
        return results[0] if results else []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates chat completion using Gemini API."""
        if not self.is_configured():
            raise ValueError("Google Gemini provider is not configured.")

        api_key = Config.get_gemini_api_key()
        model_name = self.get_chat_model_name()

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            logger.info(f"Generating Gemini completion (model: '{model_name}')")

            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System Instruction:\n{system_prompt}\n\nUser Question:\n{prompt}"

            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt
            )
            return response.text or ""
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Gemini generation error: {clean_err}")
            raise RuntimeError(f"Gemini Generation Error: {clean_err}")
