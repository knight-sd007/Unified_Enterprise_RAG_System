"""
Unit and integration tests for Dynamic Model Selection and Provider Health Probes.
"""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.dependencies import create_session_token, set_rag_pipeline
from api.main import app
from rag.chunker import Chunk
from rag.pipeline import RAGPipeline
from rag.vector_store import InMemoryVectorStore
from rag.storage.metadata_db import get_metadata_repo


class TestDynamicModelsAndHealth(unittest.TestCase):
    """Test suite for decoupled model execution and provider connectivity probes."""

    def setUp(self):
        self.session_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-session-signing-key-789")
        self.session_key_patcher.start()

        self.client = TestClient(app)
        self.store = InMemoryVectorStore()
        self.pipeline = RAGPipeline(vector_store=self.store)
        set_rag_pipeline(self.pipeline)

        # Seed OAuth tokens in SQLite for Google Drive access
        get_metadata_repo().save_oauth_tokens(
            user_id="google_alice",
            provider="google",
            token_data={"access_token": "ya29.mock_token", "refresh_token": "mock_refresh"},
        )

        self.auth_token = create_session_token(user_id="google_alice", role="user", auth_type="google")
        self.auth_headers = {"Authorization": f"Bearer {self.auth_token}"}

    def tearDown(self):
        get_metadata_repo().delete_all_global()
        get_metadata_repo().delete_oauth_tokens("google_alice", "google")
        set_rag_pipeline(None)
        self.session_key_patcher.stop()

    def test_providers_health_endpoint(self):
        """GET /health/providers returns accurate status for each provider."""
        res = self.client.get("/health/providers")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(len(data["providers"]), 3)

        p_map = {p["provider_id"]: p for p in data["providers"]}
        self.assertIn("gemini", p_map)
        self.assertIn("openai", p_map)
        self.assertIn("nvidia_nim", p_map)
        for p in p_map.values():
            self.assertIn(p["status"], ["connected", "not_configured", "unreachable"])

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    @patch("providers.openai_provider.OpenAIProvider.is_configured", return_value=True)
    @patch("providers.gemini_provider.GeminiProvider.embed_query", return_value=[0.1] * 768)
    @patch("providers.openai_provider.OpenAIProvider.generate", return_value="OpenAI generated answer.")
    def test_decoupled_rag_query(self, mock_generate, mock_embed, _mock_oai_conf, _mock_gemini_conf):
        """RAG query can use Gemini for embedding search and OpenAI for text generation."""
        chunk = Chunk(
            content="Enterprise compliance guidelines.",
            metadata={"filename": "doc.pdf", "chunk_id": "c1", "doc_id": "d1", "owner_id": "google_alice"},
        )
        self.store.add_chunks([chunk], [[0.1] * 768], "gemini", owner_id="google_alice")

        res = self.client.post(
            "/api/v1/rag/query",
            headers=self.auth_headers,
            json={
                "query": "What are the guidelines?",
                "embedding_provider_id": "gemini",
                "embedding_model": "gemini-embedding-2",
                "chat_provider_id": "openai",
                "chat_model": "gpt-4o",
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["answer"], "OpenAI generated answer.")
        self.assertEqual(data["chat_provider_id"], "openai")
        self.assertEqual(data["chat_model"], "gpt-4o")
        self.assertEqual(data["embedding_provider_id"], "gemini")
        self.assertEqual(data["embedding_model"], "gemini-embedding-2")
        self.assertEqual(data["retrieved_count"], 1)


if __name__ == "__main__":
    unittest.main()
