"""
Unit tests for RAG Semantic Querying, Telemetry, and Index Management API endpoints.
"""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.dependencies import create_session_token, set_rag_pipeline
from api.main import app
from rag.chunker import Chunk
from rag.pipeline import RAGPipeline
from rag.vector_store import InMemoryVectorStore


class TestAPIRAGOperations(unittest.TestCase):
    """Test suite for /api/v1/rag/* routes (query, stats, index clearing)."""

    def setUp(self):
        self.key_patcher = patch("config.settings.Config.get_app_access_key", return_value="test-app-key-123")
        self.admin_key_patcher = patch("config.settings.Config.get_admin_access_key", return_value="test-admin-key-999")
        self.session_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-signing-key-789")
        self.key_patcher.start()
        self.admin_key_patcher.start()
        self.session_key_patcher.start()
        self.client = TestClient(app)
        self.auth_token = create_session_token()
        self.auth_headers = {"Authorization": f"Bearer {self.auth_token}"}
        # Use isolated in-memory vector store and pipeline for tests
        self.mock_store = InMemoryVectorStore()
        self.pipeline = RAGPipeline(vector_store=self.mock_store)
        set_rag_pipeline(self.pipeline)

    def tearDown(self):
        set_rag_pipeline(None)
        self.session_key_patcher.stop()
        self.admin_key_patcher.stop()
        self.key_patcher.stop()

    # -------------------------------------------------------------------------
    # Authentication Enforcement Tests
    # -------------------------------------------------------------------------

    def test_unauthenticated_query_rejected(self):
        """Unauthenticated POST /api/v1/rag/query is rejected with 401."""
        response = self.client.post(
            "/api/v1/rag/query",
            json={"query": "What are the compliance rules?"},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_stats_rejected(self):
        """Unauthenticated GET /api/v1/rag/stats is rejected with 401."""
        response = self.client.get("/api/v1/rag/stats")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_clear_index_rejected(self):
        """Unauthenticated DELETE /api/v1/rag/index is rejected with 401."""
        response = self.client.delete("/api/v1/rag/index")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    # -------------------------------------------------------------------------
    # Semantic Query & QA Tests
    # -------------------------------------------------------------------------

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    @patch("providers.gemini_provider.GeminiProvider.embed_query")
    @patch("providers.gemini_provider.GeminiProvider.generate")
    def test_successful_query_with_grounded_answer(self, mock_gen, mock_embed_q, _mock_conf):
        """Authenticated query returns grounded answer with full source citation details."""
        # Seed the in-memory store with a Gemini vector
        chunk = Chunk(
            content="Section 4 requires all enterprise logs to be encrypted with AES-256.",
            metadata={"filename": "security_policy.pdf", "chunk_id": "chunk_0"},
        )
        self.mock_store.add_chunks([chunk], [[0.1] * 768], "gemini")

        mock_embed_q.return_value = [0.1] * 768
        mock_gen.return_value = "Enterprise logs must be encrypted using AES-256 as specified in Section 4."

        response = self.client.post(
            "/api/v1/rag/query",
            headers=self.auth_headers,
            json={
                "query": "What encryption standard is required for logs?",
                "provider_id": "gemini",
                "top_k": 3,
                "similarity_threshold": 0.2,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("AES-256", data["answer"])
        self.assertEqual(data["provider_id"], "gemini")
        self.assertEqual(data["retrieved_count"], 1)
        self.assertEqual(len(data["sources"]), 1)

        source = data["sources"][0]
        self.assertEqual(source["chunk_id"], "chunk_0")
        self.assertIn("AES-256", source["content"])
        self.assertGreater(source["score"], 0.99)
        self.assertEqual(source["metadata"]["filename"], "security_policy.pdf")

    def test_query_with_invalid_provider_rejected(self):
        """Query with unknown provider ID is rejected with 400."""
        response = self.client.post(
            "/api/v1/rag/query",
            headers=self.auth_headers,
            json={"query": "Test question?", "provider_id": "unknown_ai_provider"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_PROVIDER")

    @patch("providers.openai_provider.OpenAIProvider.is_configured", return_value=False)
    def test_query_with_unconfigured_provider_rejected(self, _mock_conf):
        """Query using unconfigured provider is rejected with 400."""
        response = self.client.post(
            "/api/v1/rag/query",
            headers=self.auth_headers,
            json={"query": "Test question?", "provider_id": "openai"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "PROVIDER_NOT_CONFIGURED")

    def test_query_empty_string_rejected(self):
        """Empty query string fails request validation."""
        response = self.client.post(
            "/api/v1/rag/query",
            headers=self.auth_headers,
            json={"query": "", "provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "INVALID_REQUEST")

    @patch("providers.openai_provider.OpenAIProvider.is_configured", return_value=True)
    def test_query_provider_mismatch_rejected(self, _mock_conf):
        """Attempting to query OpenAI against an index created with Gemini is rejected."""
        chunk = Chunk(
            content="Gemini chunk.",
            metadata={"filename": "doc.txt", "chunk_id": "c0"},
        )
        self.mock_store.add_chunks([chunk], [[0.1] * 768], "gemini")

        response = self.client.post(
            "/api/v1/rag/query",
            headers=self.auth_headers,
            json={"query": "Sample question?", "provider_id": "openai"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "PROVIDER_MISMATCH")

    # -------------------------------------------------------------------------
    # Telemetry & Stats Tests
    # -------------------------------------------------------------------------

    def test_rag_stats_empty_store(self):
        """GET /api/v1/rag/stats returns valid telemetry when vector store is empty."""
        response = self.client.get(
            "/api/v1/rag/stats",
            headers=self.auth_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["provider_id"], "gemini")
        self.assertEqual(data["count"], 0)
        self.assertEqual(data["dimension"], 768)
        self.assertEqual(data["collection_name"], "p06_gemini_embedding_2_768")
        self.assertIn("status", data)

    def test_rag_stats_populated_store(self):
        """GET /api/v1/rag/stats reflects indexed chunk count and vector properties."""
        chunk = Chunk(
            content="Sample text.",
            metadata={"filename": "doc.txt", "chunk_id": "c0"},
        )
        self.mock_store.add_chunks([chunk], [[0.1] * 768], "gemini")

        response = self.client.get(
            "/api/v1/rag/stats",
            headers=self.auth_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["dimension"], 768)
        self.assertEqual(data["status"], "Indexed")

    def test_rag_stats_invalid_provider_rejected(self):
        """GET /api/v1/rag/stats with unknown provider is rejected with 400."""
        response = self.client.get(
            "/api/v1/rag/stats",
            headers=self.auth_headers,
            params={"provider_id": "invalid_provider"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "INVALID_PROVIDER")

    # -------------------------------------------------------------------------
    # Index Clearing and Administration Tests
    # -------------------------------------------------------------------------

    def test_clear_vector_index_user_role_forbidden(self):
        """DELETE /api/v1/rag/index called by non-admin user is rejected with 403 Forbidden."""
        response = self.client.delete(
            "/api/v1/rag/index",
            headers=self.auth_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "FORBIDDEN")

    def test_clear_vector_index_admin_role_success(self):
        """DELETE /api/v1/rag/index called by admin user clears global vector index."""
        chunk = Chunk(
            content="Content to clear.",
            metadata={"filename": "doc.txt", "chunk_id": "c0", "owner_id": "default_user"},
        )
        self.mock_store.add_chunks([chunk], [[0.1] * 768], "gemini", owner_id="default_user")
        self.assertEqual(self.mock_store.count(), 1)

        admin_token = create_session_token(user_id="admin_user", role="admin")
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        response = self.client.delete(
            "/api/v1/rag/index",
            headers=admin_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["provider_id"], "gemini")
        self.assertEqual(self.mock_store.count(), 0)


if __name__ == "__main__":
    unittest.main()
