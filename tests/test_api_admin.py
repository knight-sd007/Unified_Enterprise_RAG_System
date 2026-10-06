"""
Unit and integration tests for Privileged Admin Console API endpoints (/api/v1/admin/*).
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.dependencies import create_session_token, set_rag_pipeline
from api.main import app
from rag.chunker import Chunk
from rag.pipeline import RAGPipeline
from rag.storage.metadata_db import get_metadata_repo
from rag.vector_store import InMemoryVectorStore


class TestAPIAdminConsole(unittest.TestCase):
    """Test suite for /api/v1/admin/* endpoints."""

    def setUp(self):
        self.key_patcher = patch("config.settings.Config.get_app_access_key", return_value="test-app-key-123")
        self.admin_key_patcher = patch("config.settings.Config.get_admin_access_key", return_value="test-admin-key-999")
        self.signing_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-session-signing-key-789")
        self.key_patcher.start()
        self.admin_key_patcher.start()
        self.signing_key_patcher.start()

        get_metadata_repo().delete_all_global()

        self.client = TestClient(app)

        self.store = InMemoryVectorStore()
        self.pipeline = RAGPipeline(vector_store=self.store)
        set_rag_pipeline(self.pipeline)

        self.user_token = create_session_token(user_id="alice", role="user")
        self.admin_token = create_session_token(user_id="root_admin", role="admin")

        self.user_headers = {"Authorization": f"Bearer {self.user_token}"}
        self.admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

    def tearDown(self):
        get_metadata_repo().delete_all_global()
        set_rag_pipeline(None)
        self.signing_key_patcher.stop()
        self.admin_key_patcher.stop()
        self.key_patcher.stop()

    def test_admin_overview_unauthenticated_rejected(self):
        """Unauthenticated GET /api/v1/admin/overview is rejected with 401."""
        res = self.client.get("/api/v1/admin/overview")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"]["code"], "UNAUTHORIZED")

    def test_admin_overview_user_role_forbidden(self):
        """Regular user role GET /api/v1/admin/overview is rejected with 403."""
        res = self.client.get("/api/v1/admin/overview", headers=self.user_headers)
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["error"]["code"], "FORBIDDEN")

    def test_admin_overview_admin_success(self):
        """Admin user can retrieve overview statistics."""
        repo = get_metadata_repo()
        repo.save_document(
            doc_id="doc_1",
            owner_id="alice",
            filename="report.pdf",
            file_size=2048,
            char_count=500,
            provider_id="gemini",
            embedding_model="gemini-embedding-2",
            chunk_count=3,
        )

        res = self.client.get("/api/v1/admin/overview", headers=self.admin_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["total_documents"], 1)
        self.assertEqual(data["total_chunks"], 3)
        self.assertEqual(data["total_bytes"], 2048)
        self.assertEqual(data["total_users"], 1)
        self.assertIn("vector_store_backend", data)

    def test_admin_diagnostics_no_secret_leaks(self):
        """Admin diagnostics must not leak API keys, access keys, or signing secrets."""
        res = self.client.get("/api/v1/admin/diagnostics", headers=self.admin_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["app_status"], "healthy")
        self.assertIn("providers", data)
        self.assertEqual(len(data["providers"]), 3)

        raw_text = res.text.lower()
        self.assertNotIn("test-app-key-123", raw_text)
        self.assertNotIn("test-admin-key-999", raw_text)
        self.assertNotIn("test-session-signing-key-789", raw_text)

    def test_admin_global_purge_invalid_confirmation_rejected(self):
        """Global purge with invalid confirmation phrase is rejected with 400."""
        res = self.client.post(
            "/api/v1/admin/vectors/clear",
            headers=self.admin_headers,
            json={"confirmation": "wrong_confirmation"},
        )
        self.assertEqual(res.status_code, 400)

    def test_admin_global_purge_regular_user_forbidden(self):
        """Regular user cannot execute global vector purge."""
        res = self.client.post(
            "/api/v1/admin/vectors/clear",
            headers=self.user_headers,
            json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
        )
        self.assertEqual(res.status_code, 403)

    def test_admin_global_purge_success(self):
        """Admin global purge wipes metadata and vector store."""
        repo = get_metadata_repo()
        repo.save_document(
            doc_id="doc_1",
            owner_id="alice",
            filename="report.pdf",
            file_size=2048,
            char_count=500,
            provider_id="gemini",
            embedding_model="gemini-embedding-2",
            chunk_count=3,
        )
        chunk = Chunk(content="Test text", metadata={"chunk_id": "c1", "doc_id": "doc_1"})
        self.store.add_chunks([chunk], [[0.1] * 768], "gemini", owner_id="alice")

        res = self.client.post(
            "/api/v1/admin/vectors/clear",
            headers=self.admin_headers,
            json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["purged_documents"], 1)
        self.assertEqual(data["purged_chunks"], 3)

        # Confirm database and store are now empty
        self.assertEqual(len(repo.list_documents()), 0)
        self.assertEqual(self.store.get_stats()["count"], 0)

    def test_admin_global_purge_first_collection_failure_preserves_metadata(self):
        """Failure: First vector collection fails -> SQLite metadata remains untouched -> HTTP 500."""
        repo = get_metadata_repo()
        repo.save_document(
            doc_id="doc_fail1",
            owner_id="alice",
            filename="critical.pdf",
            file_size=1024,
            char_count=200,
            provider_id="gemini",
            embedding_model="gemini-embedding-2",
            chunk_count=2,
        )

        with patch.object(self.pipeline, "clear_index", side_effect=RuntimeError("Qdrant connection timeout on collection 1")):
            res = self.client.post(
                "/api/v1/admin/vectors/clear",
                headers=self.admin_headers,
                json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
            )
            self.assertEqual(res.status_code, 500)
            self.assertIn("Global vector purge failed", res.json()["error"]["message"])

        # Crucial assertion: SQLite metadata MUST NOT have been deleted
        remaining_docs = repo.list_documents()
        self.assertEqual(len(remaining_docs), 1)
        self.assertEqual(remaining_docs[0]["doc_id"], "doc_fail1")

    def test_admin_global_purge_later_collection_failure_preserves_metadata(self):
        """Later collection failure: Earlier collection cleared, later fails -> SQLite metadata remains untouched -> HTTP 500."""
        repo = get_metadata_repo()
        repo.save_document(
            doc_id="doc_fail2",
            owner_id="bob",
            filename="ledger.pdf",
            file_size=4096,
            char_count=800,
            provider_id="openai",
            embedding_model="text-embedding-3-small",
            chunk_count=5,
        )

        with patch.object(self.pipeline, "clear_index", side_effect=RuntimeError("Qdrant quota exceeded on collection 2")):
            res = self.client.post(
                "/api/v1/admin/vectors/clear",
                headers=self.admin_headers,
                json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
            )
            self.assertEqual(res.status_code, 500)
            self.assertIn("Global vector purge failed", res.json()["error"]["message"])

        # Crucial assertion: SQLite metadata MUST NOT have been deleted
        remaining_docs = repo.list_documents()
        self.assertEqual(len(remaining_docs), 1)
        self.assertEqual(remaining_docs[0]["doc_id"], "doc_fail2")

    def test_admin_global_purge_retry_is_idempotent(self):
        """Retry: A subsequent global purge after partial failure or on empty store succeeds safely."""
        repo = get_metadata_repo()
        repo.save_document(
            doc_id="doc_retry",
            owner_id="alice",
            filename="retry.pdf",
            file_size=1024,
            char_count=200,
            provider_id="gemini",
            embedding_model="gemini-embedding-2",
            chunk_count=1,
        )

        # 1. First attempt fails due to simulated transient error
        with patch.object(self.pipeline, "clear_index", side_effect=RuntimeError("Transient network failure")):
            res1 = self.client.post(
                "/api/v1/admin/vectors/clear",
                headers=self.admin_headers,
                json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
            )
            self.assertEqual(res1.status_code, 500)

        # Metadata remains
        self.assertEqual(len(repo.list_documents()), 1)

        # 2. Retry succeeds
        res2 = self.client.post(
            "/api/v1/admin/vectors/clear",
            headers=self.admin_headers,
            json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
        )
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(len(repo.list_documents()), 0)

        # 3. Another subsequent purge on empty store is safe and returns 200
        res3 = self.client.post(
            "/api/v1/admin/vectors/clear",
            headers=self.admin_headers,
            json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
        )
        self.assertEqual(res3.status_code, 200)
        self.assertEqual(res3.json()["purged_documents"], 0)


if __name__ == "__main__":
    unittest.main()
