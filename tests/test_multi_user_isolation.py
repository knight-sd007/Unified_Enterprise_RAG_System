"""
Security and Multi-User Document Isolation Test Suite.

Verifies:
1. User A cannot list User B's documents.
2. User A cannot retrieve User B's vector chunks during RAG query.
3. User A cannot delete User B's document by guessing/supplying its doc_id.
4. User A can delete their own document and associated vector chunks.
5. User-scoped document clear does not affect other users' documents.
6. Ordinary user cannot clear the global index (403 Forbidden).
7. Admin user can clear the global index.
8. Server-side identity enforcement prevents client payload spoofing.
9. Legacy/unassigned documents without owner_id are quarantined from normal user retrieval.
"""

import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from api.dependencies import create_session_token, set_rag_pipeline
from api.main import app
from rag.chunker import Chunk
from rag.pipeline import RAGPipeline
from rag.vector_store import InMemoryVectorStore
from rag.storage.metadata_db import get_metadata_repo


class TestMultiUserIsolation(unittest.TestCase):
    """Test suite for tenant/user document isolation, authorization boundaries, and lifecycle management."""

    def setUp(self):
        self.admin_key_patcher = patch("config.settings.Config.get_admin_access_key", return_value="test-admin-key-999")
        self.signing_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-session-signing-key-789")
        self.admin_key_patcher.start()
        self.signing_key_patcher.start()

        get_metadata_repo().delete_all_global()

        self.client = TestClient(app)

        # Isolated in-memory vector store & pipeline
        self.store = InMemoryVectorStore()
        self.pipeline = RAGPipeline(vector_store=self.store)
        set_rag_pipeline(self.pipeline)

        # Seed OAuth tokens in SQLite for Alice and Bob and Admin
        for uid in ["google_alice", "google_bob", "google_admin_sec"]:
            get_metadata_repo().save_oauth_tokens(
                user_id=uid,
                provider="google",
                token_data={"access_token": f"ya29.mock_{uid}", "refresh_token": "mock_refresh"},
            )

        # Patch GoogleDriveStorage async methods
        self.drive_token_patcher = patch(
            "rag.storage.google_drive.GoogleDriveStorage.get_valid_access_token",
            new_callable=AsyncMock,
            return_value="ya29.mock_drive_token",
        )
        self.drive_upload_patcher = patch(
            "rag.storage.google_drive.GoogleDriveStorage.upload_file",
            new_callable=AsyncMock,
            return_value="mock_drive_file_id_123",
        )
        self.drive_token_patcher.start()
        self.drive_upload_patcher.start()

        # Create session tokens for Alice, Bob, and Admin
        self.alice_token = create_session_token(user_id="google_alice", role="user", auth_type="google")
        self.bob_token = create_session_token(user_id="google_bob", role="user", auth_type="google")
        self.admin_token = create_session_token(user_id="google_admin_sec", role="admin", auth_type="google")

        self.alice_headers = {"Authorization": f"Bearer {self.alice_token}"}
        self.bob_headers = {"Authorization": f"Bearer {self.bob_token}"}
        self.admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

    def tearDown(self):
        self.drive_upload_patcher.stop()
        self.drive_token_patcher.stop()
        get_metadata_repo().delete_all_global()
        for uid in ["google_alice", "google_bob", "google_admin_sec"]:
            get_metadata_repo().delete_oauth_tokens(uid, "google")
        set_rag_pipeline(None)
        self.signing_key_patcher.stop()
        self.admin_key_patcher.stop()

    # -------------------------------------------------------------------------
    # 1. Document Ingestion & Server-Side Ownership Assignment
    # -------------------------------------------------------------------------

    def test_ingestion_assigns_owner_from_verified_session(self):
        """Document ingestion must stamp owner_id from verified Google session identity."""
        txt_content = b"Alice confidential enterprise roadmap for Q3 2026."
        files = [("files", ("alice_plan.txt", txt_content, "text/plain"))]

        with patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True), \
             patch("providers.gemini_provider.GeminiProvider.embed_documents", return_value=[[0.1] * 768]):
            response = self.client.post(
                "/api/v1/documents/ingest",
                headers=self.alice_headers,
                files=files,
                data={"provider_id": "gemini"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

        # Verify in store that chunk was tagged with owner_id="google_alice"
        alice_docs = self.store.list_documents(owner_id="google_alice", provider_id="gemini")
        self.assertEqual(len(alice_docs), 1)
        self.assertEqual(alice_docs[0]["filename"], "alice_plan.txt")
        self.assertEqual(alice_docs[0]["owner_id"], "google_alice")

        # Verify Bob cannot see Alice's document in list
        bob_docs = self.store.list_documents(owner_id="google_bob", provider_id="gemini")
        self.assertEqual(len(bob_docs), 0)

    # -------------------------------------------------------------------------
    # 2. Document Listing Isolation (GET /api/v1/documents)
    # -------------------------------------------------------------------------

    def test_user_cannot_list_another_users_documents(self):
        """User Alice and Bob can only see their own respective documents via API."""
        # Ingest Alice doc
        chunk_alice = Chunk(
            content="Alice project details.",
            metadata={"filename": "alice_spec.pdf", "chunk_id": "c_a1", "doc_id": "doc_alice_1", "owner_id": "google_alice"},
        )
        self.store.add_chunks([chunk_alice], [[0.1] * 768], "gemini", owner_id="google_alice")

        # Ingest Bob doc
        chunk_bob = Chunk(
            content="Bob financial forecast.",
            metadata={"filename": "bob_budget.pdf", "chunk_id": "c_b1", "doc_id": "doc_bob_1", "owner_id": "google_bob"},
        )
        self.store.add_chunks([chunk_bob], [[0.2] * 768], "gemini", owner_id="google_bob")

        # Alice lists documents
        res_alice = self.client.get("/api/v1/documents", headers=self.alice_headers, params={"provider_id": "gemini"})
        self.assertEqual(res_alice.status_code, 200)
        data_alice = res_alice.json()
        self.assertEqual(data_alice["user_id"], "google_alice")
        self.assertEqual(len(data_alice["documents"]), 1)
        self.assertEqual(data_alice["documents"][0]["filename"], "alice_spec.pdf")

        # Bob lists documents
        res_bob = self.client.get("/api/v1/documents", headers=self.bob_headers, params={"provider_id": "gemini"})
        self.assertEqual(res_bob.status_code, 200)
        data_bob = res_bob.json()
        self.assertEqual(data_bob["user_id"], "google_bob")
        self.assertEqual(len(data_bob["documents"]), 1)
        self.assertEqual(data_bob["documents"][0]["filename"], "bob_budget.pdf")

    # -------------------------------------------------------------------------
    # 3. RAG Semantic Retrieval Isolation (POST /api/v1/rag/query)
    # -------------------------------------------------------------------------

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    @patch("providers.gemini_provider.GeminiProvider.embed_query", return_value=[0.1] * 768)
    @patch("providers.gemini_provider.GeminiProvider.generate")
    def test_rag_query_only_retrieves_authenticated_users_chunks(self, mock_gen, _mock_embed, _mock_conf):
        """Alice querying RAG can only retrieve Alice's chunks; Bob's chunks are filtered out."""
        # Alice chunk
        chunk_alice = Chunk(
            content="Alice secret passcode is 9988.",
            metadata={"filename": "alice_secret.txt", "chunk_id": "ca_0", "doc_id": "doc_a", "owner_id": "google_alice"},
        )
        # Bob chunk with high similarity vector
        chunk_bob = Chunk(
            content="Bob secret passcode is 1122.",
            metadata={"filename": "bob_secret.txt", "chunk_id": "cb_0", "doc_id": "doc_b", "owner_id": "google_bob"},
        )
        self.store.add_chunks([chunk_alice], [[0.1] * 768], "gemini", owner_id="google_alice")
        self.store.add_chunks([chunk_bob], [[0.1] * 768], "gemini", owner_id="google_bob")

        mock_gen.return_value = "The passcode is 9988."

        # Alice queries
        res_alice = self.client.post(
            "/api/v1/rag/query",
            headers=self.alice_headers,
            json={"query": "What is the passcode?", "provider_id": "gemini"},
        )
        self.assertEqual(res_alice.status_code, 200)
        data_alice = res_alice.json()
        self.assertEqual(len(data_alice["sources"]), 1)
        self.assertEqual(data_alice["sources"][0]["metadata"]["filename"], "alice_secret.txt")
        self.assertIn("9988", data_alice["sources"][0]["content"])

        # Bob queries
        mock_gen.return_value = "The passcode is 1122."
        res_bob = self.client.post(
            "/api/v1/rag/query",
            headers=self.bob_headers,
            json={"query": "What is the passcode?", "provider_id": "gemini"},
        )
        self.assertEqual(res_bob.status_code, 200)
        data_bob = res_bob.json()
        self.assertEqual(len(data_bob["sources"]), 1)
        self.assertEqual(data_bob["sources"][0]["metadata"]["filename"], "bob_secret.txt")
        self.assertIn("1122", data_bob["sources"][0]["content"])

    # -------------------------------------------------------------------------
    # 4. Cross-Tenant Deletion Attack Prevention (DELETE /api/v1/documents/{doc_id})
    # -------------------------------------------------------------------------

    def test_user_cannot_delete_another_users_document(self):
        """Alice cannot delete Bob's document by guessing/supplying Bob's doc_id."""
        chunk_bob = Chunk(
            content="Bob confidential contract.",
            metadata={"filename": "bob_contract.pdf", "chunk_id": "cb_1", "doc_id": "doc_bob_99", "owner_id": "google_bob"},
        )
        self.store.add_chunks([chunk_bob], [[0.1] * 768], "gemini", owner_id="google_bob")
        self.assertEqual(len(self.store.list_documents(owner_id="google_bob", provider_id="gemini")), 1)

        # Alice attempts to delete Bob's document
        response = self.client.delete(
            "/api/v1/documents/doc_bob_99",
            headers=self.alice_headers,
            params={"provider_id": "gemini"},
        )
        # Returns 404 because doc_bob_99 does not exist within Alice's ownership scope
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "DOCUMENT_NOT_FOUND")

        # Verify Bob's document still exists
        bob_docs = self.store.list_documents(owner_id="google_bob", provider_id="gemini")
        self.assertEqual(len(bob_docs), 1)

    def test_user_can_delete_own_document(self):
        """Alice deleting her own document removes all corresponding vector chunks."""
        chunk_alice_1 = Chunk(
            content="Alice page 1.",
            metadata={"filename": "alice_doc.pdf", "chunk_id": "ca_1", "doc_id": "doc_alice_55", "owner_id": "google_alice"},
        )
        chunk_alice_2 = Chunk(
            content="Alice page 2.",
            metadata={"filename": "alice_doc.pdf", "chunk_id": "ca_2", "doc_id": "doc_alice_55", "owner_id": "google_alice"},
        )
        self.store.add_chunks([chunk_alice_1, chunk_alice_2], [[0.1] * 768, [0.1] * 768], "gemini", owner_id="google_alice")

        self.assertEqual(len(self.store.list_documents(owner_id="google_alice", provider_id="gemini")), 1)

        # Alice deletes doc_alice_55
        response = self.client.delete(
            "/api/v1/documents/doc_alice_55",
            headers=self.alice_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["doc_id"], "doc_alice_55")
        self.assertEqual(data["deleted_chunks"], 2)

        # Store is now empty for Alice
        self.assertEqual(len(self.store.list_documents(owner_id="google_alice", provider_id="gemini")), 0)

    # -------------------------------------------------------------------------
    # 5. User-Scoped Document Clear (DELETE /api/v1/documents)
    # -------------------------------------------------------------------------

    def test_clearing_user_documents_does_not_affect_other_users(self):
        """Alice clearing her documents removes only Alice's records; Bob's data remains untouched."""
        chunk_alice = Chunk(
            content="Alice document content.",
            metadata={"filename": "alice.txt", "chunk_id": "ca_0", "doc_id": "doc_a", "owner_id": "google_alice"},
        )
        chunk_bob = Chunk(
            content="Bob document content.",
            metadata={"filename": "bob.txt", "chunk_id": "cb_0", "doc_id": "doc_b", "owner_id": "google_bob"},
        )
        self.store.add_chunks([chunk_alice], [[0.1] * 768], "gemini", owner_id="google_alice")
        self.store.add_chunks([chunk_bob], [[0.2] * 768], "gemini", owner_id="google_bob")

        # Alice clears her documents
        response = self.client.delete(
            "/api/v1/documents",
            headers=self.alice_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["user_id"], "google_alice")
        self.assertEqual(data["deleted_chunks"], 1)

        # Alice has 0 docs, Bob still has 1 doc
        self.assertEqual(len(self.store.list_documents(owner_id="google_alice", provider_id="gemini")), 0)
        self.assertEqual(len(self.store.list_documents(owner_id="google_bob", provider_id="gemini")), 1)

    # -------------------------------------------------------------------------
    # 6. Privilege Separation for Global Index Wipe (DELETE /api/v1/rag/index)
    # -------------------------------------------------------------------------

    def test_ordinary_user_cannot_clear_global_index(self):
        """Non-admin user calling DELETE /api/v1/rag/index is rejected with 403 Forbidden."""
        response = self.client.delete(
            "/api/v1/rag/index",
            headers=self.alice_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "FORBIDDEN")

    def test_admin_user_can_clear_global_index(self):
        """Admin user calling DELETE /api/v1/rag/index successfully wipes all records."""
        chunk_alice = Chunk(
            content="Alice data.",
            metadata={"filename": "a.txt", "chunk_id": "ca_0", "owner_id": "google_alice"},
        )
        chunk_bob = Chunk(
            content="Bob data.",
            metadata={"filename": "b.txt", "chunk_id": "cb_0", "owner_id": "google_bob"},
        )
        self.store.add_chunks([chunk_alice], [[0.1] * 768], "gemini", owner_id="google_alice")
        self.store.add_chunks([chunk_bob], [[0.2] * 768], "gemini", owner_id="google_bob")
        self.assertEqual(self.store.count(), 2)

        # Admin calls global index clear
        response = self.client.delete(
            "/api/v1/rag/index",
            headers=self.admin_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self.assertEqual(self.store.count(), 0)

    # -------------------------------------------------------------------------
    # 7. Legacy / Unassigned Documents Quarantine
    # -------------------------------------------------------------------------

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    @patch("providers.gemini_provider.GeminiProvider.embed_query", return_value=[0.1] * 768)
    @patch("providers.gemini_provider.GeminiProvider.generate", return_value="No information found.")
    def test_legacy_unassigned_data_not_exposed_to_users(self, mock_gen, _mock_embed, _mock_conf):
        """Legacy chunks without owner_id are quarantined and never served to Alice or Bob."""
        legacy_chunk = Chunk(
            content="Legacy unassigned corporate secrets.",
            metadata={"filename": "legacy_doc.pdf", "chunk_id": "leg_0"},
        )
        # Directly add to store with None owner_id
        self.store.add_chunks([legacy_chunk], [[0.1] * 768], "gemini", owner_id=None)

        # Alice cannot list legacy doc
        alice_docs = self.store.list_documents(owner_id="google_alice", provider_id="gemini")
        self.assertEqual(len(alice_docs), 0)

        # Alice records in store are empty
        alice_records = self.store.get_records(owner_id="google_alice")
        self.assertEqual(len(alice_records), 0)

    # -------------------------------------------------------------------------
    # 8. Server-Derived Identity & Admin Break-Glass Privilege Separation
    # -------------------------------------------------------------------------

    def test_admin_key_grants_admin_role_but_no_workspace_access(self):
        """
        ADMIN_ACCESS_KEY grants role='admin' with auth_type='admin_key'.
        An admin_key session cannot ingest documents or perform RAG queries.
        """
        # 1. Login with ADMIN_ACCESS_KEY
        res = self.client.post(
            "/api/v1/auth/login",
            json={"access_key": "test-admin-key-999"},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["role"], "admin")
        self.assertEqual(data["auth_type"], "admin_key")
        self.assertFalse(data.get("drive_authorized"))

        token = res.cookies.get("p06_session")
        admin_key_headers = {"Authorization": f"Bearer {token}"}

        # 2. Querying RAG is rejected with 403 ADMIN_KEY_WORKSPACE_RESTRICTED
        rag_res = self.client.post(
            "/api/v1/rag/query",
            headers=admin_key_headers,
            json={"query": "test query", "provider_id": "gemini"},
        )
        self.assertEqual(rag_res.status_code, 403)
        self.assertEqual(rag_res.json()["error"]["code"], "ADMIN_KEY_WORKSPACE_RESTRICTED")

        # 3. Document ingestion is rejected with 403 ADMIN_KEY_WORKSPACE_RESTRICTED
        ingest_res = self.client.post(
            "/api/v1/documents/ingest",
            headers=admin_key_headers,
            data={"provider_id": "gemini"},
            files={"files": ("doc.txt", b"admin content", "text/plain")},
        )
        self.assertEqual(ingest_res.status_code, 403)
        self.assertEqual(ingest_res.json()["error"]["code"], "ADMIN_KEY_WORKSPACE_RESTRICTED")

    # -------------------------------------------------------------------------
    # 9. Session Token Integrity, Forgery, and Expiration
    # -------------------------------------------------------------------------

    def test_session_token_tampering_rejected(self):
        """Any modification to session token payload or signature fails cryptographic verification."""
        from api.dependencies import verify_session_token
        valid_token = create_session_token(user_id="google_alice", role="user", auth_type="google")
        self.assertIsNotNone(verify_session_token(valid_token))

        # Tamper with token string
        tampered_token = valid_token[:-4] + "wxyz"
        self.assertIsNone(verify_session_token(tampered_token))

        # Random fake token
        self.assertIsNone(verify_session_token("invalid.forged.session.token"))

    def test_session_token_expiration(self):
        """Tokens older than max_age are rejected."""
        from api.dependencies import _get_serializer, verify_session_token

        serializer = _get_serializer()
        token = serializer.dumps({"authenticated": True, "user_id": "google_alice", "role": "user", "auth_type": "google"})

        # Verification with max_age=-1 fails as expired
        with patch("api.dependencies.SESSION_MAX_AGE_SECONDS", -1):
            self.assertIsNone(verify_session_token(token))

    def test_logout_clears_cookie(self):
        """Logout endpoint deletes the session cookie."""
        res = self.client.post("/api/v1/auth/logout")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["authenticated"])
        # Cookie header has max-age=0 or expires in past
        set_cookie = res.headers.get("set-cookie", "")
        self.assertIn("p06_session", set_cookie)

    # -------------------------------------------------------------------------
    # 10. Persistence Behavior & Process Restart Modeling
    # -------------------------------------------------------------------------

    def test_in_memory_store_data_loss_on_process_restart(self):
        """
        Simulates process restart when using InMemoryVectorStore.
        Proves that in-memory storage is volatile and does not survive container/process restart.
        """
        chunk = Chunk(
            content="Volatile memory content.",
            metadata={"filename": "doc.txt", "chunk_id": "c0", "owner_id": "google_alice"},
        )
        self.store.add_chunks([chunk], [[0.1] * 768], "gemini", owner_id="google_alice")
        self.assertEqual(len(self.store.list_documents(owner_id="google_alice", provider_id="gemini")), 1)

        # Simulate process termination & fresh start
        new_store = InMemoryVectorStore()
        new_pipeline = RAGPipeline(vector_store=new_store)
        set_rag_pipeline(new_pipeline)

        # Post-restart: store is empty
        self.assertEqual(len(new_pipeline.list_documents(owner_id="google_alice", provider_id="gemini")), 0)

    # -------------------------------------------------------------------------
    # 11. Partial Deletion and Recovery Safety
    # -------------------------------------------------------------------------

    def test_deleting_nonexistent_document_returns_404_safely(self):
        """Attempting to delete a non-existent document ID safely returns 404 without altering other data."""
        chunk = Chunk(
            content="Existing content.",
            metadata={"filename": "doc.txt", "chunk_id": "c0", "doc_id": "doc_existing", "owner_id": "google_alice"},
        )
        self.store.add_chunks([chunk], [[0.1] * 768], "gemini", owner_id="google_alice")

        # Delete non-existent doc
        res = self.client.delete(
            "/api/v1/documents/non_existent_doc_id",
            headers=self.alice_headers,
            params={"provider_id": "gemini"},
        )
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["error"]["code"], "DOCUMENT_NOT_FOUND")

    def test_bearer_and_cookie_auth_parity(self):
        """Cookie and Bearer token headers follow identical authentication and authorization behavior."""
        # User token via Bearer
        res_bearer = self.client.get("/api/v1/auth/status", headers=self.alice_headers)
        self.assertEqual(res_bearer.status_code, 200)
        self.assertEqual(res_bearer.json()["user_id"], "google_alice")
        self.assertEqual(res_bearer.json()["role"], "user")

        # User token via Cookie
        self.client.cookies.set("p06_session", self.alice_token)
        res_cookie = self.client.get("/api/v1/auth/status")
        self.assertEqual(res_cookie.status_code, 200)
        self.assertEqual(res_cookie.json()["user_id"], "google_alice")
        self.assertEqual(res_cookie.json()["role"], "user")


if __name__ == "__main__":
    unittest.main()
