"""
Unit tests verifying SQLite metadata repository durability across simulated container restarts.
"""

import os
import tempfile
import unittest
from rag.storage.metadata_db import MetadataRepository


class TestSQLiteDurability(unittest.TestCase):
    """Verifies that SQLite schema initialization and data persist across restarts."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_persistence.db")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_survives_reinstantiation(self):
        """Metadata and encrypted tokens survive process teardown and re-instantiation."""
        # Session 1: Create and populate repository
        repo1 = MetadataRepository(db_path=self.db_path)
        repo1.save_document(
            doc_id="doc_alice_001",
            owner_id="alice",
            filename="financial_q3.pdf",
            file_size=10240,
            char_count=5000,
            provider_id="openai",
            embedding_model="text-embedding-3-small",
            chunk_count=10,
        )
        repo1.save_document(
            doc_id="doc_bob_001",
            owner_id="bob",
            filename="engineering_spec.docx",
            file_size=20480,
            char_count=9500,
            provider_id="gemini",
            embedding_model="gemini-embedding-2",
            chunk_count=20,
        )
        repo1.save_oauth_tokens("alice", "google", {"access_token": "alice_at", "refresh_token": "alice_rt"})
        del repo1

        # Session 2: Re-open the database with a brand new repository instance (simulating container restart)
        repo2 = MetadataRepository(db_path=self.db_path)

        # Re-run init_db explicitly to confirm idempotent schema creation
        repo2.init_db()

        # Verify Alice's document is intact
        alice_doc = repo2.get_document("doc_alice_001")
        self.assertIsNotNone(alice_doc)
        self.assertEqual(alice_doc["filename"], "financial_q3.pdf")
        self.assertEqual(alice_doc["owner_id"], "alice")
        self.assertEqual(alice_doc["chunk_count"], 10)

        # Verify Bob's document is intact
        bob_doc = repo2.get_document("doc_bob_001")
        self.assertIsNotNone(bob_doc)
        self.assertEqual(bob_doc["filename"], "engineering_spec.docx")
        self.assertEqual(bob_doc["owner_id"], "bob")
        self.assertEqual(bob_doc["chunk_count"], 20)

        # Verify global and user stats
        global_stats = repo2.get_global_stats()
        self.assertEqual(global_stats["total_documents"], 2)
        self.assertEqual(global_stats["total_chunks"], 30)
        self.assertEqual(global_stats["total_users"], 2)

        # Verify decrypted OAuth tokens
        alice_tokens = repo2.get_oauth_tokens("alice", "google")
        self.assertIsNotNone(alice_tokens)
        self.assertEqual(alice_tokens["access_token"], "alice_at")
        self.assertEqual(alice_tokens["refresh_token"], "alice_rt")


if __name__ == "__main__":
    unittest.main()
