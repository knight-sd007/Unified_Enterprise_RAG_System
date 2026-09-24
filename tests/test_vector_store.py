"""
Unit tests for In-Memory Vector Store, Provider Isolation, and Deduplication.
"""

import unittest
from rag.chunker import Chunk
from rag.vector_store import InMemoryVectorStore, generate_point_id


class TestInMemoryVectorStore(unittest.TestCase):
    """Test suite verifying in-memory vector storage, provider isolation, and deduplication."""

    def setUp(self):
        self.store = InMemoryVectorStore()

    def test_add_chunks_and_count(self):
        """Storing chunks should increase vector record count."""
        chunks = [
            Chunk(content="Chunk 1", metadata={"filename": "doc1.txt", "chunk_id": "c1"}),
            Chunk(content="Chunk 2", metadata={"filename": "doc1.txt", "chunk_id": "c2"})
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        added = self.store.add_chunks(chunks, embeddings, provider_id="test_provider")
        self.assertEqual(added, 2)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(self.store.get_embedding_dimension(), 3)

    def test_dimension_mismatch_error(self):
        """Adding vectors of different dimension should raise ValueError."""
        chunks1 = [Chunk(content="A", metadata={"filename": "doc1.txt", "chunk_id": "c1"})]
        embeds1 = [[0.1, 0.2, 0.3]]
        self.store.add_chunks(chunks1, embeds1, provider_id="custom_p1")

        chunks2 = [Chunk(content="B", metadata={"filename": "doc2.txt", "chunk_id": "c2"})]
        embeds2 = [[0.1, 0.2]]  # Dim 2 instead of Dim 3

        with self.assertRaises(ValueError) as ctx:
            self.store.add_chunks(chunks2, embeds2, provider_id="custom_p1")
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_gemini_dimension_mismatch(self):
        """Adding non-768 vectors for gemini provider should raise ValueError."""
        chunks = [Chunk(content="A", metadata={"filename": "doc1.txt", "chunk_id": "c1"})]
        embeds = [[0.1] * 512]
        with self.assertRaises(ValueError) as ctx:
            self.store.add_chunks(chunks, embeds, provider_id="gemini")
        self.assertIn("Expected dimension 768", str(ctx.exception))

    def test_nvidia_dimension_mismatch(self):
        """Adding non-2048 vectors for nvidia_nim provider should raise ValueError."""
        chunks = [Chunk(content="A", metadata={"filename": "doc1.txt", "chunk_id": "c1"})]
        embeds = [[0.1] * 1024]
        with self.assertRaises(ValueError) as ctx:
            self.store.add_chunks(chunks, embeds, provider_id="nvidia_nim")
        self.assertIn("Expected dimension 2048", str(ctx.exception))

    def test_provider_mismatch_error(self):
        """Adding vectors with a different provider without clearing should raise ValueError."""
        chunks1 = [Chunk(content="A", metadata={"filename": "doc1.txt", "chunk_id": "c1"})]
        embeds1 = [[0.1] * 768]
        self.store.add_chunks(chunks1, embeds1, provider_id="gemini")

        chunks2 = [Chunk(content="B", metadata={"filename": "doc2.txt", "chunk_id": "c2"})]
        embeds2 = [[0.2] * 2048]

        with self.assertRaises(ValueError) as ctx:
            self.store.add_chunks(chunks2, embeds2, provider_id="nvidia_nim")
        self.assertIn("Provider mismatch", str(ctx.exception))

    def test_clear_store(self):
        """Clearing store should remove all records and reset dimensions and provider."""
        chunks = [Chunk(content="A", metadata={"filename": "doc1.txt", "chunk_id": "c1"})]
        embeds = [[0.1, 0.2]]
        self.store.add_chunks(chunks, embeds, provider_id="p1")

        self.store.clear_store()
        self.assertEqual(self.store.count(), 0)
        self.assertIsNone(self.store.get_embedding_dimension())
        self.assertIsNone(self.store.get_active_provider_id())

    def test_idempotent_duplicate_ingestion(self):
        """Re-indexing identical chunk should not increase vector store count (ISSUE-03)."""
        chunk = Chunk(content="Unique enterprise text.", metadata={"filename": "policy.pdf", "chunk_id": "c0"})
        embed = [[0.1, 0.2, 0.3]]

        self.store.add_chunks([chunk], embed, provider_id="p1")
        self.assertEqual(self.store.count(), 1)

        # Ingest the exact same chunk a second time
        self.store.add_chunks([chunk], embed, provider_id="p1")
        self.assertEqual(self.store.count(), 1)

    def test_deterministic_point_id_generation(self):
        """Point ID generation must be deterministic and sensitive to content and metadata."""
        id1 = generate_point_id("gemini", "doc1.pdf", "chunk_0", "Hello world")
        id2 = generate_point_id("gemini", "doc1.pdf", "chunk_0", "Hello world")
        id3 = generate_point_id("gemini", "doc1.pdf", "chunk_0", "Different content")
        id4 = generate_point_id("nvidia_nim", "doc1.pdf", "chunk_0", "Hello world")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertNotEqual(id1, id4)

    def test_get_stats_signature_parity_and_provider(self):
        """get_stats accepts optional provider_id without TypeError and returns accurate metadata."""
        stats_no_param = self.store.get_stats()
        self.assertEqual(stats_no_param["count"], 0)
        self.assertEqual(stats_no_param["dimension"], "N/A")
        self.assertEqual(stats_no_param["provider_id"], "None")

        stats_with_gemini = self.store.get_stats(provider_id="gemini")
        self.assertEqual(stats_with_gemini["provider_id"], "gemini")
        self.assertEqual(stats_with_gemini["count"], 0)

        chunk = Chunk(content="Test content", metadata={"filename": "doc.txt", "chunk_id": "c0"})
        self.store.add_chunks([chunk], [[0.1] * 768], provider_id="gemini")

        stats_after_add = self.store.get_stats(provider_id="gemini")
        self.assertEqual(stats_after_add["count"], 1)
        self.assertEqual(stats_after_add["dimension"], 768)
        self.assertEqual(stats_after_add["provider_id"], "gemini")
        self.assertEqual(stats_after_add["status"], "Indexed")

    def test_in_memory_clear_store_signature_accepts_provider_id(self):
        """InMemoryVectorStore.clear_store accepts provider_id without TypeError and clears store."""
        chunk = Chunk(content="Test content", metadata={"filename": "doc.txt", "chunk_id": "c0"})
        self.store.add_chunks([chunk], [[0.1] * 768], provider_id="gemini")
        self.assertEqual(self.store.count(), 1)

        self.store.clear_store(provider_id="gemini")
        self.assertEqual(self.store.count(), 0)
        self.assertIsNone(self.store.get_embedding_dimension())
        self.assertIsNone(self.store.get_active_provider_id())


if __name__ == "__main__":
    unittest.main()
