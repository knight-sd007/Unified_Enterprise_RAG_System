"""
Unit tests for In-Memory Vector Store.
"""

import unittest
from rag.chunker import Chunk
from rag.vector_store import InMemoryVectorStore


class TestInMemoryVectorStore(unittest.TestCase):
    """Test suite verifying in-memory vector storage and dimension safety."""

    def setUp(self):
        self.store = InMemoryVectorStore()

    def test_add_chunks_and_count(self):
        """Storing chunks should increase vector record count."""
        chunks = [
            Chunk(content="Chunk 1", metadata={"filename": "doc1.txt"}),
            Chunk(content="Chunk 2", metadata={"filename": "doc1.txt"})
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        added = self.store.add_chunks(chunks, embeddings, provider_id="test_provider")
        self.assertEqual(added, 2)
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(self.store.get_embedding_dimension(), 3)

    def test_dimension_mismatch_error(self):
        """Adding vectors of different dimension should raise ValueError."""
        chunks1 = [Chunk(content="A", metadata={})]
        embeds1 = [[0.1, 0.2, 0.3]]
        self.store.add_chunks(chunks1, embeds1, provider_id="p1")

        chunks2 = [Chunk(content="B", metadata={})]
        embeds2 = [[0.1, 0.2]]  # Dim 2 instead of Dim 3

        with self.assertRaises(ValueError):
            self.store.add_chunks(chunks2, embeds2, provider_id="p2")

    def test_clear_store(self):
        """Clearing store should remove all records and reset dimensions."""
        chunks = [Chunk(content="A", metadata={})]
        embeds = [[0.1, 0.2]]
        self.store.add_chunks(chunks, embeds, provider_id="p1")

        self.store.clear_store()
        self.assertEqual(self.store.count(), 0)
        self.assertIsNone(self.store.get_embedding_dimension())


if __name__ == "__main__":
    unittest.main()
