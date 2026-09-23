"""
Unit tests for Semantic Retriever & Cosine Similarity.
"""

import unittest
from unittest.mock import MagicMock
from rag.retriever import SemanticRetriever
from rag.vector_store import InMemoryVectorStore
from rag.chunker import Chunk
from providers.base import BaseAIProvider


class MockProvider(BaseAIProvider):
    """Mock Provider for unit testing retriever."""

    def __init__(self, provider_id="gemini", name="Google Gemini", dim=768):
        super().__init__(name=name, provider_id=provider_id)
        self.dim = dim

    def is_configured(self) -> bool:
        return True

    def get_embedding_model_name(self) -> str:
        if self.provider_id == "gemini":
            return "gemini-embedding-2"
        elif self.provider_id == "nvidia_nim":
            return "nvidia/llama-nemotron-embed-1b-v2"
        return "text-embedding-3-small"

    def get_chat_model_name(self) -> str:
        return "gemini-2.5-flash"

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, query: str):
        return [0.1] * self.dim

    def generate(self, prompt: str, system_prompt=None) -> str:
        return "Answer"


class TestSemanticRetriever(unittest.TestCase):
    """Test suite verifying Cosine Similarity and retrieval mechanics."""

    def setUp(self):
        self.store = InMemoryVectorStore()
        self.retriever = SemanticRetriever(self.store)

    def test_cosine_similarity_identical_vectors(self):
        """Identical vectors should yield similarity score of 1.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        sim = SemanticRetriever.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Orthogonal vectors should yield similarity score of 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = SemanticRetriever.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 0.0, places=4)

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should yield 0.0 without division by zero crash."""
        vec1 = [0.0, 0.0]
        vec2 = [1.0, 2.0]
        sim = SemanticRetriever.cosine_similarity(vec1, vec2)
        self.assertEqual(sim, 0.0)

    def test_retrieve_matching_provider(self):
        """Retrieval with matching provider returns top-k chunks above threshold."""
        provider = MockProvider(provider_id="gemini", dim=768)
        chunks = [
            Chunk(content="Important policy detail", metadata={"filename": "doc.txt", "chunk_id": "c1"}),
            Chunk(content="Another enterprise clause", metadata={"filename": "doc.txt", "chunk_id": "c2"})
        ]
        self.store.add_chunks(chunks, [[0.1] * 768, [0.1] * 768], provider_id="gemini")

        results = self.retriever.retrieve(
            query="policy detail",
            provider=provider,
            top_k=5,
            similarity_threshold=0.5
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_retrieve_mismatched_provider_raises_error(self):
        """Retrieval with a provider different from active index provider must be rejected."""
        gemini_provider = MockProvider(provider_id="gemini", dim=768)
        nvidia_provider = MockProvider(provider_id="nvidia_nim", name="NVIDIA NIM", dim=2048)

        chunks = [Chunk(content="Sample content", metadata={"filename": "doc.txt", "chunk_id": "c1"})]
        self.store.add_chunks(chunks, [[0.1] * 768], provider_id="gemini")

        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve(
                query="Sample query",
                provider=nvidia_provider,
                top_k=5
            )
        self.assertIn("Provider mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
