"""
Unit tests for Semantic Retriever & Cosine Similarity.
"""

import unittest
from rag.retriever import SemanticRetriever
from rag.vector_store import InMemoryVectorStore


class TestSemanticRetriever(unittest.TestCase):
    """Test suite verifying Cosine Similarity mathematical calculations."""

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


if __name__ == "__main__":
    unittest.main()
