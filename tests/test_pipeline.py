"""
Unit tests for RAGPipeline Telemetry, Statistics Aggregation, and Provider Forwarding.
"""

import unittest
from unittest.mock import MagicMock, patch
from rag.pipeline import RAGPipeline
from rag.vector_store import InMemoryVectorStore, QdrantVectorStore


class TestRAGPipelineTelemetry(unittest.TestCase):
    """Test suite verifying RAGPipeline.get_stats telemetry forwarding and contracts."""

    def test_pipeline_get_stats_forwards_gemini_to_qdrant_store(self):
        """pipeline.get_stats('gemini') must forward provider_id='gemini' to vector_store."""
        mock_qdrant_client = MagicMock()
        mock_count = MagicMock()
        mock_count.count = 14
        mock_qdrant_client.count.return_value = mock_count
        mock_qdrant_client.collection_exists.return_value = True

        store = QdrantVectorStore(client=mock_qdrant_client)
        pipeline = RAGPipeline(vector_store=store)

        stats = pipeline.get_stats("gemini")
        self.assertEqual(stats["provider_id"], "gemini")
        self.assertEqual(stats["dimension"], 768)
        self.assertEqual(stats["collection_name"], "p06_gemini_embedding_2_768")
        self.assertEqual(stats["count"], 14)
        mock_qdrant_client.count.assert_called_once_with(collection_name="p06_gemini_embedding_2_768")

    def test_pipeline_get_stats_forwards_nvidia_to_qdrant_store(self):
        """pipeline.get_stats('nvidia_nim') must forward provider_id='nvidia_nim' to vector_store."""
        mock_qdrant_client = MagicMock()
        mock_count = MagicMock()
        mock_count.count = 25
        mock_qdrant_client.count.return_value = mock_count
        mock_qdrant_client.collection_exists.return_value = True

        store = QdrantVectorStore(client=mock_qdrant_client)
        pipeline = RAGPipeline(vector_store=store)

        stats = pipeline.get_stats("nvidia_nim")
        self.assertEqual(stats["provider_id"], "nvidia_nim")
        self.assertEqual(stats["dimension"], 2048)
        self.assertEqual(stats["collection_name"], "p06_nvidia_llama_nemotron_embed_1b_v2_2048")
        self.assertEqual(stats["count"], 25)
        mock_qdrant_client.count.assert_called_once_with(collection_name="p06_nvidia_llama_nemotron_embed_1b_v2_2048")

    def test_pipeline_get_stats_forwards_openai_to_qdrant_store(self):
        """pipeline.get_stats('openai') must forward provider_id='openai' to vector_store."""
        mock_qdrant_client = MagicMock()
        mock_count = MagicMock()
        mock_count.count = 50
        mock_qdrant_client.count.return_value = mock_count
        mock_qdrant_client.collection_exists.return_value = True

        store = QdrantVectorStore(client=mock_qdrant_client)
        pipeline = RAGPipeline(vector_store=store)

        stats = pipeline.get_stats("openai")
        self.assertEqual(stats["provider_id"], "openai")
        self.assertEqual(stats["dimension"], 1536)
        self.assertEqual(stats["collection_name"], "p06_openai_text_embedding_3_small")
        self.assertEqual(stats["count"], 50)
        mock_qdrant_client.count.assert_called_once_with(collection_name="p06_openai_text_embedding_3_small")

    def test_pipeline_get_stats_with_in_memory_store(self):
        """pipeline.get_stats with InMemoryVectorStore gracefully handles provider_id without TypeError."""
        in_memory_store = InMemoryVectorStore()
        pipeline = RAGPipeline(vector_store=in_memory_store)

        stats = pipeline.get_stats("gemini")
        self.assertEqual(stats["provider_id"], "gemini")
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["dimension"], "N/A")
        self.assertEqual(stats["store_type"], "In-Memory NumPy Vector Index")

    def test_pipeline_get_stats_no_provider_fallback(self):
        """pipeline.get_stats() without provider_id returns safe default stats."""
        mock_qdrant_client = MagicMock()
        store = QdrantVectorStore(client=mock_qdrant_client)
        pipeline = RAGPipeline(vector_store=store)

        stats = pipeline.get_stats()
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["dimension"], "N/A")
        self.assertEqual(stats["provider_id"], "None")


if __name__ == "__main__":
    unittest.main()
