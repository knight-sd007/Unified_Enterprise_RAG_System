"""
Unit tests for Qdrant Cloud Vector Store Integration.
"""

import unittest
from unittest.mock import MagicMock, patch
from rag.chunker import Chunk
from rag.vector_store import QdrantVectorStore, generate_point_id
from config.settings import Config


class TestQdrantVectorStore(unittest.TestCase):
    """Test suite verifying QdrantVectorStore with mocked Qdrant client."""

    def setUp(self):
        self.mock_client = MagicMock()
        # Default mock: collection exists
        self.mock_client.collection_exists.return_value = True
        self.store = QdrantVectorStore(client=self.mock_client)

    def test_missing_config_raises_sanitized_error(self):
        """When Qdrant credentials are not configured, attempting to connect raises a clear error."""
        with patch("config.settings.Config.is_qdrant_configured", return_value=False):
            store = QdrantVectorStore(client=None)
            with self.assertRaises(RuntimeError) as ctx:
                store._get_client()
            self.assertIn("Qdrant Cloud is not configured", str(ctx.exception))

    def test_absent_collection_raises_actionable_error(self):
        """When collection is not provisioned on Qdrant, fail clearly without auto-creating."""
        self.mock_client.collection_exists.return_value = False
        chunks = [Chunk(content="Enterprise text", metadata={"filename": "doc.txt", "chunk_id": "c0"})]
        embeddings = [[0.05] * 768]

        with self.assertRaises(RuntimeError) as ctx:
            self.store.add_chunks(chunks, embeddings, provider_id="gemini")

        self.assertIn("p06_gemini_embedding_2_768", str(ctx.exception))
        self.assertIn("does not exist on cluster", str(ctx.exception))
        # Ensure create_collection was NOT called
        self.assertFalse(hasattr(self.mock_client, "create_collection") and self.mock_client.create_collection.called)

    def test_add_chunks_upserts_to_correct_collection(self):
        """Ingestion must route to the provider-specific collection using deterministic UUIDv5."""
        chunks = [
            Chunk(content="Chunk A", metadata={"filename": "doc.txt", "chunk_id": "c1"}),
            Chunk(content="Chunk B", metadata={"filename": "doc.txt", "chunk_id": "c2"})
        ]
        embeddings = [[0.01] * 768, [0.02] * 768]

        count = self.store.add_chunks(chunks, embeddings, provider_id="gemini")
        self.assertEqual(count, 2)

        # Verify upsert called with correct collection name
        self.mock_client.upsert.assert_called_once()
        call_kwargs = self.mock_client.upsert.call_args[1]
        self.assertEqual(call_kwargs["collection_name"], "p06_gemini_embedding_2_768")
        points = call_kwargs["points"]
        self.assertEqual(len(points), 2)
        # Point ID must be deterministic UUIDv5
        expected_id = generate_point_id("gemini", "doc.txt", "c1", "Chunk A")
        self.assertEqual(points[0].id, expected_id)

    def test_nvidia_routing_and_dimension(self):
        """NVIDIA provider routes to 2048-dim collection p06_nvidia_llama_nemotron_embed_1b_v2_2048."""
        chunks = [Chunk(content="NVIDIA chunk", metadata={"filename": "gpu.txt", "chunk_id": "c0"})]
        embeddings = [[0.03] * 2048]

        count = self.store.add_chunks(chunks, embeddings, provider_id="nvidia_nim")
        self.assertEqual(count, 1)

        call_kwargs = self.mock_client.upsert.call_args[1]
        self.assertEqual(call_kwargs["collection_name"], "p06_nvidia_llama_nemotron_embed_1b_v2_2048")

    def test_provider_mismatch_rejection_on_ingestion(self):
        """Switching providers without clearing raises ValueError."""
        chunks1 = [Chunk(content="Chunk A", metadata={"filename": "doc.txt", "chunk_id": "c1"})]
        self.store.add_chunks(chunks1, [[0.01] * 768], provider_id="gemini")

        chunks2 = [Chunk(content="Chunk B", metadata={"filename": "doc.txt", "chunk_id": "c2"})]
        with self.assertRaises(ValueError) as ctx:
            self.store.add_chunks(chunks2, [[0.02] * 2048], provider_id="nvidia_nim")
        self.assertIn("Provider mismatch", str(ctx.exception))

    def test_search_retrieves_scored_chunks(self):
        """Search query delegates to query_points and formats results."""
        self.store._active_provider_id = "gemini"
        self.store._embedding_dimension = 768

        mock_point = MagicMock()
        mock_point.id = "uuid-123"
        mock_point.score = 0.8876
        mock_point.payload = {
            "content": "Result content",
            "metadata": {"filename": "file.pdf"},
            "chunk_id": "c0"
        }
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        self.mock_client.query_points.return_value = mock_response

        results = self.store.search(
            query_vector=[0.05] * 768,
            provider_id="gemini",
            top_k=3,
            similarity_threshold=0.25
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "Result content")
        self.assertEqual(results[0]["score"], 0.8876)
        self.assertEqual(results[0]["chunk_id"], "c0")

    def test_openai_collection_not_required_for_gemini(self):
        """Absence of OpenAI collection does not affect Gemini or NVIDIA functionality."""
        # Define side effect: Gemini collection exists, OpenAI does not
        def collection_exists_side_effect(collection_name):
            return collection_name != "p06_openai_text_embedding_3_small"

        self.mock_client.collection_exists.side_effect = collection_exists_side_effect

        chunks = [Chunk(content="Gemini chunk", metadata={"filename": "test.txt", "chunk_id": "c1"})]
        count = self.store.add_chunks(chunks, [[0.01] * 768], provider_id="gemini")
        self.assertEqual(count, 1)

    def test_get_stats_gemini_routing(self):
        """get_stats with provider_id='gemini' queries p06_gemini_embedding_2_768 and returns 768 dims."""
        mock_count = MagicMock()
        mock_count.count = 42
        self.mock_client.count.return_value = mock_count

        stats = self.store.get_stats(provider_id="gemini")
        self.assertEqual(stats["count"], 42)
        self.assertEqual(stats["dimension"], 768)
        self.assertEqual(stats["provider_id"], "gemini")
        self.assertEqual(stats["collection_name"], "p06_gemini_embedding_2_768")
        self.assertEqual(stats["status"], "Indexed")
        self.mock_client.count.assert_called_with(collection_name="p06_gemini_embedding_2_768")

    def test_get_stats_nvidia_routing(self):
        """get_stats with provider_id='nvidia_nim' queries p06_nvidia_llama_nemotron_embed_1b_v2_2048 and returns 2048 dims."""
        mock_count = MagicMock()
        mock_count.count = 100
        self.mock_client.count.return_value = mock_count

        stats = self.store.get_stats(provider_id="nvidia_nim")
        self.assertEqual(stats["count"], 100)
        self.assertEqual(stats["dimension"], 2048)
        self.assertEqual(stats["provider_id"], "nvidia_nim")
        self.assertEqual(stats["collection_name"], "p06_nvidia_llama_nemotron_embed_1b_v2_2048")
        self.assertEqual(stats["status"], "Indexed")
        self.mock_client.count.assert_called_with(collection_name="p06_nvidia_llama_nemotron_embed_1b_v2_2048")

    def test_get_stats_no_provider_fallback(self):
        """get_stats without provider_id and without active provider returns safe Ready defaults."""
        stats = self.store.get_stats()
        self.assertEqual(stats["count"], 0)
        self.assertEqual(stats["dimension"], "N/A")
        self.assertEqual(stats["provider_id"], "None")
        self.assertEqual(stats["collection_name"], "N/A")
        self.assertEqual(stats["status"], "Ready")

    def test_clear_store_gemini_deletes_points_in_qdrant(self):
        """Clearing Gemini provider must call client.delete on p06_gemini_embedding_2_768 with filter selector."""
        self.store.clear_store(provider_id="gemini")

        self.mock_client.delete.assert_called_once()
        call_kwargs = self.mock_client.delete.call_args[1]
        self.assertEqual(call_kwargs["collection_name"], "p06_gemini_embedding_2_768")
        self.assertTrue(call_kwargs.get("wait", False))
        self.assertIsNotNone(call_kwargs.get("points_selector"))

    def test_clear_store_provider_isolation(self):
        """Clearing Gemini must target ONLY p06_gemini_embedding_2_768 and NOT other collections."""
        self.store.clear_store(provider_id="gemini")

        call_kwargs = self.mock_client.delete.call_args[1]
        self.assertEqual(call_kwargs["collection_name"], "p06_gemini_embedding_2_768")
        self.assertNotEqual(call_kwargs["collection_name"], "p06_nvidia_llama_nemotron_embed_1b_v2_2048")
        self.assertNotEqual(call_kwargs["collection_name"], "p06_openai_text_embedding_3_small")

    def test_clear_store_resets_active_provider_and_dimension(self):
        """Clearing store must reset _active_provider_id and _embedding_dimension to None."""
        self.store._active_provider_id = "gemini"
        self.store._embedding_dimension = 768

        self.store.clear_store(provider_id="gemini")
        self.assertIsNone(self.store.get_active_provider_id())
        self.assertIsNone(self.store.get_embedding_dimension())

    def test_clear_store_no_provider_safe_return(self):
        """When no provider is specified and none active, clear_store must not perform destructive delete."""
        self.store._active_provider_id = None
        self.store.clear_store(provider_id=None)

        self.mock_client.delete.assert_not_called()
        self.assertIsNone(self.store.get_active_provider_id())
        self.assertIsNone(self.store.get_embedding_dimension())


if __name__ == "__main__":
    unittest.main()
