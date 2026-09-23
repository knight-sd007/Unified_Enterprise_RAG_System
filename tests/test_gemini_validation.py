"""
Unit tests for Gemini Provider Vector Validation (ISSUE-01).
"""

import unittest
from unittest.mock import patch, MagicMock
from providers.gemini_provider import GeminiProvider


class TestGeminiProviderValidation(unittest.TestCase):
    """Test suite verifying strict validation of Gemini embeddings."""

    def setUp(self):
        self.provider = GeminiProvider()

    def test_validate_valid_768_dim_vector(self):
        """Valid 768-dimensional float vector should pass validation."""
        valid_vec = [0.01 * (i % 10) for i in range(768)]
        result = GeminiProvider._validate_vector(valid_vec, expected_dim=768)
        self.assertEqual(len(result), 768)
        self.assertIsInstance(result[0], float)

    def test_reject_empty_vector(self):
        """Empty vector should raise RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            GeminiProvider._validate_vector([], expected_dim=768)
        self.assertIn("Empty vector received", str(ctx.exception))

    def test_reject_wrong_dimension(self):
        """Vector with dimension 512 instead of 768 should raise RuntimeError."""
        wrong_dim_vec = [0.1] * 512
        with self.assertRaises(RuntimeError) as ctx:
            GeminiProvider._validate_vector(wrong_dim_vec, expected_dim=768)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_reject_non_numeric_elements(self):
        """Vector containing strings or non-numeric items should raise RuntimeError."""
        malformed_vec = [0.1] * 767 + ["invalid_string"]
        with self.assertRaises(RuntimeError) as ctx:
            GeminiProvider._validate_vector(malformed_vec, expected_dim=768)
        self.assertIn("Non-numeric value", str(ctx.exception))

    def test_reject_boolean_elements(self):
        """Vector containing booleans should raise RuntimeError."""
        malformed_vec = [0.1] * 767 + [True]
        with self.assertRaises(RuntimeError) as ctx:
            GeminiProvider._validate_vector(malformed_vec, expected_dim=768)
        self.assertIn("Non-numeric value", str(ctx.exception))

    def test_reject_non_finite_elements(self):
        """Vector containing NaN or Inf should raise RuntimeError."""
        malformed_vec = [0.1] * 767 + [float("nan")]
        with self.assertRaises(RuntimeError) as ctx:
            GeminiProvider._validate_vector(malformed_vec, expected_dim=768)
        self.assertIn("Non-finite value", str(ctx.exception))

    @patch("config.settings.Config.is_gemini_configured", return_value=True)
    @patch("config.settings.Config.get_gemini_api_key", return_value="fake-gemini-key")
    def test_embed_documents_rejects_empty_api_embedding(self, mock_key, mock_cfg):
        """When Gemini API returns an empty or missing embedding response, a sanitized error is raised."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            # Simulate empty response without embedding values
            mock_response = MagicMock()
            mock_response.embedding = None
            mock_response.embeddings = []
            mock_client.models.embed_content.return_value = mock_response

            with self.assertRaises(RuntimeError) as ctx:
                self.provider.embed_documents(["Test content"])
            self.assertIn("Gemini Embedding Error", str(ctx.exception))

    @patch("config.settings.Config.is_gemini_configured", return_value=True)
    @patch("config.settings.Config.get_gemini_api_key", return_value="fake-gemini-key")
    def test_embed_documents_success_with_768_dim(self, mock_key, mock_cfg):
        """When Gemini API returns valid 768-dim vector, it is successfully returned and passes EmbedContentConfig."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.embedding = MagicMock(values=[0.05] * 768)
            mock_response.embeddings = None
            mock_client.models.embed_content.return_value = mock_response

            embeddings = self.provider.embed_documents(["Test content"])
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(len(embeddings[0]), 768)

            # Verify call arguments
            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args[1]
            self.assertEqual(call_kwargs["model"], "gemini-embedding-2")
            self.assertEqual(call_kwargs["contents"], "title: none | text: Test content")
            config = call_kwargs["config"]
            self.assertEqual(config.output_dimensionality, 768)
            self.assertFalse(hasattr(config, "task_type") and config.task_type is not None)

    @patch("config.settings.Config.is_gemini_configured", return_value=True)
    @patch("config.settings.Config.get_gemini_api_key", return_value="fake-gemini-key")
    def test_embed_query_success_with_768_dim_and_task_format(self, mock_key, mock_cfg):
        """embed_query formats query with 'task: question answering | query:' and does not pass task_type."""
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_response = MagicMock()
            mock_response.embedding = MagicMock(values=[0.05] * 768)
            mock_response.embeddings = None
            mock_client.models.embed_content.return_value = mock_response

            query_vec = self.provider.embed_query("What is enterprise RAG?")
            self.assertEqual(len(query_vec), 768)

            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args[1]
            self.assertEqual(call_kwargs["model"], "gemini-embedding-2")
            self.assertEqual(call_kwargs["contents"], "task: question answering | query: What is enterprise RAG?")
            config = call_kwargs["config"]
            self.assertEqual(config.output_dimensionality, 768)
            self.assertFalse(hasattr(config, "task_type") and config.task_type is not None)


if __name__ == "__main__":
    unittest.main()
