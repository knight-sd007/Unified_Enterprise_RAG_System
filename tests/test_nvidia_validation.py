"""
Unit tests for NVIDIA NIM Provider Vector Validation and Payload Contracts.
"""

import unittest
from unittest.mock import patch, MagicMock
from providers.nvidia_nim_provider import NvidiaNimProvider


class TestNvidiaNimProviderValidation(unittest.TestCase):
    """Test suite verifying NVIDIA NIM embeddings, input_type payloads, and vector validation."""

    def setUp(self):
        self.provider = NvidiaNimProvider()

    def test_validate_valid_2048_dim_vector(self):
        """Valid 2048-dimensional float vector should pass validation."""
        valid_vec = [0.01 * (i % 10) for i in range(2048)]
        result = NvidiaNimProvider._validate_vector(valid_vec, expected_dim=2048)
        self.assertEqual(len(result), 2048)
        self.assertIsInstance(result[0], float)

    def test_reject_empty_vector(self):
        """Empty vector should raise RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            NvidiaNimProvider._validate_vector([], expected_dim=2048)
        self.assertIn("Empty vector received", str(ctx.exception))

    def test_reject_wrong_dimension(self):
        """Vector with dimension 1024 instead of 2048 should raise RuntimeError."""
        wrong_dim_vec = [0.1] * 1024
        with self.assertRaises(RuntimeError) as ctx:
            NvidiaNimProvider._validate_vector(wrong_dim_vec, expected_dim=2048)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_reject_non_numeric_elements(self):
        """Vector containing non-numeric items should raise RuntimeError."""
        malformed_vec = [0.1] * 2047 + ["invalid_string"]
        with self.assertRaises(RuntimeError) as ctx:
            NvidiaNimProvider._validate_vector(malformed_vec, expected_dim=2048)
        self.assertIn("Non-numeric value", str(ctx.exception))

    def test_reject_boolean_elements(self):
        """Vector containing booleans should raise RuntimeError."""
        malformed_vec = [0.1] * 2047 + [True]
        with self.assertRaises(RuntimeError) as ctx:
            NvidiaNimProvider._validate_vector(malformed_vec, expected_dim=2048)
        self.assertIn("Non-numeric value", str(ctx.exception))

    def test_reject_non_finite_elements(self):
        """Vector containing NaN or Inf should raise RuntimeError."""
        malformed_vec = [0.1] * 2047 + [float("nan")]
        with self.assertRaises(RuntimeError) as ctx:
            NvidiaNimProvider._validate_vector(malformed_vec, expected_dim=2048)
        self.assertIn("Non-finite value", str(ctx.exception))

    @patch("config.settings.Config.is_nvidia_configured", return_value=True)
    @patch("config.settings.Config.get_nvidia_api_key", return_value="fake-nv-key")
    def test_embed_documents_passes_passage_input_type(self, mock_key, mock_cfg):
        """embed_documents passes input_type='passage', truncate='NONE', model='nvidia/llama-nemotron-embed-1b-v2'."""
        with patch.object(self.provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            mock_item = MagicMock()
            mock_item.embedding = [0.02] * 2048
            mock_response = MagicMock()
            mock_response.data = [mock_item]
            mock_client.embeddings.create.return_value = mock_response

            embeddings = self.provider.embed_documents(["Doc text chunk"])
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(len(embeddings[0]), 2048)

            mock_client.embeddings.create.assert_called_once()
            call_kwargs = mock_client.embeddings.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "nvidia/llama-nemotron-embed-1b-v2")
            self.assertEqual(call_kwargs["input"], ["Doc text chunk"])
            self.assertEqual(call_kwargs["extra_body"], {"input_type": "passage", "truncate": "NONE"})

    @patch("config.settings.Config.is_nvidia_configured", return_value=True)
    @patch("config.settings.Config.get_nvidia_api_key", return_value="fake-nv-key")
    def test_embed_query_passes_query_input_type(self, mock_key, mock_cfg):
        """embed_query passes input_type='query', truncate='NONE', model='nvidia/llama-nemotron-embed-1b-v2'."""
        with patch.object(self.provider, "_get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            mock_item = MagicMock()
            mock_item.embedding = [0.03] * 2048
            mock_response = MagicMock()
            mock_response.data = [mock_item]
            mock_client.embeddings.create.return_value = mock_response

            query_vec = self.provider.embed_query("Search query string")
            self.assertEqual(len(query_vec), 2048)

            mock_client.embeddings.create.assert_called_once()
            call_kwargs = mock_client.embeddings.create.call_args[1]
            self.assertEqual(call_kwargs["model"], "nvidia/llama-nemotron-embed-1b-v2")
            self.assertEqual(call_kwargs["input"], ["Search query string"])
            self.assertEqual(call_kwargs["extra_body"], {"input_type": "query", "truncate": "NONE"})


if __name__ == "__main__":
    unittest.main()
