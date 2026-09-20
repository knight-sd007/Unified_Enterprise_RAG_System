"""
Unit tests for Centralized Configuration and Security Utilities.
"""

import unittest
from unittest.mock import patch
from config.settings import Config, PROVIDER_VECTOR_SPECS
from utils.security import sanitize_error_message, verify_access_key


class TestConfigurationAndSecurity(unittest.TestCase):
    """Test suite verifying settings retrieval, vector specs, and security sanitization."""

    def test_provider_vector_specs_definitions(self):
        """All 3 provider vector specifications must have correct dimensions, models, and collections."""
        gemini_spec = PROVIDER_VECTOR_SPECS["gemini"]
        self.assertEqual(gemini_spec.dimension, 768)
        self.assertEqual(gemini_spec.collection_name, "p06_gemini_text_embedding_004")
        self.assertEqual(gemini_spec.distance_metric, "Cosine")

        nvidia_spec = PROVIDER_VECTOR_SPECS["nvidia_nim"]
        self.assertEqual(nvidia_spec.dimension, 1024)
        self.assertEqual(nvidia_spec.collection_name, "p06_nvidia_nv_embedqa_e5_v5")
        self.assertEqual(nvidia_spec.distance_metric, "Cosine")

        openai_spec = PROVIDER_VECTOR_SPECS["openai"]
        self.assertEqual(openai_spec.dimension, 1536)
        self.assertEqual(openai_spec.collection_name, "p06_openai_text_embedding_3_small")
        self.assertEqual(openai_spec.distance_metric, "Cosine")

    @patch.dict("os.environ", {"QDRANT_URL": "", "QDRANT_API_KEY": ""}, clear=True)
    def test_qdrant_unconfigured_when_vars_empty(self):
        """is_qdrant_configured returns False when environment variables are empty."""
        self.assertFalse(Config.is_qdrant_configured())

    @patch.dict("os.environ", {"QDRANT_URL": "https://test.qdrant.tech:6333", "QDRANT_API_KEY": "test_key"})
    def test_qdrant_configured_when_vars_present(self):
        """is_qdrant_configured returns True when both URL and API key are set."""
        self.assertTrue(Config.is_qdrant_configured())
        self.assertEqual(Config.get_qdrant_url(), "https://test.qdrant.tech:6333")
        self.assertEqual(Config.get_qdrant_api_key(), "test_key")

    def test_sanitize_error_message_redacts_credentials(self):
        """Sensitive credentials and API keys in exception messages must be redacted."""
        e1 = Exception("Failed connecting with key sk-12345678901234567890123456")
        self.assertNotIn("sk-12345678901234567890123456", sanitize_error_message(e1))
        self.assertIn("[REDACTED_API_KEY]", sanitize_error_message(e1))

        e2 = Exception("Connection error at https://user:secretpassword123@cluster.qdrant.io:6333")
        self.assertNotIn("secretpassword123", sanitize_error_message(e2))
        self.assertIn("[REDACTED]", sanitize_error_message(e2))

    def test_verify_access_key_timing_safe(self):
        """verify_access_key properly authenticates matching key and rejects incorrect keys."""
        self.assertTrue(verify_access_key("admin123", "admin123"))
        self.assertFalse(verify_access_key("wrong_key", "admin123"))
        self.assertFalse(verify_access_key("", "admin123"))


if __name__ == "__main__":
    unittest.main()
