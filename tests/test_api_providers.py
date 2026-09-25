"""
Unit tests for AI Providers API Endpoint.
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import create_session_token


class TestAPIProviders(unittest.TestCase):
    """Test suite for /api/v1/providers endpoint."""

    def setUp(self):
        self.client = TestClient(app)

    def test_providers_unauthenticated_rejected(self):
        """GET /api/v1/providers without credentials must return 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/providers")
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "UNAUTHORIZED")
        self.assertEqual(data["error"]["message"], "Authentication required.")

    @patch("config.settings.Config.get_app_access_key", return_value="test-key")
    def test_providers_authenticated_returns_metadata(self, _mock_key):
        """Authenticated request returns safe provider list."""
        token = create_session_token()
        response = self.client.get("/api/v1/providers", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("providers", data)
        providers_list = data["providers"]
        self.assertEqual(len(providers_list), 3)

        provider_ids = [p["provider_id"] for p in providers_list]
        self.assertIn("openai", provider_ids)
        self.assertIn("gemini", provider_ids)
        self.assertIn("nvidia_nim", provider_ids)

        # Check Gemini metadata
        gemini_data = next(p for p in providers_list if p["provider_id"] == "gemini")
        self.assertEqual(gemini_data["name"], "Google Gemini")
        self.assertEqual(gemini_data["dimension"], 768)
        self.assertEqual(gemini_data["embedding_model"], "gemini-embedding-2")
        self.assertEqual(gemini_data["chat_model"], "gemini-2.5-flash")
        self.assertIsInstance(gemini_data["configured"], bool)

        # Check NVIDIA NIM metadata
        nvidia_data = next(p for p in providers_list if p["provider_id"] == "nvidia_nim")
        self.assertEqual(nvidia_data["name"], "NVIDIA NIM")
        self.assertEqual(nvidia_data["dimension"], 2048)
        self.assertEqual(nvidia_data["embedding_model"], "nvidia/llama-nemotron-embed-1b-v2")
        self.assertEqual(nvidia_data["chat_model"], "nvidia/nemotron-3-super-120b-a12b")

        # Check OpenAI metadata
        openai_data = next(p for p in providers_list if p["provider_id"] == "openai")
        self.assertEqual(openai_data["name"], "OpenAI")
        self.assertEqual(openai_data["dimension"], 1536)
        self.assertEqual(openai_data["embedding_model"], "text-embedding-3-small")
        self.assertEqual(openai_data["chat_model"], "gpt-4o-mini")

    @patch("config.settings.Config.get_app_access_key", return_value="test-key")
    def test_providers_no_credentials_exposed(self, _mock_key):
        """Response must strictly not expose API keys, secrets, or internal collection names."""
        token = create_session_token()
        response = self.client.get("/api/v1/providers", headers={"Authorization": f"Bearer {token}"})
        raw_text = response.text.lower()
        self.assertNotIn("api_key", raw_text)
        self.assertNotIn("qdrant_url", raw_text)
        self.assertNotIn("collection_name", raw_text)
        self.assertNotIn("p06_gemini_embedding_2_768", raw_text)


if __name__ == "__main__":
    unittest.main()
