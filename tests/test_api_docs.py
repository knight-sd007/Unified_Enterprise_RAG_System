"""
Unit tests for Protected OpenAPI Schema & Swagger Documentation.
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import create_session_token


class TestAPIDocs(unittest.TestCase):
    """Test suite verifying authentication enforcement on Swagger UI and OpenAPI schema."""

    def setUp(self):
        self.client = TestClient(app)

    def test_unauthenticated_docs_rejected(self):
        """GET /docs without authentication returns 401 UNAUTHORIZED."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "UNAUTHORIZED")
        self.assertEqual(data["error"]["message"], "Authentication required.")

    def test_unauthenticated_openapi_json_rejected(self):
        """GET /openapi.json without authentication returns 401 UNAUTHORIZED."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_versioned_docs_rejected(self):
        """GET /api/v1/docs without authentication returns 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/docs")
        self.assertEqual(response.status_code, 401)

    def test_unauthenticated_versioned_openapi_rejected(self):
        """GET /api/v1/openapi.json without authentication returns 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 401)

    @patch("config.settings.Config.get_app_access_key", return_value="test-key")
    def test_authenticated_docs_accessible(self, _mock_key):
        """GET /docs with authenticated session returns 200 HTML Swagger UI."""
        token = create_session_token()
        response = self.client.get("/docs", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("swagger-ui", response.text.lower())

    @patch("config.settings.Config.get_app_access_key", return_value="test-key")
    def test_authenticated_openapi_json_accessible(self, _mock_key):
        """GET /openapi.json with authenticated session returns 200 OpenAPI schema."""
        token = create_session_token()
        response = self.client.get("/openapi.json", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(response.status_code, 200)
        schema = response.json()

        self.assertIn("openapi", schema)
        self.assertIn("info", schema)
        self.assertEqual(schema["info"]["title"], "Unified Enterprise RAG System API")
        self.assertEqual(schema["info"]["version"], "1.0.0")

        # Paths must only contain implemented endpoints
        paths = schema.get("paths", {})
        self.assertIn("/api/v1/health", paths)
        self.assertIn("/api/v1/auth/login", paths)
        self.assertIn("/api/v1/auth/status", paths)
        self.assertIn("/api/v1/auth/logout", paths)
        self.assertIn("/api/v1/providers", paths)

        # Confirm non-implemented endpoints are NOT in schema
        self.assertNotIn("/api/v1/query", paths)
        self.assertNotIn("/api/v1/documents", paths)
        self.assertNotIn("/api/v1/ingest", paths)


if __name__ == "__main__":
    unittest.main()
