"""
Unit tests for API Health Endpoint & Security Middleware.
"""

import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestAPIHealth(unittest.TestCase):
    """Test suite for /api/v1/health endpoint and global middleware."""

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint_unauthenticated_success(self):
        """GET /api/v1/health must succeed without authentication."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "p06-enterprise-rag")
        self.assertEqual(data.get("version"), "1.0.0")

    def test_health_no_credentials_in_payload(self):
        """Health payload must not expose credentials, env variables, or secrets."""
        response = self.client.get("/api/v1/health")
        payload_str = response.text.lower()
        self.assertNotIn("key", payload_str)
        self.assertNotIn("secret", payload_str)
        self.assertNotIn("token", payload_str)
        self.assertNotIn("password", payload_str)
        self.assertNotIn("qdrant", payload_str)

    def test_security_headers_present(self):
        """Response must include standard hardening security headers."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")


if __name__ == "__main__":
    unittest.main()
