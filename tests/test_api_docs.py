"""
Unit tests for OpenAPI Schema, Swagger UI, and ReDoc Documentation.
"""

import unittest
from fastapi.testclient import TestClient
from api.main import app, _generate_custom_openapi


class TestAPIDocs(unittest.TestCase):
    """Test suite verifying accessibility, route registration, and security schemes for OpenAPI/Swagger UI."""

    def setUp(self):
        self.client = TestClient(app)

    def test_swagger_ui_accessible(self):
        """GET /docs returns 200 HTML Swagger UI."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("swagger-ui", response.text.lower())

    def test_openapi_json_accessible(self):
        """GET /openapi.json returns 200 JSON with OpenAPI 3.x schema."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        schema = response.json()
        self.assertIn("openapi", schema)
        self.assertEqual(schema["info"]["title"], "Unified Enterprise RAG System API")
        self.assertEqual(schema["info"]["version"], "1.0.0")

    def test_redoc_accessible(self):
        """GET /redoc returns 200 HTML ReDoc documentation."""
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("redoc", response.text.lower())

    def test_versioned_docs_alias_accessible(self):
        """GET /api/v1/docs returns 200 HTML Swagger UI."""
        response = self.client.get("/api/v1/docs")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("swagger-ui", response.text.lower())

    def test_versioned_openapi_alias_accessible(self):
        """GET /api/v1/openapi.json returns 200 JSON OpenAPI schema."""
        response = self.client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        schema = response.json()
        self.assertIn("openapi", schema)

    def test_versioned_redoc_alias_accessible(self):
        """GET /api/v1/redoc returns 200 HTML ReDoc documentation."""
        response = self.client.get("/api/v1/redoc")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("redoc", response.text.lower())

    def test_openapi_schema_contains_registered_routes(self):
        """OpenAPI schema contains all verified API endpoints."""
        schema = _generate_custom_openapi()
        paths = schema.get("paths", {})

        # Verified endpoints
        expected_endpoints = [
            "/api/v1/health",
            "/api/v1/auth/login",
            "/api/v1/auth/status",
            "/api/v1/auth/logout",
            "/api/v1/providers",
            "/api/v1/documents",
            "/api/v1/documents/{doc_id}",
            "/api/v1/documents/ingest",
            "/api/v1/rag/query",
            "/api/v1/rag/stats",
            "/api/v1/rag/index",
        ]
        for ep in expected_endpoints:
            self.assertIn(ep, paths, f"Expected endpoint {ep} missing from OpenAPI schema")

        # Confirm un-implemented endpoints are NOT in schema
        self.assertNotIn("/api/v1/query", paths)
        self.assertNotIn("/api/v1/search", paths)
        self.assertNotIn("/api/v1/ingest", paths)

    def test_openapi_schema_security_schemes(self):
        """OpenAPI schema contains CookieAuth and BearerAuth security definitions."""
        schema = _generate_custom_openapi()
        components = schema.get("components", {})
        self.assertIn("securitySchemes", components)

        security_schemes = components["securitySchemes"]
        self.assertIn("CookieAuth", security_schemes)
        self.assertEqual(security_schemes["CookieAuth"]["type"], "apiKey")
        self.assertEqual(security_schemes["CookieAuth"]["in"], "cookie")
        self.assertEqual(security_schemes["CookieAuth"]["name"], "p06_session")

        self.assertIn("BearerAuth", security_schemes)
        self.assertEqual(security_schemes["BearerAuth"]["type"], "http")
        self.assertEqual(security_schemes["BearerAuth"]["scheme"], "bearer")

    def test_protected_routes_have_security_and_401_responses(self):
        """Protected routes declare security requirements and 401 response in schema."""
        schema = _generate_custom_openapi()
        paths = schema.get("paths", {})

        protected_paths = [
            "/api/v1/providers",
            "/api/v1/documents",
            "/api/v1/documents/{doc_id}",
            "/api/v1/documents/ingest",
            "/api/v1/rag/query",
            "/api/v1/rag/stats",
            "/api/v1/rag/index",
        ]

        for path_name in protected_paths:
            self.assertIn(path_name, paths)
            path_item = paths[path_name]
            for method, op in path_item.items():
                if method.lower() in ("get", "post", "put", "delete"):
                    self.assertIn("security", op, f"{method.upper()} {path_name} missing security declaration")
                    self.assertIn("401", op.get("responses", {}), f"{method.upper()} {path_name} missing 401 response")


if __name__ == "__main__":
    unittest.main()
