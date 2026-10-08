"""
Unit tests for OpenAPI Schema, Swagger UI, and ReDoc Documentation.
Verifies authenticated access for users and admins, rejection of unauthenticated requests,
and authorization boundaries for privileged endpoints.
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.dependencies import create_session_token
from api.main import app, _generate_custom_openapi


class TestAPIDocs(unittest.TestCase):
    """Test suite verifying accessibility, route registration, and security schemes for OpenAPI/Swagger UI."""

    def setUp(self):
        self.signing_key_patcher = patch(
            "config.settings.Config.get_session_signing_key",
            return_value="test-session-signing-key-for-docs-789",
        )
        self.admin_key_patcher = patch(
            "config.settings.Config.get_admin_access_key",
            return_value="test-admin-key-for-docs-999",
        )
        self.signing_key_patcher.start()
        self.admin_key_patcher.start()

        self.client = TestClient(app)

        # Normal Google user session
        self.user_token = create_session_token(
            user_id="google_123456789012345678901",
            role="user",
            auth_type="google",
            name="Ada Lovelace",
            email="ada@example.com",
        )
        self.user_cookie_header = {"Cookie": f"p06_session={self.user_token}"}
        self.user_bearer_header = {"Authorization": f"Bearer {self.user_token}"}

        # Admin Google user session
        self.admin_token = create_session_token(
            user_id="google_999999999999999999999",
            role="admin",
            auth_type="google",
            name="Admin Grace",
            email="admin@example.com",
        )
        self.admin_cookie_header = {"Cookie": f"p06_session={self.admin_token}"}

        # Admin break-glass key session
        self.admin_key_token = create_session_token(
            user_id="admin",
            role="admin",
            auth_type="admin_key",
            name="admin",
        )
        self.admin_key_cookie_header = {"Cookie": f"p06_session={self.admin_key_token}"}

    def tearDown(self):
        self.admin_key_patcher.stop()
        self.signing_key_patcher.stop()

    # -------------------------------------------------------------------------
    # Unauthenticated Access Tests (Must be rejected with 401)
    # -------------------------------------------------------------------------

    def test_unauthenticated_swagger_ui_rejected(self):
        """Unauthenticated GET /docs is rejected with 401 UNAUTHORIZED."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_openapi_json_rejected(self):
        """Unauthenticated GET /openapi.json is rejected with 401 UNAUTHORIZED."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_redoc_rejected(self):
        """Unauthenticated GET /redoc is rejected with 401 UNAUTHORIZED."""
        response = self.client.get("/redoc")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_versioned_docs_alias_rejected(self):
        """Unauthenticated GET /api/v1/docs is rejected with 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/docs")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_versioned_openapi_alias_rejected(self):
        """Unauthenticated GET /api/v1/openapi.json is rejected with 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/openapi.json")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_unauthenticated_versioned_redoc_alias_rejected(self):
        """Unauthenticated GET /api/v1/redoc is rejected with 401 UNAUTHORIZED."""
        response = self.client.get("/api/v1/redoc")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    # -------------------------------------------------------------------------
    # Authenticated Normal Google User Tests (Must succeed with 200)
    # -------------------------------------------------------------------------

    def test_user_authenticated_swagger_ui_accessible(self):
        """Authenticated normal Google user GET /docs returns 200 HTML Swagger UI with credentials."""
        response = self.client.get("/docs", headers=self.user_cookie_header)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("swagger-ui", response.text.lower())
        self.assertIn('"withcredentials": true', response.text.lower())

    def test_user_authenticated_openapi_json_accessible(self):
        """Authenticated normal Google user GET /openapi.json returns 200 JSON OpenAPI schema."""
        response = self.client.get("/openapi.json", headers=self.user_cookie_header)
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        schema = response.json()
        self.assertIn("openapi", schema)
        self.assertEqual(schema["info"]["title"], "Unified Enterprise RAG System API")

    def test_user_authenticated_redoc_accessible(self):
        """Authenticated normal Google user GET /redoc returns 200 HTML ReDoc."""
        response = self.client.get("/redoc", headers=self.user_cookie_header)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("redoc", response.text.lower())

    def test_user_authenticated_versioned_aliases_accessible(self):
        """Authenticated normal Google user GET /api/v1/* docs aliases succeed with 200."""
        res_docs = self.client.get("/api/v1/docs", headers=self.user_cookie_header)
        self.assertEqual(res_docs.status_code, 200)
        self.assertIn("swagger-ui", res_docs.text.lower())

        res_openapi = self.client.get("/api/v1/openapi.json", headers=self.user_cookie_header)
        self.assertEqual(res_openapi.status_code, 200)
        self.assertIn("openapi", res_openapi.json())

        res_redoc = self.client.get("/api/v1/redoc", headers=self.user_cookie_header)
        self.assertEqual(res_redoc.status_code, 200)
        self.assertIn("redoc", res_redoc.text.lower())

    def test_user_authenticated_via_bearer_header(self):
        """Authenticated normal Google user via Bearer header can access /docs and /openapi.json."""
        res_docs = self.client.get("/docs", headers=self.user_bearer_header)
        self.assertEqual(res_docs.status_code, 200)

        res_openapi = self.client.get("/openapi.json", headers=self.user_bearer_header)
        self.assertEqual(res_openapi.status_code, 200)
        self.assertIn("openapi", res_openapi.json())

    # -------------------------------------------------------------------------
    # Authenticated Admin Tests (Must succeed with 200)
    # -------------------------------------------------------------------------

    def test_admin_authenticated_swagger_ui_accessible(self):
        """Authenticated Google admin GET /docs returns 200 HTML Swagger UI."""
        response = self.client.get("/docs", headers=self.admin_cookie_header)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("swagger-ui", response.text.lower())

    def test_admin_authenticated_openapi_json_accessible(self):
        """Authenticated Google admin GET /openapi.json returns 200 JSON OpenAPI schema."""
        response = self.client.get("/openapi.json", headers=self.admin_cookie_header)
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        self.assertIn("openapi", response.json())

    def test_admin_authenticated_redoc_accessible(self):
        """Authenticated Google admin GET /redoc returns 200 HTML ReDoc."""
        response = self.client.get("/redoc", headers=self.admin_cookie_header)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers.get("content-type", ""))
        self.assertIn("redoc", response.text.lower())

    def test_admin_key_authenticated_docs_accessible(self):
        """Break-glass admin-key session can access documentation routes."""
        res_docs = self.client.get("/docs", headers=self.admin_key_cookie_header)
        self.assertEqual(res_docs.status_code, 200)
        res_openapi = self.client.get("/openapi.json", headers=self.admin_key_cookie_header)
        self.assertEqual(res_openapi.status_code, 200)

    # -------------------------------------------------------------------------
    # Authorization Boundary Verification
    # -------------------------------------------------------------------------

    def test_docs_access_does_not_grant_admin_privileges(self):
        """
        Exposing Swagger to normal users does NOT grant access to admin-only endpoints.
        Normal user session can access /docs and /openapi.json, but cannot access admin endpoints.
        """
        # User can access docs
        docs_res = self.client.get("/docs", headers=self.user_cookie_header)
        self.assertEqual(docs_res.status_code, 200)

        # Privileged admin endpoints are rejected with 403 FORBIDDEN
        overview_res = self.client.get("/api/v1/admin/overview", headers=self.user_cookie_header)
        self.assertEqual(overview_res.status_code, 403)
        self.assertEqual(overview_res.json()["error"]["code"], "FORBIDDEN")

        clear_res = self.client.post(
            "/api/v1/admin/vectors/clear",
            headers=self.user_cookie_header,
            json={"confirmation": "CONFIRM_ADMIN_GLOBAL_PURGE"},
        )
        self.assertEqual(clear_res.status_code, 403)
        self.assertEqual(clear_res.json()["error"]["code"], "FORBIDDEN")

    # -------------------------------------------------------------------------
    # Schema Content & Security Specification Tests
    # -------------------------------------------------------------------------

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
