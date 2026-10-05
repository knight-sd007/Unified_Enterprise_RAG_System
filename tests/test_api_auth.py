"""
Unit tests for API Authentication & Session Management.
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.main import app
from api.dependencies import SESSION_COOKIE_NAME, create_session_token


class TestAPIAuth(unittest.TestCase):
    """Test suite for /api/v1/auth routes and session lifecycle."""

    def setUp(self):
        self.app_key_patcher = patch("config.settings.Config.get_app_access_key", return_value="test-access-key-123")
        self.admin_key_patcher = patch("config.settings.Config.get_admin_access_key", return_value="test-admin-key-456")
        self.signing_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-signing-key-789")
        self.app_key_patcher.start()
        self.admin_key_patcher.start()
        self.signing_key_patcher.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.signing_key_patcher.stop()
        self.admin_key_patcher.stop()
        self.app_key_patcher.stop()

    def test_login_success_with_valid_key(self):
        """Valid access key succeeds and sets HttpOnly session cookie."""
        response = self.client.post("/api/v1/auth/login", json={"access_key": "test-access-key-123"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("authenticated"))
        self.assertEqual(data.get("role"), "user")
        self.assertEqual(data.get("message"), "Authentication successful.")

        # Check session cookie
        self.assertIn(SESSION_COOKIE_NAME, response.cookies)
        cookie_val = response.cookies[SESSION_COOKIE_NAME]
        self.assertTrue(len(cookie_val) > 20)

        # Confirm access key itself is NOT in response or cookie value
        self.assertNotIn("test-access-key-123", response.text)
        self.assertNotIn("test-access-key-123", cookie_val)

    def test_login_success_with_admin_key(self):
        """Valid admin access key succeeds and sets role='admin'."""
        response = self.client.post("/api/v1/auth/login", json={"access_key": "test-admin-key-456"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("authenticated"))
        self.assertEqual(data.get("role"), "admin")

    def test_login_failure_with_invalid_key(self):
        """Invalid access key returns 401 with structured AUTHENTICATION_FAILED error."""
        response = self.client.post("/api/v1/auth/login", json={"access_key": "wrong-key"})
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "AUTHENTICATION_FAILED")
        self.assertEqual(data["error"]["message"], "Invalid access key.")
        self.assertNotIn(SESSION_COOKIE_NAME, response.cookies)

    def test_login_failure_with_empty_key(self):
        """Empty access key string is rejected."""
        response = self.client.post("/api/v1/auth/login", json={"access_key": ""})
        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["code"], "INVALID_REQUEST")

    def test_auth_status_unauthenticated(self):
        """GET /api/v1/auth/status returns authenticated: False when no cookie is sent."""
        response = self.client.get("/api/v1/auth/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data.get("authenticated"))

    def test_auth_status_authenticated_via_cookie(self):
        """GET /api/v1/auth/status returns authenticated: True when valid session cookie is provided."""
        # 1. Login
        login_res = self.client.post("/api/v1/auth/login", json={"access_key": "test-access-key-123"})
        self.assertEqual(login_res.status_code, 200)

        # 2. Check status (TestClient retains cookies automatically)
        status_res = self.client.get("/api/v1/auth/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertTrue(status_res.json().get("authenticated"))

    def test_auth_status_authenticated_via_bearer_header(self):
        """Bearer token in Authorization header also authenticates session."""
        token = create_session_token()
        res = self.client.get("/api/v1/auth/status", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json().get("authenticated"))

    def test_auth_status_tampered_cookie_rejected(self):
        """Forged or malformed session cookie is rejected."""
        self.client.cookies.set(SESSION_COOKIE_NAME, "forged.tampered.token.123")
        res = self.client.get("/api/v1/auth/status")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json().get("authenticated"))

    def test_logout_clears_session(self):
        """POST /api/v1/auth/logout invalidates session and clears cookie."""
        # 1. Login
        self.client.post("/api/v1/auth/login", json={"access_key": "test-access-key-123"})
        self.assertTrue(self.client.get("/api/v1/auth/status").json().get("authenticated"))

        # 2. Logout
        logout_res = self.client.post("/api/v1/auth/logout")
        self.assertEqual(logout_res.status_code, 200)
        self.assertFalse(logout_res.json().get("authenticated"))

        # 3. Status is now false
        status_res = self.client.get("/api/v1/auth/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertFalse(status_res.json().get("authenticated"))

    def test_login_fails_when_access_key_not_configured(self):
        """When server has no configured APP_ACCESS_KEY and no ADMIN_ACCESS_KEY, login returns 500."""
        with patch("config.settings.Config.get_app_access_key", return_value=""), \
             patch("config.settings.Config.get_admin_access_key", return_value=""):
            response = self.client.post("/api/v1/auth/login", json={"access_key": "any-key"})
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "AUTH_CONFIGURATION_ERROR")

    def test_login_fails_when_session_signing_key_not_configured(self):
        """When SESSION_SIGNING_KEY is missing, login returns 500 AUTH_CONFIGURATION_ERROR."""
        with patch("config.settings.Config.get_session_signing_key", return_value=""):
            response = self.client.post("/api/v1/auth/login", json={"access_key": "test-access-key-123"})
            self.assertEqual(response.status_code, 500)
            data = response.json()
            self.assertIn("error", data)
            self.assertEqual(data["error"]["code"], "AUTH_CONFIGURATION_ERROR")


if __name__ == "__main__":
    unittest.main()
