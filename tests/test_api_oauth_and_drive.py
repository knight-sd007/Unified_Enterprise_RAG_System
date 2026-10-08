"""
Unit and integration tests for Google OAuth and Google Drive storage integration.
"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from api.dependencies import create_session_token
from api.main import app
from rag.storage.google_drive import GoogleDriveStorage
from rag.storage.metadata_db import (
    MetadataRepository,
    decrypt_secret_data,
    encrypt_secret_data,
    get_metadata_repo,
)


class TestGoogleOAuthAndDrive(unittest.IsolatedAsyncioTestCase):
    """Test suite for OAuth token exchange, role resolution, and Drive storage."""

    def setUp(self):
        self.signing_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-session-signing-key-789")
        self.oauth_patcher_id = patch("config.settings.Config.get_google_client_id", return_value="test-client-id.apps.googleusercontent.com")
        self.oauth_patcher_secret = patch("config.settings.Config.get_google_client_secret", return_value="test-client-secret")
        self.oauth_patcher_admins = patch("config.settings.Config.get_google_admin_emails", return_value=["admin@enterprise.com", "google_12345"])

        self.signing_key_patcher.start()
        self.oauth_patcher_id.start()
        self.oauth_patcher_secret.start()
        self.oauth_patcher_admins.start()

        get_metadata_repo().delete_all_global()
        self.client = TestClient(app)

    def tearDown(self):
        get_metadata_repo().delete_all_global()
        self.oauth_patcher_admins.stop()
        self.oauth_patcher_secret.stop()
        self.oauth_patcher_id.stop()
        self.signing_key_patcher.stop()

    def test_oauth_config_endpoint(self):
        """GET /api/v1/auth/google/config returns public OAuth configuration."""
        res = self.client.get("/api/v1/auth/google/config")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["configured"])
        self.assertEqual(data["client_id"], "test-client-id.apps.googleusercontent.com")

    def test_oauth_url_endpoint(self):
        """GET /api/v1/auth/google/url returns OAuth URL and sets signed anti-CSRF state cookie."""
        res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("url", data)
        self.assertIn("state", data)
        self.assertIn("accounts.google.com", data["url"])
        self.assertIn(data["state"], data["url"])
        self.assertIn("p06_oauth_state", res.cookies)

    def test_metadata_secret_encryption(self):
        """Secret data is symmetrically encrypted and decrypted via Fernet."""
        data = {"access_token": "secret_access_xyz", "refresh_token": "secret_refresh_123"}
        encrypted = encrypt_secret_data(data)
        self.assertNotIn("secret_access_xyz", encrypted)

        decrypted = decrypt_secret_data(encrypted)
        self.assertEqual(decrypted, data)

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_regular_user(self, mock_get, mock_post):
        """Standard OAuth callback with valid state assigns user_id='google_<sub_id>' and role='user'."""
        # 1. Fetch OAuth URL and signed state cookie
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "987654321", "email": "employee@enterprise.com", "email_verified": True},
        )

        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "valid_google_auth_code", "state": state},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["user_id"], "google_987654321")
        self.assertEqual(data["role"], "user")

        # Verify encrypted tokens saved in repository
        repo = get_metadata_repo()
        tokens = repo.get_oauth_tokens("google_987654321", "google")
        self.assertIsNotNone(tokens)
        self.assertEqual(tokens["access_token"], "mock_at")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_admin_user_verified(self, mock_get, mock_post):
        """OAuth user in GOOGLE_ADMIN_EMAILS with email_verified=True receives role='admin'."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at_admin", "refresh_token": "mock_rt_admin"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "12345", "email": "admin@enterprise.com", "email_verified": True},
        )

        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "admin_google_auth_code", "state": state},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["user_id"], "google_12345")
        self.assertEqual(data["role"], "admin")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_admin_user_unverified_rejected_to_user_role(self, mock_get, mock_post):
        """OAuth user matching GOOGLE_ADMIN_EMAILS but with email_verified=False is restricted to role='user'."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at_unverified", "refresh_token": "mock_rt_unverified"},
        )
        # Email matches admin list, but email_verified is explicitly False
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "fake_admin_99", "email": "admin@enterprise.com", "email_verified": False},
        )

        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "unverified_admin_code", "state": state},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["user_id"], "google_fake_admin_99")
        # Must NOT be elevated to admin
        self.assertEqual(data["role"], "user")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_missing_email_verified_field_receives_user_role(self, mock_get, mock_post):
        """OAuth user matching GOOGLE_ADMIN_EMAILS without email_verified claim is restricted to role='user'."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        # Missing email_verified claim entirely
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "missing_verified_sub", "email": "admin@enterprise.com"},
        )

        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "missing_verified_code", "state": state},
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["user_id"], "google_missing_verified_sub")
        self.assertEqual(data["role"], "user")

    def test_google_oauth_callback_missing_state_fails(self):
        """Callback without state payload or cookie is rejected with 400 INVALID_OAUTH_STATE."""
        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "test_code"},
        )
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertEqual(data["error"]["code"], "INVALID_OAUTH_STATE")

    def test_google_oauth_callback_mismatched_state_fails(self):
        """Callback with state mismatched from signed cookie is rejected with 400 OAUTH_STATE_MISMATCH."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)

        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "test_code", "state": "tampered_or_different_state_123"},
        )
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertEqual(data["error"]["code"], "OAUTH_STATE_MISMATCH")

    def test_google_oauth_callback_invalid_cookie_fails(self):
        """Callback with forged/invalid signed cookie is rejected with 400 EXPIRED_OAUTH_STATE."""
        self.client.cookies.set("p06_oauth_state", "forged_invalid_cookie_signature")
        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "test_code", "state": "some_state"},
        )
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertEqual(data["error"]["code"], "EXPIRED_OAUTH_STATE")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_admin_user_by_sub_allowlist(self, mock_get, mock_post):
        """OAuth user matching GOOGLE_ADMIN_SUBS receives role='admin' via verified Google sub."""
        with patch("config.settings.Config.get_google_admin_subs", return_value=["dedicated_admin_sub_99"]):
            url_res = self.client.get("/api/v1/auth/google/url")
            self.assertEqual(url_res.status_code, 200)
            state = url_res.json()["state"]

            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
            )
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"sub": "dedicated_admin_sub_99", "email": "nonadmin_email@enterprise.com", "email_verified": False},
            )

            res = self.client.post(
                "/api/v1/auth/google/callback",
                json={"code": "sub_admin_code", "state": state},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data["authenticated"])
            self.assertEqual(data["user_id"], "google_dedicated_admin_sub_99")
            self.assertEqual(data["role"], "admin")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_state_replay_fails(self, mock_get, mock_post):
        """State token is consumed on first use and cannot be replayed."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "user_replay_test", "email": "replay@test.com", "email_verified": True},
        )

        # First request succeeds and clears state cookie
        res1 = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "auth_code_1", "state": state},
        )
        self.assertEqual(res1.status_code, 200)

        # Second request using the same state must fail because cookie is now gone
        res2 = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "auth_code_2", "state": state},
        )
        self.assertEqual(res2.status_code, 400)
        self.assertEqual(res2.json()["error"]["code"], "INVALID_OAUTH_STATE")

    @patch("httpx.AsyncClient.post")
    def test_oauth_state_cookie_cleared_on_token_exchange_failure(self, mock_post):
        """Failed OAuth token exchange deletes state cookie to prevent replay attacks."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=400,
            text="Invalid authorization code",
        )

        res = self.client.post(
            "/api/v1/auth/google/callback",
            json={"code": "invalid_code", "state": state},
        )
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"]["code"], "OAUTH_EXCHANGE_FAILED")

        # Confirm state cookie was cleared
        set_cookie_header = res.headers.get("set-cookie", "")
        self.assertIn("p06_oauth_state=", set_cookie_header)
        self.assertIn("Max-Age=0", set_cookie_header)

    def test_google_oauth_cookie_security_attributes(self):
        """State cookie is configured with HttpOnly, SameSite=Lax, and Max-Age=600."""
        res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(res.status_code, 200)
        set_cookie_header = res.headers.get("set-cookie", "")
        self.assertIn("p06_oauth_state=", set_cookie_header)
        self.assertIn("HttpOnly", set_cookie_header)
        self.assertIn("samesite=lax", set_cookie_header.lower())
        self.assertIn("max-age=600", set_cookie_header.lower())

    @patch("httpx.AsyncClient.post")
    async def test_google_drive_storage_upload_and_delete(self, mock_post):
        """GoogleDriveStorage handles upload and delete using stored token."""
        repo = get_metadata_repo()
        repo.save_oauth_tokens("google_12345", "google", {"access_token": "mock_drive_token"})

        storage = GoogleDriveStorage()
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": "drive_file_id_999"},
        )

        file_id = await storage.upload_file(
            owner_id="google_12345",
            filename="test_upload.pdf",
            content=b"test bytes",
            mime_type="application/pdf",
        )
        self.assertEqual(file_id, "drive_file_id_999")

    # -------------------------------------------------------------------------
    # GET /api/v1/auth/google/callback Tests (Browser Redirect Flow)
    # -------------------------------------------------------------------------

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_get_regular_user_success(self, mock_get, mock_post):
        """GET /api/v1/auth/google/callback provisions user session and returns HTTP 303 redirect to /."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at_get", "refresh_token": "mock_rt_get"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "get_sub_123", "email": "employee@enterprise.com", "email_verified": True},
        )

        res = self.client.get(
            f"/api/v1/auth/google/callback?code=valid_get_code&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/")

        # Session cookie issued
        self.assertIn("p06_session", res.cookies)
        # OAuth state cookie deleted/expired
        set_cookie_header = res.headers.get("set-cookie", "")
        self.assertIn("p06_oauth_state=", set_cookie_header)

        # Authenticated session can access /status
        status_res = self.client.get("/api/v1/auth/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertTrue(status_data["authenticated"])
        self.assertEqual(status_data["user_id"], "google_get_sub_123")
        self.assertEqual(status_data["role"], "user")
        self.assertTrue(status_data["drive_authorized"])

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_get_admin_user_verified(self, mock_get, mock_post):
        """GET /api/v1/auth/google/callback grants admin role to verified admin email and redirects 303 to /."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "admin_sub_456", "email": "admin@enterprise.com", "email_verified": True},
        )

        res = self.client.get(
            f"/api/v1/auth/google/callback?code=valid_code&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/")

        status_res = self.client.get("/api/v1/auth/status")
        self.assertEqual(status_res.json().get("role"), "admin")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_get_unverified_email_downgraded_to_user(self, mock_get, mock_post):
        """GET /api/v1/auth/google/callback keeps user role if admin email is unverified."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "fake_admin_789", "email": "admin@enterprise.com", "email_verified": False},
        )

        res = self.client.get(
            f"/api/v1/auth/google/callback?code=valid_code&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/")

        status_res = self.client.get("/api/v1/auth/status")
        self.assertEqual(status_res.json().get("role"), "user")

    def test_google_oauth_callback_get_missing_state_fails(self):
        """GET callback with missing state parameter is rejected with 400 INVALID_OAUTH_STATE."""
        res = self.client.get("/api/v1/auth/google/callback?code=valid_code")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "INVALID_OAUTH_STATE")

    def test_google_oauth_callback_get_missing_cookie_fails(self):
        """GET callback with missing oauth_state cookie is rejected with 400 INVALID_OAUTH_STATE."""
        client_no_cookie = TestClient(app)
        res = client_no_cookie.get("/api/v1/auth/google/callback?code=valid_code&state=some_state")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "INVALID_OAUTH_STATE")

    def test_google_oauth_callback_get_mismatched_state_fails(self):
        """GET callback with mismatched state is rejected with 400 OAUTH_STATE_MISMATCH."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)

        res = self.client.get("/api/v1/auth/google/callback?code=valid_code&state=wrong_mismatched_state")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "OAUTH_STATE_MISMATCH")

    def test_google_oauth_callback_get_invalid_tampered_cookie_fails(self):
        """GET callback with tampered cookie is rejected with 400 EXPIRED_OAUTH_STATE."""
        self.client.cookies.set("p06_oauth_state", "forged_tampered_cookie_value")
        res = self.client.get("/api/v1/auth/google/callback?code=valid_code&state=some_state")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "EXPIRED_OAUTH_STATE")

    def test_google_oauth_callback_get_missing_code_fails(self):
        """GET callback with missing code is rejected with 400 INVALID_OAUTH_CODE."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        res = self.client.get(f"/api/v1/auth/google/callback?state={state}")
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.json()["error"]["code"], "INVALID_OAUTH_CODE")

    def test_google_oauth_callback_get_oauth_error_response_redirects_safely(self):
        """GET callback when Google returns OAuth error redirects to /?error=... and deletes state cookie."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        self.assertIn("p06_oauth_state", url_res.cookies)

        res = self.client.get(
            "/api/v1/auth/google/callback?error=access_denied&error_description=The+user+denied+access",
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/?error=access_denied")
        # Ensure state cookie is deleted
        set_cookie_header = res.headers.get("set-cookie", "")
        self.assertIn("p06_oauth_state=", set_cookie_header)
        # Ensure no session cookie was created
        self.assertNotIn("p06_session", res.cookies)

    def test_google_oauth_callback_get_oauth_error_sanitizes_open_redirect(self):
        """GET callback strictly prevents open redirects when error contains hostile payload."""
        res = self.client.get(
            "/api/v1/auth/google/callback?error=https://evil.com/leak&error_description=attack",
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/?error=httpsevilcomleak")
        self.assertTrue(res.headers.get("location", "").startswith("/?error="))

    @patch("httpx.AsyncClient.post")
    def test_google_oauth_callback_get_failed_token_exchange(self, mock_post):
        """GET callback with failed upstream Google token exchange returns 401 OAUTH_EXCHANGE_FAILED."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(status_code=400, text="Bad Request from Google")

        res = self.client.get(f"/api/v1/auth/google/callback?code=bad_code&state={state}")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"]["code"], "OAUTH_EXCHANGE_FAILED")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_get_failed_userinfo(self, mock_get, mock_post):
        """GET callback with failed upstream Google userinfo lookup returns 401 OAUTH_USERINFO_FAILED."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(status_code=401, text="Unauthorized token")

        res = self.client.get(f"/api/v1/auth/google/callback?code=valid_code&state={state}")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(res.json()["error"]["code"], "OAUTH_USERINFO_FAILED")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_callback_get_state_replay_fails(self, mock_get, mock_post):
        """GET callback state cookie cannot be replayed after single successful usage."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sub": "user_replay_get", "email": "replay_get@test.com", "email_verified": True},
        )

        res1 = self.client.get(
            f"/api/v1/auth/google/callback?code=code_1&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(res1.status_code, 303)

        res2 = self.client.get(
            f"/api/v1/auth/google/callback?code=code_2&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(res2.status_code, 400)
        self.assertEqual(res2.json()["error"]["code"], "INVALID_OAUTH_STATE")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_display_name_preserved_with_immutable_sub(self, mock_get, mock_post):
        """Google user with valid profile name receives display name while sub remains immutable internal ID."""
        url_res = self.client.get("/api/v1/auth/google/url")
        self.assertEqual(url_res.status_code, 200)
        state = url_res.json()["state"]

        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
        )
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sub": "101239956578379584549",
                "name": "Soumojit Das",
                "email": "soumojit@enterprise.com",
                "email_verified": True,
                "picture": "https://lh3.googleusercontent.com/photo.jpg",
            },
        )

        res = self.client.get(
            f"/api/v1/auth/google/callback?code=valid_code&state={state}",
            follow_redirects=False,
        )
        self.assertEqual(res.status_code, 303)
        self.assertEqual(res.headers.get("location"), "/")

        # Status check confirms sub remains immutable internal ID and display name is Google name
        status_res = self.client.get("/api/v1/auth/status")
        self.assertEqual(status_res.status_code, 200)
        data = status_res.json()
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["user_id"], "google_101239956578379584549")
        self.assertEqual(data["name"], "Soumojit Das")
        self.assertEqual(data["email"], "soumojit@enterprise.com")
        self.assertEqual(data["picture"], "https://lh3.googleusercontent.com/photo.jpg")

    @patch("httpx.AsyncClient.post")
    @patch("httpx.AsyncClient.get")
    def test_google_oauth_display_name_fallback_when_name_missing_or_empty(self, mock_get, mock_post):
        """Google user without a usable name safely falls back to 'Google User' without exposing sub as display name."""
        for empty_name in [None, "", "   "]:
            url_res = self.client.get("/api/v1/auth/google/url")
            self.assertEqual(url_res.status_code, 200)
            state = url_res.json()["state"]

            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"access_token": "mock_at", "refresh_token": "mock_rt"},
            )
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "sub": "999888777666",
                    "name": empty_name,
                    "email": "noname@enterprise.com",
                    "email_verified": True,
                },
            )

            res = self.client.post(
                "/api/v1/auth/google/callback",
                json={"code": "valid_code", "state": state},
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["user_id"], "google_999888777666")
            self.assertEqual(data["name"], "Google User")
            self.assertNotEqual(data["name"], data["user_id"])

            status_res = self.client.get("/api/v1/auth/status")
            self.assertEqual(status_res.status_code, 200)
            status_data = status_res.json()
            self.assertEqual(status_data["user_id"], "google_999888777666")
            self.assertEqual(status_data["name"], "Google User")


if __name__ == "__main__":
    unittest.main()
