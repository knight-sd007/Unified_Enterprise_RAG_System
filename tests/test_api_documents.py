"""
Unit tests for Document Ingestion API endpoints (/api/v1/documents/ingest).
"""

import io
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.dependencies import create_session_token, set_rag_pipeline
from api.main import app
from rag.loaders import Document
from rag.pipeline import RAGPipeline
from rag.vector_store import InMemoryVectorStore


class TestAPIDocumentIngestion(unittest.TestCase):
    """Test suite for /api/v1/documents/ingest route."""

    def setUp(self):
        self.key_patcher = patch("config.settings.Config.get_app_access_key", return_value="test-app-key-123")
        self.session_key_patcher = patch("config.settings.Config.get_session_signing_key", return_value="test-signing-key-789")
        self.key_patcher.start()
        self.session_key_patcher.start()
        self.client = TestClient(app)
        self.auth_token = create_session_token()
        self.auth_headers = {"Authorization": f"Bearer {self.auth_token}"}
        # Use isolated in-memory RAGPipeline for test suite
        self.mock_store = InMemoryVectorStore()
        self.pipeline = RAGPipeline(vector_store=self.mock_store)
        set_rag_pipeline(self.pipeline)

    def tearDown(self):
        set_rag_pipeline(None)
        self.session_key_patcher.stop()
        self.key_patcher.stop()

    def test_unauthenticated_ingestion_rejected(self):
        """Unauthenticated request to /api/v1/documents/ingest is rejected with 401."""
        file_content = b"Enterprise security policies and compliance rules."
        response = self.client.post(
            "/api/v1/documents/ingest",
            files={"files": ("policy.txt", io.BytesIO(file_content), "text/plain")},
        )
        self.assertEqual(response.status_code, 401)
        data = response.json()
        self.assertEqual(data["error"]["code"], "UNAUTHORIZED")

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    @patch("providers.gemini_provider.GeminiProvider.embed_documents")
    def test_successful_txt_document_ingestion(self, mock_embed, _mock_conf):
        """Authenticated user can successfully ingest a valid TXT document."""
        mock_embed.return_value = [[0.1] * 768]
        file_content = b"Enterprise standard operating procedures for data governance."

        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "gemini", "chunk_size": "500", "chunk_overlap": "50"},
            files={"files": ("sop.txt", io.BytesIO(file_content), "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_count"], 1)
        self.assertEqual(data["chunk_count"], 1)
        self.assertEqual(data["provider_id"], "gemini")
        self.assertEqual(data["vector_dimension"], 768)
        self.assertIn("sop.txt", data["filenames"])

    @patch("providers.openai_provider.OpenAIProvider.is_configured", return_value=True)
    @patch("providers.openai_provider.OpenAIProvider.embed_documents")
    def test_multiple_documents_ingestion(self, mock_embed, _mock_conf):
        """Multiple documents uploaded in a single request are chunked and indexed."""
        mock_embed.return_value = [[0.05] * 1536, [0.06] * 1536]
        file1 = ("doc1.txt", io.BytesIO(b"Document one content."), "text/plain")
        file2 = ("doc2.txt", io.BytesIO(b"Document two content."), "text/plain")

        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "openai"},
            files=[("files", file1), ("files", file2)],
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["document_count"], 2)
        self.assertEqual(data["chunk_count"], 2)
        self.assertEqual(data["provider_id"], "openai")
        self.assertIn("doc1.txt", data["filenames"])
        self.assertIn("doc2.txt", data["filenames"])

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    def test_unsupported_file_type_rejected(self, _mock_conf):
        """Unsupported file extension (e.g. .py, .exe, .csv) is rejected with 400."""
        file_payload = ("script.py", io.BytesIO(b"print('malicious')"), "text/x-python")

        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "gemini"},
            files={"files": file_payload},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "UNSUPPORTED_FILE_TYPE")
        self.assertIn("Allowed formats", data["error"]["message"])

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    def test_empty_file_rejected(self, _mock_conf):
        """Empty file upload is rejected with 400."""
        empty_file = ("empty.txt", io.BytesIO(b""), "text/plain")

        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "gemini"},
            files={"files": empty_file},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "EMPTY_FILE")

    def test_invalid_provider_rejected(self):
        """Specifying an unknown provider ID is rejected with 400."""
        file_content = b"Some valid document text."
        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "invalid_provider_xyz"},
            files={"files": ("valid.txt", io.BytesIO(file_content), "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "INVALID_PROVIDER")

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=False)
    def test_unconfigured_provider_rejected(self, _mock_conf):
        """Attempting ingestion with an unconfigured provider is rejected with 400."""
        file_content = b"Some valid document text."
        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "gemini"},
            files={"files": ("valid.txt", io.BytesIO(file_content), "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["error"]["code"], "PROVIDER_NOT_CONFIGURED")

    @patch("providers.gemini_provider.GeminiProvider.is_configured", return_value=True)
    @patch("providers.gemini_provider.GeminiProvider.embed_documents")
    def test_filename_path_traversal_sanitized(self, mock_embed, _mock_conf):
        """Filenames with directory traversal patterns are sanitized via os.path.basename."""
        mock_embed.return_value = [[0.1] * 768]
        file_content = b"Some valid document text."

        response = self.client.post(
            "/api/v1/documents/ingest",
            headers=self.auth_headers,
            data={"provider_id": "gemini"},
            files={"files": ("../../etc/passwd.txt", io.BytesIO(file_content), "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("passwd.txt", data["filenames"])
        self.assertNotIn("..", data["filenames"][0])


if __name__ == "__main__":
    unittest.main()
