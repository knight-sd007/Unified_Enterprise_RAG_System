"""
Durable SQLite metadata repository for documents and encrypted OAuth tokens.
Enforces multi-user isolation, document lifecycle persistence, and audit capabilities.
"""

import base64
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
from config.settings import Config


def _get_fernet_key() -> bytes:
    """Derives a deterministic 32-byte Fernet key from SESSION_SIGNING_KEY."""
    signing_key = Config.get_session_signing_key()
    digest = hashlib.sha256(signing_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def encrypt_secret_data(data: Dict[str, Any]) -> str:
    """Encrypts a dictionary into a fernet token string."""
    fernet = Fernet(_get_fernet_key())
    raw_bytes = json.dumps(data).encode("utf-8")
    return fernet.encrypt(raw_bytes).decode("utf-8")


def decrypt_secret_data(token_str: str) -> Optional[Dict[str, Any]]:
    """Decrypts a fernet token string into a dictionary."""
    try:
        fernet = Fernet(_get_fernet_key())
        raw_bytes = fernet.decrypt(token_str.encode("utf-8"))
        return json.loads(raw_bytes.decode("utf-8"))
    except Exception:
        return None


class MetadataRepository:
    """SQLite-backed metadata repository for documents and OAuth credentials."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Config.get_metadata_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes database schema and indexes."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    provider_id TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    drive_file_id TEXT,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'indexed'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_owner ON documents(owner_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_provider ON documents(provider_id)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    encrypted_tokens TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, provider)
                )
            """)
            conn.commit()

    def save_document(
        self,
        doc_id: str,
        owner_id: str,
        filename: str,
        file_size: int,
        char_count: int,
        provider_id: str,
        embedding_model: str,
        chunk_count: int,
        drive_file_id: Optional[str] = None,
        status: str = "indexed",
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inserts or updates a document metadata record."""
        now = created_at or datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, owner_id, filename, file_size, char_count,
                    provider_id, embedding_model, chunk_count, drive_file_id, created_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    owner_id,
                    filename,
                    file_size,
                    char_count,
                    provider_id,
                    embedding_model,
                    chunk_count,
                    drive_file_id,
                    now,
                    status,
                ),
            )
            conn.commit()
        return self.get_document(doc_id)  # type: ignore

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by doc_id."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_documents(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists documents, optionally scoped by owner_id."""
        with self._get_conn() as conn:
            if owner_id:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE owner_id = ? ORDER BY created_at DESC",
                    (owner_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY created_at DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, doc_id: str, owner_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Deletes a single document. If owner_id is specified, strictly verifies ownership.
        Returns the deleted document metadata if found and deleted, else None.
        """
        doc = self.get_document(doc_id)
        if not doc:
            return None
        if owner_id and doc["owner_id"] != owner_id:
            return None

        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
        return doc

    def delete_all_for_user(self, owner_id: str) -> List[Dict[str, Any]]:
        """Deletes all documents for a specific user and returns their metadata."""
        docs = self.list_documents(owner_id=owner_id)
        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents WHERE owner_id = ?", (owner_id,))
            conn.commit()
        return docs

    def delete_all_global(self) -> List[Dict[str, Any]]:
        """Admin operation: Deletes all document records globally and returns their metadata."""
        docs = self.list_documents(owner_id=None)
        with self._get_conn() as conn:
            conn.execute("DELETE FROM documents")
            conn.commit()
        return docs

    def get_user_stats(self, owner_id: str) -> Dict[str, Any]:
        """Calculates document and chunk statistics for a single user."""
        with self._get_conn() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as document_count,
                    COALESCE(SUM(chunk_count), 0) as chunk_count,
                    COALESCE(SUM(file_size), 0) as total_bytes,
                    COALESCE(SUM(char_count), 0) as total_characters
                FROM documents
                WHERE owner_id = ?
                """,
                (owner_id,),
            ).fetchone()
            return dict(row) if row else {
                "document_count": 0,
                "chunk_count": 0,
                "total_bytes": 0,
                "total_characters": 0,
            }

    def get_global_stats(self) -> Dict[str, Any]:
        """Calculates aggregate statistics across all users."""
        with self._get_conn() as conn:
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) as total_documents,
                    COALESCE(SUM(chunk_count), 0) as total_chunks,
                    COALESCE(SUM(file_size), 0) as total_bytes,
                    COUNT(DISTINCT owner_id) as total_users
                FROM documents
                """
            ).fetchone()

            providers = conn.execute(
                """
                SELECT provider_id, COUNT(*) as doc_count, COALESCE(SUM(chunk_count), 0) as chunk_count
                FROM documents
                GROUP BY provider_id
                """
            ).fetchall()

            return {
                "total_documents": summary["total_documents"] if summary else 0,
                "total_chunks": summary["total_chunks"] if summary else 0,
                "total_bytes": summary["total_bytes"] if summary else 0,
                "total_users": summary["total_users"] if summary else 0,
                "providers": [dict(p) for p in providers],
            }

    # --- OAuth Token Management ---

    def save_oauth_tokens(self, user_id: str, provider: str, token_data: Dict[str, Any]) -> None:
        """Stores encrypted OAuth tokens for a user and provider."""
        encrypted = encrypt_secret_data(token_data)
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO oauth_tokens (user_id, provider, encrypted_tokens, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, provider, encrypted, now),
            )
            conn.commit()

    def get_oauth_tokens(self, user_id: str, provider: str) -> Optional[Dict[str, Any]]:
        """Retrieves and decrypts OAuth tokens for a user and provider."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT encrypted_tokens FROM oauth_tokens WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            ).fetchone()
            if not row:
                return None
            return decrypt_secret_data(row["encrypted_tokens"])

    def delete_oauth_tokens(self, user_id: str, provider: str) -> bool:
        """Deletes stored OAuth tokens for a user and provider."""
        with self._get_conn() as conn:
            res = conn.execute(
                "DELETE FROM oauth_tokens WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            )
            conn.commit()
            return res.rowcount > 0


# Global singleton instance
_metadata_repo: Optional[MetadataRepository] = None


def get_metadata_repo() -> MetadataRepository:
    """Returns the singleton MetadataRepository instance."""
    global _metadata_repo
    if _metadata_repo is None:
        _metadata_repo = MetadataRepository()
    return _metadata_repo
