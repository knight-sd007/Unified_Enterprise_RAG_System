"""
Google Drive storage integration using the drive.file scope.
Allows users who authenticate with Google OAuth to retain their uploaded source files in their own Drive.
"""

import json
import logging
from typing import Any, Dict, Optional

import httpx
from config.settings import Config
from rag.storage.base import BaseDocumentStorage
from rag.storage.metadata_db import get_metadata_repo

logger = logging.getLogger(__name__)


class GoogleDriveStorage(BaseDocumentStorage):
    """Integrates with Google Drive API v3 to store and manage original documents."""

    def __init__(self):
        self.client_id = Config.get_google_client_id()
        self.client_secret = Config.get_google_client_secret()

    async def get_valid_access_token(self, user_id: str) -> Optional[str]:
        """Retrieves and refreshes user access token if needed."""
        repo = get_metadata_repo()
        tokens = repo.get_oauth_tokens(user_id, "google")
        if not tokens:
            return None

        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")

        if not access_token and not refresh_token:
            return None

        # Check if we have a refresh token and need to refresh or if we can use existing access token
        if refresh_token and self.client_id and self.client_secret:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        "https://oauth2.googleapis.com/token",
                        data={
                            "client_id": self.client_id,
                            "client_secret": self.client_secret,
                            "refresh_token": refresh_token,
                            "grant_type": "refresh_token",
                        },
                    )
                    if resp.status_code == 200:
                        new_data = resp.json()
                        access_token = new_data.get("access_token", access_token)
                        tokens["access_token"] = access_token
                        repo.save_oauth_tokens(user_id, "google", tokens)
            except Exception as e:
                logger.warning(f"Failed to refresh Google OAuth token for {user_id}: {e}")

        return access_token

    async def upload_file(
        self,
        owner_id: str,
        filename: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """
        Uploads document bytes to user's Google Drive.
        Returns the created Google Drive file ID, or None if Drive storage is not active/fails.
        """
        access_token = await self.get_valid_access_token(owner_id)
        if not access_token:
            return None

        try:
            metadata = {
                "name": filename,
                "description": "Uploaded via Unified Enterprise RAG System",
                "appProperties": {
                    "source": "p06_rag_system",
                    "owner_id": owner_id,
                },
            }

            boundary = "==============drive_upload_boundary_p06=="
            body = (
                f"--{boundary}\r\n"
                f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
                f"{json.dumps(metadata)}\r\n"
                f"--{boundary}\r\n"
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8") + content + f"\r\n--{boundary}--\r\n".encode("utf-8")

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
                    headers=headers,
                    content=body,
                )
                if response.status_code in (200, 201):
                    data = response.json()
                    return data.get("id")
                else:
                    logger.warning(f"Google Drive upload returned {response.status_code}: {response.text}")
                    return None
        except Exception as e:
            logger.error(f"Error uploading file to Google Drive: {e}")
            return None

    async def delete_file(self, file_id: str, user_id: Optional[str] = None) -> bool:
        """Deletes a file from Google Drive."""
        if not file_id:
            return True

        # If user_id is provided, try to use their token
        access_token = None
        if user_id:
            access_token = await self.get_valid_access_token(user_id)

        if not access_token:
            # Without active access token, we gracefully return True since local vector/metadata cleanup succeeds
            return True

        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.delete(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}",
                    headers=headers,
                )
                return resp.status_code in (200, 204, 404)
        except Exception as e:
            logger.warning(f"Error deleting file {file_id} from Google Drive: {e}")
            return False

    async def get_file_metadata(self, file_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves metadata of a Google Drive file."""
        if not file_id:
            return None

        access_token = None
        if user_id:
            access_token = await self.get_valid_access_token(user_id)

        if not access_token:
            return None

        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=id,name,mimeType,size,webViewLink,createdTime",
                    headers=headers,
                )
                if resp.status_code == 200:
                    return resp.json()
                return None
        except Exception as e:
            logger.warning(f"Error getting file metadata for {file_id}: {e}")
            return None
