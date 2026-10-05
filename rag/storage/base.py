"""
Base storage abstraction for document lifecycle and raw file retention.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseDocumentStorage(ABC):
    """Abstract interface for external document file storage."""

    @abstractmethod
    async def upload_file(
        self,
        owner_id: str,
        filename: str,
        content: bytes,
        mime_type: str = "application/octet-stream",
    ) -> Optional[str]:
        """
        Uploads document bytes to external storage.
        Returns the external file_id or None if storage is disabled/skipped.
        """
        pass

    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """
        Deletes the file from external storage by file_id.
        Returns True if deleted or ignored, False on critical error.
        """
        pass

    @abstractmethod
    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves remote metadata for the file."""
        pass
