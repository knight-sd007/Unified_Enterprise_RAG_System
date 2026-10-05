"""
Storage abstractions and metadata persistence module.
"""

from rag.storage.base import BaseDocumentStorage
from rag.storage.metadata_db import MetadataRepository, get_metadata_repo
from rag.storage.google_drive import GoogleDriveStorage

__all__ = [
    "BaseDocumentStorage",
    "MetadataRepository",
    "get_metadata_repo",
    "GoogleDriveStorage",
]
