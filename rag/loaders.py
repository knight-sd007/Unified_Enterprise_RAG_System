"""
Document Loader for TXT and PDF files with multi-user ownership metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import os
import uuid
from utils.logging import logger
from utils.security import sanitize_error_message


@dataclass
class Document:
    """Represents ingested document content and ownership metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Returns document ID if present in metadata."""
        return self.metadata.get("doc_id", "")

    @property
    def owner_id(self) -> str:
        """Returns owner user ID if present in metadata."""
        return self.metadata.get("owner_id", "default_user")


class DocumentLoader:
    """Handles parsing TXT and PDF uploaded files with ownership binding."""

    @staticmethod
    def load_from_text(
        text: str,
        filename: str = "user_input.txt",
        owner_id: str = "default_user",
        doc_id: Optional[str] = None,
    ) -> Document:
        """Creates Document from raw text input with ownership."""
        assigned_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        return Document(
            content=text.strip(),
            metadata={
                "doc_id": assigned_doc_id,
                "owner_id": owner_id,
                "filename": filename,
                "source": "raw_text",
                "char_count": len(text),
                "created_at": now_iso,
            }
        )

    @staticmethod
    def load_from_file(
        file_bytes: bytes,
        filename: str,
        owner_id: str = "default_user",
        doc_id: Optional[str] = None,
    ) -> Document:
        """Parses file bytes (PDF or TXT) into Document with ownership."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = DocumentLoader._extract_pdf_text(file_bytes, filename)
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

        assigned_doc_id = doc_id or f"doc_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        return Document(
            content=text.strip(),
            metadata={
                "doc_id": assigned_doc_id,
                "owner_id": owner_id,
                "filename": filename,
                "extension": ext,
                "char_count": len(text),
                "source": "file_upload",
                "created_at": now_iso,
            }
        )

    @staticmethod
    def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
        """Extracts text from PDF bytes using PyPDF."""
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append(txt)
            logger.info(f"Extracted {len(pages_text)} page(s) from PDF '{filename}'")
            return "\n\n".join(pages_text)
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Error parsing PDF '{filename}': {clean_err}")
            raise ValueError(f"Failed to parse PDF file '{filename}': {clean_err}")
