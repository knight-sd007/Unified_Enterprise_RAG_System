"""
Document Loader for TXT and PDF files.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
from utils.logging import logger
from utils.security import sanitize_error_message


@dataclass
class Document:
    """Represents ingested document content and metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentLoader:
    """Handles parsing TXT and PDF uploaded files."""

    @staticmethod
    def load_from_text(text: str, filename: str = "user_input.txt") -> Document:
        """Creates Document from raw text input."""
        return Document(
            content=text.strip(),
            metadata={"filename": filename, "source": "raw_text", "char_count": len(text)}
        )

    @staticmethod
    def load_from_file(file_bytes: bytes, filename: str) -> Document:
        """Parses file bytes (PDF or TXT) into Document."""
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".pdf":
            text = DocumentLoader._extract_pdf_text(file_bytes, filename)
        else:
            text = file_bytes.decode("utf-8", errors="ignore")

        return Document(
            content=text.strip(),
            metadata={
                "filename": filename,
                "extension": ext,
                "char_count": len(text),
                "source": "file_upload"
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
