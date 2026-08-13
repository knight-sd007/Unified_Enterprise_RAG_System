"""
Document Chunker supporting sliding window text splitting with overlap.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from rag.loaders import Document
from utils.logging import logger


@dataclass
class Chunk:
    """Document chunk with metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Configurable text chunker preserving metadata across splits."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 1))

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Splits Document objects into Chunk objects."""
        all_chunks = []
        global_idx = 0

        for doc in documents:
            text = doc.content
            if not text:
                continue

            splits = self._split_text(text)
            for sub_idx, chunk_text in enumerate(splits):
                global_idx += 1
                meta = doc.metadata.copy()
                meta.update({
                    "chunk_id": f"{meta.get('filename', 'doc')}_c{global_idx}",
                    "sub_chunk_index": sub_idx
                })
                all_chunks.append(Chunk(content=chunk_text, metadata=meta))

        logger.info(f"Chunked {len(documents)} document(s) into {len(all_chunks)} chunk(s).")
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Splits raw text using character sliding window with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = start + self.chunk_size

            if end < length:
                boundary = text.rfind("\n", start, end)
                if boundary == -1 or boundary < start + (self.chunk_size // 2):
                    boundary = text.rfind(". ", start, end)
                if boundary == -1 or boundary < start + (self.chunk_size // 2):
                    boundary = text.rfind(" ", start, end)
                if boundary != -1 and boundary > start:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap
            if start >= length or end >= length:
                break

        return chunks
