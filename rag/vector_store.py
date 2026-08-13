"""
In-Memory Vector Store for Unified Enterprise RAG System.

Maintains document embeddings and text chunks directly in Python memory.
Uses NumPy vector dot product and norm operations for exact cosine similarity search.
"""

from typing import List, Dict, Any, Optional
from rag.chunker import Chunk
from utils.logging import logger


class InMemoryVectorStore:
    """In-memory vector storage and metadata record manager."""

    def __init__(self):
        self._records: List[Dict[str, Any]] = []
        self._embedding_dimension: Optional[int] = None
        self._active_provider_id: Optional[str] = None

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], provider_id: str) -> int:
        """
        Stores text chunks paired with their vector embeddings in memory.
        Validates embedding dimensions for consistency.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        if not embeddings:
            return 0

        current_dim = len(embeddings[0])

        # Enforce dimension consistency if vectors already exist
        if self._records and self._embedding_dimension is not None:
            if current_dim != self._embedding_dimension:
                raise ValueError(
                    f"Vector dimension mismatch! Store has vectors of dimension {self._embedding_dimension} "
                    f"from provider '{self._active_provider_id}', but new embeddings are dimension {current_dim} "
                    f"from provider '{provider_id}'. Clear index before switching providers."
                )

        self._embedding_dimension = current_dim
        self._active_provider_id = provider_id

        added_count = 0
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                "id": chunk.metadata.get("chunk_id", f"chunk_{len(self._records)}"),
                "content": chunk.content,
                "embedding": embedding,
                "metadata": chunk.metadata,
                "provider_id": provider_id
            }
            self._records.append(record)
            added_count += 1

        logger.info(
            f"Added {added_count} vector record(s) [dim={current_dim}, provider='{provider_id}']. "
            f"Total store count: {len(self._records)}"
        )
        return added_count

    def clear_store(self):
        """Clears all stored records and resets dimension metadata."""
        count_before = len(self._records)
        self._records.clear()
        self._embedding_dimension = None
        self._active_provider_id = None
        logger.info(f"Cleared in-memory vector store ({count_before} records removed).")

    def get_records(self) -> List[Dict[str, Any]]:
        """Returns all stored vector records."""
        return self._records

    def count(self) -> int:
        """Returns total vector record count."""
        return len(self._records)

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns active embedding vector dimension."""
        return self._embedding_dimension

    def get_active_provider_id(self) -> Optional[str]:
        """Returns active provider ID used for current vector index."""
        return self._active_provider_id

    def get_stats(self) -> Dict[str, Any]:
        """Returns vector store status statistics."""
        return {
            "count": len(self._records),
            "dimension": self._embedding_dimension or "N/A",
            "provider_id": self._active_provider_id or "None",
            "store_type": "In-Memory NumPy Vector Index",
            "status": "Indexed" if len(self._records) > 0 else "Empty"
        }
