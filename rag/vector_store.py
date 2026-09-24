"""
Vector Store Architecture for Unified Enterprise RAG System.

Supports:
1. QdrantVectorStore: Cloud-backed persistent vector index on Qdrant Cloud.
2. InMemoryVectorStore: In-memory vector index for standalone testing and local fallback.
3. Deterministic point ID generation and idempotent duplicate protection.
4. Strict provider identity enforcement and collection routing.
"""

import hashlib
import math
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
from rag.chunker import Chunk
from config.settings import Config, ProviderVectorSpec
from utils.logging import logger
from utils.security import sanitize_error_message


def generate_point_id(provider_id: str, filename: str, chunk_id: str, content: str) -> str:
    """
    Generates a deterministic UUIDv5 identifier for a chunk point.
    Ensures idempotent duplicate-ingestion protection across indexing cycles.
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    unique_key = f"{provider_id}:{filename}:{chunk_id}:{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_key))


def _validate_embeddings(embeddings: List[List[float]], expected_dim: int, provider_id: str):
    """Validates vector array structure, finiteness, and dimension consistency."""
    if not embeddings:
        return
    for idx, vec in enumerate(embeddings):
        if not isinstance(vec, (list, tuple)):
            raise ValueError(f"Vector at index {idx} must be a list or tuple of numbers.")
        if len(vec) != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch for provider '{provider_id}'! "
                f"Expected dimension {expected_dim}, but vector at index {idx} has dimension {len(vec)}."
            )
        for v_idx, val in enumerate(vec):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(
                    f"Non-numeric value in vector at chunk {idx}, element {v_idx} for provider '{provider_id}'."
                )
            if not math.isfinite(float(val)):
                raise ValueError(
                    f"Non-finite value (NaN/Inf) in vector at chunk {idx}, element {v_idx} for provider '{provider_id}'."
                )


class InMemoryVectorStore:
    """In-memory vector storage with deterministic point IDs and provider isolation."""

    def __init__(self):
        self._records_by_id: Dict[str, Dict[str, Any]] = {}
        self._embedding_dimension: Optional[int] = None
        self._active_provider_id: Optional[str] = None

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], provider_id: str) -> int:
        """
        Stores text chunks paired with their vector embeddings in memory.
        Enforces provider identity, dimension consistency, and deterministic deduplication.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        if not embeddings:
            return 0

        spec = Config.get_provider_spec(provider_id)
        current_dim = spec.dimension if spec else len(embeddings[0])

        # Enforce provider identity isolation
        if self._records_by_id and self._active_provider_id is not None:
            if provider_id != self._active_provider_id:
                raise ValueError(
                    f"Provider mismatch! Store has vectors from provider '{self._active_provider_id}', "
                    f"but new embeddings are from provider '{provider_id}'. Clear index before switching providers."
                )
            if self._embedding_dimension is not None and current_dim != self._embedding_dimension:
                raise ValueError(
                    f"Vector dimension mismatch! Store has vectors of dimension {self._embedding_dimension} "
                    f"from provider '{self._active_provider_id}', but new embeddings are dimension {current_dim} "
                    f"from provider '{provider_id}'. Clear index before switching providers."
                )

        _validate_embeddings(embeddings, current_dim, provider_id)

        self._embedding_dimension = current_dim
        self._active_provider_id = provider_id

        added_or_updated = 0
        for chunk, embedding in zip(chunks, embeddings):
            filename = chunk.metadata.get("filename", "unknown_doc")
            chunk_id = chunk.metadata.get("chunk_id", f"chunk_{len(self._records_by_id)}")
            point_id = generate_point_id(provider_id, filename, chunk_id, chunk.content)

            record = {
                "id": point_id,
                "chunk_id": chunk_id,
                "content": chunk.content,
                "embedding": [float(x) for x in embedding],
                "metadata": chunk.metadata,
                "provider_id": provider_id,
                "filename": filename
            }
            self._records_by_id[point_id] = record
            added_or_updated += 1

        logger.info(
            f"Upserted {added_or_updated} vector record(s) in memory [dim={current_dim}, provider='{provider_id}']. "
            f"Total unique store count: {len(self._records_by_id)}"
        )
        return added_or_updated

    def clear_store(self):
        """Clears all stored records and resets dimension and provider metadata."""
        count_before = len(self._records_by_id)
        self._records_by_id.clear()
        self._embedding_dimension = None
        self._active_provider_id = None
        logger.info(f"Cleared in-memory vector store ({count_before} records removed).")

    def get_records(self) -> List[Dict[str, Any]]:
        """Returns all stored vector records."""
        return list(self._records_by_id.values())

    def count(self) -> int:
        """Returns total unique vector record count."""
        return len(self._records_by_id)

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns active embedding vector dimension."""
        return self._embedding_dimension

    def get_active_provider_id(self) -> Optional[str]:
        """Returns active provider ID used for current vector index."""
        return self._active_provider_id

    def get_stats(self, provider_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns vector store status statistics."""
        target_provider = provider_id or self._active_provider_id or "None"
        return {
            "count": len(self._records_by_id),
            "dimension": self._embedding_dimension or "N/A",
            "provider_id": target_provider,
            "store_type": "In-Memory NumPy Vector Index",
            "status": "Indexed" if len(self._records_by_id) > 0 else "Empty"
        }


class QdrantVectorStore:
    """Qdrant Cloud vector database store enforcing collection routing and provider isolation."""

    def __init__(self, client: Optional[Any] = None, active_provider_id: Optional[str] = None):
        self._client = client
        self._active_provider_id: Optional[str] = active_provider_id
        self._embedding_dimension: Optional[int] = None

    def _get_client(self):
        """Returns initialized QdrantClient instance or raises sanitized error if unconfigured."""
        if self._client is not None:
            return self._client

        if not Config.is_qdrant_configured():
            raise RuntimeError(
                "Qdrant Cloud is not configured. Set QDRANT_URL and QDRANT_API_KEY in environment or settings."
            )

        url = Config.get_qdrant_url()
        api_key = Config.get_qdrant_api_key()

        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=url, api_key=api_key)
            return self._client
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Failed to initialize QdrantClient: {clean_err}")
            raise RuntimeError(f"Qdrant Client Initialization Error: {clean_err}")

    def _validate_and_get_spec(self, provider_id: str) -> ProviderVectorSpec:
        """Retrieves and validates provider vector spec."""
        spec = Config.get_provider_spec(provider_id)
        if not spec:
            raise ValueError(f"Unknown or unsupported provider '{provider_id}' for Qdrant storage.")
        return spec

    def _verify_collection_exists(self, client: Any, collection_name: str, provider_id: str):
        """Verifies that collection exists on Qdrant Cloud. Does NOT auto-create collections."""
        try:
            exists = client.collection_exists(collection_name=collection_name)
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Error checking collection '{collection_name}': {clean_err}")
            raise RuntimeError(f"Qdrant Collection Check Error: {clean_err}")

        if not exists:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' for provider '{provider_id}' does not exist on cluster. "
                f"Please provision the collection in Qdrant Cloud before indexing documents."
            )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], provider_id: str) -> int:
        """
        Ingests document chunks into provider-specific Qdrant Cloud collection.
        Uses deterministic UUIDv5 point IDs for idempotent duplicate protection.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        if not embeddings:
            return 0

        spec = self._validate_and_get_spec(provider_id)

        # Enforce provider identity isolation
        if self._active_provider_id is not None and self._active_provider_id != provider_id:
            raise ValueError(
                f"Provider mismatch! Active index is configured for provider '{self._active_provider_id}', "
                f"but new embeddings are from provider '{provider_id}'. Clear index before switching providers."
            )

        _validate_embeddings(embeddings, spec.dimension, provider_id)

        client = self._get_client()
        self._verify_collection_exists(client, spec.collection_name, provider_id)

        from qdrant_client import models

        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            filename = chunk.metadata.get("filename", "unknown_doc")
            chunk_id = chunk.metadata.get("chunk_id", f"chunk_{idx}")
            point_id = generate_point_id(provider_id, filename, chunk_id, chunk.content)

            payload = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk_id,
                "provider_id": provider_id,
                "embedding_model": spec.embedding_model,
                "filename": filename
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[float(x) for x in embedding],
                    payload=payload
                )
            )

        try:
            client.upsert(
                collection_name=spec.collection_name,
                points=points,
                wait=True
            )
            self._active_provider_id = provider_id
            self._embedding_dimension = spec.dimension
            logger.info(
                f"Upserted {len(points)} point(s) into Qdrant collection '{spec.collection_name}' "
                f"[dim={spec.dimension}, provider='{provider_id}']."
            )
            return len(points)
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Qdrant upsert error: {clean_err}")
            raise RuntimeError(f"Qdrant Ingestion Error: {clean_err}")

    def search(
        self,
        query_vector: List[float],
        provider_id: str,
        top_k: int = 5,
        similarity_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search in provider-specific Qdrant collection.
        Returns top-K results exceeding similarity threshold.
        """
        spec = self._validate_and_get_spec(provider_id)

        # Enforce provider identity isolation
        if self._active_provider_id is not None and self._active_provider_id != provider_id:
            raise ValueError(
                f"Provider mismatch! Active index is configured for provider '{self._active_provider_id}', "
                f"but query embedding was generated for provider '{provider_id}'."
            )

        if len(query_vector) != spec.dimension:
            raise ValueError(
                f"Vector dimension mismatch! Expected query dimension {spec.dimension} for provider '{provider_id}', "
                f"but received {len(query_vector)}."
            )

        client = self._get_client()
        self._verify_collection_exists(client, spec.collection_name, provider_id)

        try:
            # Query Qdrant with score threshold filtering
            response = client.query_points(
                collection_name=spec.collection_name,
                query=[float(x) for x in query_vector],
                limit=top_k,
                score_threshold=similarity_threshold
            )

            scored_chunks = []
            for pt in response.points:
                payload = pt.payload or {}
                scored_chunks.append({
                    "content": payload.get("content", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": round(float(pt.score), 4),
                    "chunk_id": payload.get("chunk_id", str(pt.id))
                })

            logger.info(
                f"Retrieved {len(scored_chunks)} point(s) from Qdrant collection '{spec.collection_name}' "
                f"(provider: '{provider_id}')."
            )
            return scored_chunks

        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Qdrant search error: {clean_err}")
            raise RuntimeError(f"Qdrant Search Error: {clean_err}")

    def count(self, provider_id: Optional[str] = None) -> int:
        """Returns total vector point count for active or specified provider collection."""
        target_provider = provider_id or self._active_provider_id
        if not target_provider:
            return 0

        spec = Config.get_provider_spec(target_provider)
        if not spec:
            return 0

        try:
            client = self._get_client()
            if not client.collection_exists(collection_name=spec.collection_name):
                return 0
            count_result = client.count(collection_name=spec.collection_name)
            return count_result.count
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.warning(f"Unable to count points for collection '{spec.collection_name}': {clean_err}")
            return 0

    def clear_store(self, provider_id: Optional[str] = None):
        """Clears active index session tracking."""
        self._active_provider_id = None
        self._embedding_dimension = None
        logger.info("Cleared Qdrant vector store session metadata.")

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns active embedding dimension."""
        return self._embedding_dimension

    def get_active_provider_id(self) -> Optional[str]:
        """Returns active provider ID."""
        return self._active_provider_id

    def get_records(self) -> List[Dict[str, Any]]:
        """Returns empty list for Qdrant (full record dump avoided on cloud store)."""
        return []

    def get_stats(self, provider_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns vector store status statistics."""
        target_provider = provider_id or self._active_provider_id or "None"
        spec = Config.get_provider_spec(target_provider) if target_provider != "None" else None
        point_count = self.count(target_provider) if spec else 0

        return {
            "count": point_count,
            "dimension": spec.dimension if spec else (self._embedding_dimension or "N/A"),
            "provider_id": target_provider,
            "collection_name": spec.collection_name if spec else "N/A",
            "store_type": "Qdrant Cloud Vector Database",
            "status": "Indexed" if point_count > 0 else "Ready"
        }
