"""
Vector Store Architecture for Unified Enterprise RAG System.

Supports:
1. QdrantVectorStore: Cloud-backed persistent vector index on Qdrant Cloud with owner isolation.
2. InMemoryVectorStore: In-memory vector index for standalone testing, development, and fallback.
3. Deterministic point ID generation and idempotent duplicate protection.
4. Strict multi-user document isolation, document lifecycle management, and provider collection routing.
"""

from datetime import datetime, timezone
import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np
from rag.chunker import Chunk
from config.settings import Config, ProviderVectorSpec
from utils.logging import logger
from utils.security import sanitize_error_message


def generate_point_id(
    provider_id: str,
    filename: str,
    chunk_id: str,
    content: str,
    owner_id: str = "default_user",
    doc_id: str = "",
) -> str:
    """
    Generates a deterministic UUIDv5 identifier for a chunk point with owner scoping.
    Ensures idempotent duplicate-ingestion protection across indexing cycles.
    """
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    unique_key = f"{provider_id}:{owner_id}:{doc_id}:{filename}:{chunk_id}:{content_hash}"
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
    """In-memory vector storage with deterministic point IDs, owner isolation, and lifecycle methods."""

    def __init__(self):
        self._records_by_id: Dict[str, Dict[str, Any]] = {}
        self._embedding_dimension: Optional[int] = None
        self._active_provider_id: Optional[str] = None

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        provider_id: str,
        owner_id: str = "default_user",
    ) -> int:
        """
        Stores text chunks paired with their vector embeddings in memory.
        Enforces provider identity, dimension consistency, owner scoping, and deterministic deduplication.
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
            doc_id = chunk.metadata.get("doc_id", f"doc_{filename}")
            chunk_owner = chunk.metadata.get("owner_id", owner_id)
            created_at = chunk.metadata.get("created_at", datetime.now(timezone.utc).isoformat())

            point_id = generate_point_id(provider_id, filename, chunk_id, chunk.content, owner_id=chunk_owner, doc_id=doc_id)

            record = {
                "id": point_id,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "owner_id": chunk_owner,
                "content": chunk.content,
                "embedding": [float(x) for x in embedding],
                "metadata": chunk.metadata,
                "provider_id": provider_id,
                "filename": filename,
                "created_at": created_at,
            }
            self._records_by_id[point_id] = record
            added_or_updated += 1

        logger.info(
            f"Upserted {added_or_updated} vector record(s) in memory [dim={current_dim}, provider='{provider_id}', owner='{owner_id}']. "
            f"Total unique store count: {len(self._records_by_id)}"
        )
        return added_or_updated

    def list_documents(
        self,
        owner_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists active documents aggregated from vector records, scoped to owner_id if provided."""
        docs_map: Dict[str, Dict[str, Any]] = {}

        for rec in self._records_by_id.values():
            rec_owner = rec.get("owner_id", "default_user")
            rec_prov = rec.get("provider_id")

            if owner_id and rec_owner != owner_id:
                continue
            if provider_id and rec_prov and rec_prov != provider_id:
                continue

            doc_id = rec.get("doc_id") or rec.get("metadata", {}).get("doc_id") or f"doc_{rec.get('filename')}"
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": rec.get("filename", "unknown_doc"),
                    "owner_id": rec_owner,
                    "provider_id": rec_prov or self._active_provider_id or "unknown",
                    "chunk_count": 0,
                    "created_at": rec.get("created_at") or rec.get("metadata", {}).get("created_at") or datetime.now(timezone.utc).isoformat(),
                    "char_count": 0,
                }
            docs_map[doc_id]["chunk_count"] += 1
            docs_map[doc_id]["char_count"] += len(rec.get("content", ""))

        return sorted(list(docs_map.values()), key=lambda d: d.get("created_at", ""), reverse=True)

    def delete_document(
        self,
        doc_id: str,
        owner_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> int:
        """Deletes all chunks associated with a specific document ID (and matching owner_id if provided)."""
        to_delete = []
        for point_id, rec in self._records_by_id.items():
            rec_doc_id = rec.get("doc_id") or rec.get("metadata", {}).get("doc_id")
            rec_owner = rec.get("owner_id", "default_user")
            rec_prov = rec.get("provider_id")

            if rec_doc_id == doc_id:
                if owner_id and rec_owner != owner_id:
                    continue
                if provider_id and rec_prov and rec_prov != provider_id:
                    continue
                to_delete.append(point_id)

        for pid in to_delete:
            del self._records_by_id[pid]

        if not self._records_by_id:
            self._embedding_dimension = None
            self._active_provider_id = None

        logger.info(f"Deleted document '{doc_id}' ({len(to_delete)} chunks removed, owner='{owner_id}').")
        return len(to_delete)

    def clear_user_documents(
        self,
        owner_id: str,
        provider_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Clears all documents and chunks owned by a specific user."""
        docs_before = len(self.list_documents(owner_id=owner_id, provider_id=provider_id))
        to_delete = []

        for point_id, rec in self._records_by_id.items():
            rec_owner = rec.get("owner_id", "default_user")
            rec_prov = rec.get("provider_id")

            if rec_owner == owner_id:
                if provider_id and rec_prov and rec_prov != provider_id:
                    continue
                to_delete.append(point_id)

        for pid in to_delete:
            del self._records_by_id[pid]

        if not self._records_by_id:
            self._embedding_dimension = None
            self._active_provider_id = None

        logger.info(f"Cleared {len(to_delete)} chunks across {docs_before} documents for owner '{owner_id}'.")
        return docs_before, len(to_delete)

    def clear_store(self, provider_id: Optional[str] = None):
        """Administrative wipe: Clears all stored records and resets dimension and provider metadata."""
        count_before = len(self._records_by_id)
        self._records_by_id.clear()
        self._embedding_dimension = None
        self._active_provider_id = None
        logger.info(f"Admin cleared in-memory vector store ({count_before} records removed).")

    def get_records(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns stored vector records, optionally filtered by owner_id."""
        if owner_id is None:
            return list(self._records_by_id.values())
        return [r for r in self._records_by_id.values() if r.get("owner_id", "default_user") == owner_id]

    def count(self, owner_id: Optional[str] = None) -> int:
        """Returns vector record count, optionally scoped to owner_id."""
        if owner_id is None:
            return len(self._records_by_id)
        return sum(1 for r in self._records_by_id.values() if r.get("owner_id", "default_user") == owner_id)

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns active embedding vector dimension."""
        return self._embedding_dimension

    def get_active_provider_id(self) -> Optional[str]:
        """Returns active provider ID used for current vector index."""
        return self._active_provider_id

    def get_stats(
        self,
        provider_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns vector store status statistics scoped to user."""
        target_provider = provider_id or self._active_provider_id or "None"
        user_count = self.count(owner_id=owner_id)
        return {
            "count": user_count,
            "dimension": self._embedding_dimension or "N/A",
            "provider_id": target_provider,
            "store_type": "In-Memory NumPy Vector Index",
            "status": "Indexed" if user_count > 0 else "Empty",
            "user_id": owner_id or "all",
        }


class QdrantVectorStore:
    """Qdrant Cloud vector database store enforcing collection routing, owner isolation, and lifecycle operations."""

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

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        provider_id: str,
        owner_id: str = "default_user",
    ) -> int:
        """
        Ingests document chunks into provider-specific Qdrant Cloud collection with owner metadata.
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
            doc_id = chunk.metadata.get("doc_id", "")
            chunk_owner = chunk.metadata.get("owner_id", owner_id)
            created_at = chunk.metadata.get("created_at", datetime.now(timezone.utc).isoformat())

            point_id = generate_point_id(provider_id, filename, chunk_id, chunk.content, owner_id=chunk_owner, doc_id=doc_id)

            payload = {
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "owner_id": chunk_owner,
                "provider_id": provider_id,
                "embedding_model": spec.embedding_model,
                "filename": filename,
                "created_at": created_at,
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
                f"[dim={spec.dimension}, provider='{provider_id}', owner='{owner_id}']."
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
        owner_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search in provider-specific Qdrant collection with owner isolation.
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

        from qdrant_client import models

        # Build payload query filter for owner isolation and provider matching
        filter_conditions = []
        if owner_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="owner_id",
                    match=models.MatchValue(value=owner_id)
                )
            )

        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        try:
            response = client.query_points(
                collection_name=spec.collection_name,
                query=[float(x) for x in query_vector],
                query_filter=query_filter,
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
                    "chunk_id": payload.get("chunk_id", str(pt.id)),
                    "doc_id": payload.get("doc_id", ""),
                    "owner_id": payload.get("owner_id", "default_user"),
                })

            logger.info(
                f"Retrieved {len(scored_chunks)} point(s) from Qdrant collection '{spec.collection_name}' "
                f"(provider: '{provider_id}', owner: '{owner_id}')."
            )
            return scored_chunks

        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Qdrant search error: {clean_err}")
            raise RuntimeError(f"Qdrant Search Error: {clean_err}")

    def count(self, provider_id: Optional[str] = None, owner_id: Optional[str] = None) -> int:
        """Returns total vector point count for active or specified provider collection, scoped to owner_id."""
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

            if owner_id:
                from qdrant_client import models
                count_filter = models.Filter(
                    must=[models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id))]
                )
                count_result = client.count(collection_name=spec.collection_name, count_filter=count_filter)
            else:
                count_result = client.count(collection_name=spec.collection_name)
            return count_result.count
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.warning(f"Unable to count points for collection '{spec.collection_name}': {clean_err}")
            return 0

    def list_documents(
        self,
        owner_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Scrolls and aggregates documents from Qdrant collection payload, scoped to owner_id."""
        target_provider = provider_id or self._active_provider_id or Config.get_ai_provider()
        spec = Config.get_provider_spec(target_provider)
        if not spec:
            return []

        try:
            client = self._get_client()
            if not client.collection_exists(collection_name=spec.collection_name):
                return []

            from qdrant_client import models
            scroll_filter = None
            if owner_id:
                scroll_filter = models.Filter(
                    must=[models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id))]
                )

            docs_map: Dict[str, Dict[str, Any]] = {}
            offset = None

            while True:
                records, next_offset = client.scroll(
                    collection_name=spec.collection_name,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for pt in records:
                    payload = pt.payload or {}
                    doc_id = payload.get("doc_id") or payload.get("metadata", {}).get("doc_id") or f"doc_{payload.get('filename')}"
                    rec_owner = payload.get("owner_id", "default_user")

                    if doc_id not in docs_map:
                        docs_map[doc_id] = {
                            "doc_id": doc_id,
                            "filename": payload.get("filename", "unknown_doc"),
                            "owner_id": rec_owner,
                            "provider_id": payload.get("provider_id", target_provider),
                            "chunk_count": 0,
                            "created_at": payload.get("created_at") or payload.get("metadata", {}).get("created_at") or datetime.now(timezone.utc).isoformat(),
                            "char_count": 0,
                        }
                    docs_map[doc_id]["chunk_count"] += 1
                    docs_map[doc_id]["char_count"] += len(payload.get("content", ""))

                if next_offset is None:
                    break
                offset = next_offset

            return sorted(list(docs_map.values()), key=lambda d: d.get("created_at", ""), reverse=True)
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Error listing documents from Qdrant collection: {clean_err}")
            return []

    def delete_document(
        self,
        doc_id: str,
        owner_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> int:
        """Deletes all vector points belonging to a specific document ID (and matching owner_id)."""
        target_provider = provider_id or self._active_provider_id or Config.get_ai_provider()
        spec = self._validate_and_get_spec(target_provider)
        client = self._get_client()

        if not client.collection_exists(collection_name=spec.collection_name):
            return 0

        from qdrant_client import models

        filter_conditions = [
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        ]
        if owner_id:
            filter_conditions.append(
                models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id))
            )

        doc_filter = models.Filter(must=filter_conditions)

        try:
            count_res = client.count(collection_name=spec.collection_name, count_filter=doc_filter)
            del_count = count_res.count
            if del_count > 0:
                client.delete(
                    collection_name=spec.collection_name,
                    points_selector=models.FilterSelector(filter=doc_filter),
                    wait=True,
                )
            logger.info(f"Deleted {del_count} points for doc '{doc_id}' (owner='{owner_id}') from Qdrant.")
            return del_count
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Qdrant document delete error: {clean_err}")
            raise RuntimeError(f"Qdrant Document Delete Error: {clean_err}")

    def clear_user_documents(
        self,
        owner_id: str,
        provider_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Clears all points owned by a specific user in the provider collection."""
        target_provider = provider_id or self._active_provider_id or Config.get_ai_provider()
        spec = self._validate_and_get_spec(target_provider)
        client = self._get_client()

        if not client.collection_exists(collection_name=spec.collection_name):
            return 0, 0

        docs_list = self.list_documents(owner_id=owner_id, provider_id=target_provider)
        doc_count = len(docs_list)

        from qdrant_client import models
        user_filter = models.Filter(
            must=[models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id))]
        )

        try:
            count_res = client.count(collection_name=spec.collection_name, count_filter=user_filter)
            chunk_count = count_res.count
            if chunk_count > 0:
                client.delete(
                    collection_name=spec.collection_name,
                    points_selector=models.FilterSelector(filter=user_filter),
                    wait=True,
                )
            logger.info(f"Purged {chunk_count} chunks across {doc_count} docs for owner '{owner_id}' from Qdrant.")
            return doc_count, chunk_count
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Qdrant user clear error: {clean_err}")
            raise RuntimeError(f"Qdrant User Clear Error: {clean_err}")

    def clear_store(self, provider_id: Optional[str] = None):
        """Administrative wipe: Clears all indexed vector points from provider collection on Qdrant Cloud."""
        target_provider = provider_id or self._active_provider_id
        if not target_provider:
            self._active_provider_id = None
            self._embedding_dimension = None
            logger.info("Cleared Qdrant vector store session metadata (no provider resolved).")
            return

        spec = self._validate_and_get_spec(target_provider)
        client = self._get_client()

        if not client.collection_exists(collection_name=spec.collection_name):
            self._active_provider_id = None
            self._embedding_dimension = None
            logger.warning(
                f"Collection '{spec.collection_name}' does not exist on Qdrant cluster. Skipped point deletion."
            )
            return

        try:
            from qdrant_client import models
            client.delete(
                collection_name=spec.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                ),
                wait=True,
            )
            self._active_provider_id = None
            self._embedding_dimension = None
            logger.info(
                f"Admin cleared all points from Qdrant collection '{spec.collection_name}' for provider '{target_provider}'."
            )
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Qdrant clear error on collection '{spec.collection_name}': {clean_err}")
            raise RuntimeError(f"Qdrant Clear Error: {clean_err}")

    def get_embedding_dimension(self) -> Optional[int]:
        """Returns active embedding dimension."""
        return self._embedding_dimension

    def get_active_provider_id(self) -> Optional[str]:
        """Returns active provider ID."""
        return self._active_provider_id

    def get_records(self) -> List[Dict[str, Any]]:
        """Returns empty list for Qdrant (full record dump avoided on cloud store)."""
        return []

    def get_stats(
        self,
        provider_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns vector store status statistics scoped to user."""
        target_provider = provider_id or self._active_provider_id or "None"
        spec = Config.get_provider_spec(target_provider) if target_provider != "None" else None
        point_count = self.count(provider_id=target_provider, owner_id=owner_id) if spec else 0

        return {
            "count": point_count,
            "dimension": spec.dimension if spec else (self._embedding_dimension or "N/A"),
            "provider_id": target_provider,
            "collection_name": spec.collection_name if spec else "N/A",
            "store_type": "Qdrant Cloud Vector Database",
            "status": "Indexed" if point_count > 0 else "Ready",
            "user_id": owner_id or "all",
        }
