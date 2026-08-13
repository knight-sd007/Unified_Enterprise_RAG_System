"""
Semantic Vector Retriever calculating exact Cosine Similarity using NumPy.
"""

from typing import List, Dict, Any
import numpy as np
from rag.vector_store import InMemoryVectorStore
from providers.base import BaseAIProvider
from utils.logging import logger


class SemanticRetriever:
    """Retriever computing exact cosine similarity scores against stored vector embeddings."""

    def __init__(self, vector_store: InMemoryVectorStore):
        self.vector_store = vector_store

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculates exact Cosine Similarity between two floating-point vector arrays using NumPy.
        Similarity = (vec_a · vec_b) / (||vec_a|| * ||vec_b||)
        """
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        similarity = np.dot(a, b) / (norm_a * norm_b)
        return float(similarity)

    def retrieve(
        self,
        query: str,
        provider: BaseAIProvider,
        top_k: int = 5,
        similarity_threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Executes in-memory semantic vector search.
        1. Embeds search query using active AI provider.
        2. Validates vector dimension compatibility against store.
        3. Computes cosine similarity against all stored document vectors.
        4. Returns top_k records exceeding similarity threshold.
        """
        if not query.strip():
            return []

        records = self.vector_store.get_records()
        if not records:
            logger.info("Retrieval requested on empty vector index.")
            return []

        # Generate query vector embedding via active provider
        logger.info(f"Generating query embedding via '{provider.name}' ({provider.get_embedding_model_name()})")
        query_embedding = provider.embed_query(query)

        # Dimension compatibility check
        store_dim = self.vector_store.get_embedding_dimension()
        if store_dim is not None and len(query_embedding) != store_dim:
            active_p = self.vector_store.get_active_provider_id()
            raise ValueError(
                f"Vector dimension mismatch! Document index was generated with provider '{active_p}' "
                f"(dimension {store_dim}), but query embedding generated with '{provider.provider_id}' "
                f"has dimension {len(query_embedding)}. Please re-index documents with current provider."
            )

        # Calculate cosine similarity against all vectors
        scored_chunks = []
        for rec in records:
            doc_embedding = rec["embedding"]
            sim_score = self.cosine_similarity(query_embedding, doc_embedding)

            if sim_score >= similarity_threshold:
                scored_chunks.append({
                    "content": rec["content"],
                    "metadata": rec["metadata"],
                    "score": round(sim_score, 4),
                    "chunk_id": rec["id"]
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_chunks[:top_k]

        logger.info(
            f"Retrieved {len(top_results)} chunk(s) from {len(records)} evaluated vector(s) "
            f"using provider '{provider.name}'."
        )
        return top_results
