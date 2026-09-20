"""
Semantic Vector Retriever calculating exact Cosine Similarity using NumPy.
"""

from typing import List, Dict, Any, Union
import numpy as np
from rag.vector_store import InMemoryVectorStore, QdrantVectorStore
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger


class SemanticRetriever:
    """Retriever computing cosine similarity against stored vectors or querying Qdrant Cloud."""

    def __init__(self, vector_store: Union[InMemoryVectorStore, QdrantVectorStore, Any]):
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
        Executes semantic vector search against in-memory index or Qdrant Cloud.
        1. Validates provider identity against vector store.
        2. Embeds search query using active AI provider.
        3. Executes search with score threshold filtering and top-K ranking.
        """
        if not query.strip():
            return []

        # Provider identity enforcement
        active_p = self.vector_store.get_active_provider_id()
        if active_p is not None and active_p != provider.provider_id:
            raise ValueError(
                f"Provider mismatch! Document index was generated with provider '{active_p}', "
                f"but query was submitted with provider '{provider.provider_id}'. "
                f"Please re-index documents or switch provider."
            )

        # Generate query vector embedding via active provider
        logger.info(f"Generating query embedding via '{provider.name}' ({provider.get_embedding_model_name()})")
        query_embedding = provider.embed_query(query)

        # Check dimension consistency
        spec = Config.get_provider_spec(provider.provider_id)
        expected_dim = spec.dimension if spec else self.vector_store.get_embedding_dimension()
        if expected_dim is not None and len(query_embedding) != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch! Provider '{provider.provider_id}' requires dimension {expected_dim}, "
                f"but query embedding has dimension {len(query_embedding)}."
            )

        # If vector store has a search method (e.g. QdrantVectorStore), delegate directly
        if hasattr(self.vector_store, "search") and callable(getattr(self.vector_store, "search")):
            return self.vector_store.search(
                query_vector=query_embedding,
                provider_id=provider.provider_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )

        # In-memory cosine similarity search
        records = self.vector_store.get_records()
        if not records:
            logger.info("Retrieval requested on empty vector index.")
            return []

        scored_chunks = []
        for rec in records:
            # Enforce provider match on individual records
            if rec.get("provider_id") and rec.get("provider_id") != provider.provider_id:
                continue

            doc_embedding = rec["embedding"]
            sim_score = self.cosine_similarity(query_embedding, doc_embedding)

            if sim_score >= similarity_threshold:
                scored_chunks.append({
                    "content": rec["content"],
                    "metadata": rec["metadata"],
                    "score": round(sim_score, 4),
                    "chunk_id": rec.get("chunk_id", rec["id"])
                })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        top_results = scored_chunks[:top_k]

        logger.info(
            f"Retrieved {len(top_results)} chunk(s) from {len(records)} evaluated vector(s) "
            f"using provider '{provider.name}'."
        )
        return top_results
