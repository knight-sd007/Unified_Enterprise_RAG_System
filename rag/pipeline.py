"""
RAG Pipeline Orchestrator.

Integrates Document Ingestion, Text Chunking, In-Memory Vector Indexing, Semantic Retrieval, and Provider Chat Generation.
"""

from typing import List, Dict, Any, Optional, Union
from rag.loaders import DocumentLoader, Document
from rag.chunker import DocumentChunker, Chunk
from rag.vector_store import InMemoryVectorStore, QdrantVectorStore
from rag.retriever import SemanticRetriever
from rag.prompt import RAG_SYSTEM_PROMPT, format_rag_prompt
from providers.base import BaseAIProvider
from config.settings import Config
from utils.logging import logger
from utils.security import sanitize_error_message


class RAGPipeline:
    """Orchestrates end-to-end RAG workflow with vector store abstraction."""

    def __init__(
        self,
        vector_store: Optional[Union[InMemoryVectorStore, QdrantVectorStore, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        if vector_store is not None:
            self.vector_store = vector_store
        elif Config.is_qdrant_configured():
            self.vector_store = QdrantVectorStore()
        else:
            self.vector_store = InMemoryVectorStore()

        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = SemanticRetriever(self.vector_store)

    def ingest_documents(self, documents: List[Document], provider: BaseAIProvider) -> Dict[str, Any]:
        """
        Ingests documents into RAG vector index.
        1. Chunks documents into text splits.
        2. Generates vector embeddings using active AI provider.
        3. Stores vectors in vector store (Qdrant Cloud or in-memory) with metadata.
        """
        if not documents:
            return {"status": "error", "message": "No documents provided for ingestion."}

        if not provider.is_configured():
            return {
                "status": "error",
                "message": f"Provider '{provider.name}' is not configured with valid API credentials."
            }

        try:
            # Step 1: Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            if not chunks:
                return {"status": "warning", "message": "No text content extracted from documents."}

            # Step 2: Generate embeddings via active provider
            texts = [c.content for c in chunks]
            logger.info(f"Generating embeddings for {len(texts)} chunk(s) via provider '{provider.name}'")
            embeddings = provider.embed_documents(texts)

            # Step 3: Add chunks and embeddings to vector store
            added_count = self.vector_store.add_chunks(chunks, embeddings, provider.provider_id)

            spec = Config.get_provider_spec(provider.provider_id)
            collection_name = spec.collection_name if spec else "In-Memory"

            return {
                "status": "success",
                "document_count": len(documents),
                "chunk_count": added_count,
                "provider": provider.name,
                "provider_id": provider.provider_id,
                "embedding_model": provider.get_embedding_model_name(),
                "vector_dimension": self.vector_store.get_embedding_dimension() or (spec.dimension if spec else None),
                "collection_name": collection_name
            }
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Ingestion error: {clean_err}")
            return {"status": "error", "message": clean_err}

    def clear_index(self, provider_id: Optional[str] = None):
        """Clears stored vector index."""
        if hasattr(self.vector_store, "clear_store"):
            self.vector_store.clear_store()

    def query(
        self,
        question: str,
        provider: BaseAIProvider,
        top_k: int = 5,
        similarity_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Executes Question Answering against vector index using active AI provider.
        """
        if not question.strip():
            return {"answer": "Please enter a non-empty question.", "sources": []}

        if not provider.is_configured():
            return {
                "answer": f"Selected provider '{provider.name}' is not configured with valid API credentials.",
                "sources": []
            }

        # Step 1: Retrieve top-K relevant chunks via vector similarity
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            provider=provider,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        if not retrieved_chunks:
            return {
                "answer": "Based on the provided documents, I do not have sufficient information to answer this question.",
                "sources": [],
                "provider": provider.name,
                "chat_model": provider.get_chat_model_name(),
                "retrieved_count": 0
            }

        # Step 2: Format prompt with context snippets (marked as untrusted passive reference data)
        formatted_prompt = format_rag_prompt(question, retrieved_chunks)

        # Step 3: Generate answer via active provider
        answer = provider.generate(
            prompt=formatted_prompt,
            system_prompt=RAG_SYSTEM_PROMPT
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "provider": provider.name,
            "chat_model": provider.get_chat_model_name(),
            "retrieved_count": len(retrieved_chunks)
        }

    def get_stats(self, provider_id: Optional[str] = None) -> Dict[str, Any]:
        """Returns current RAG pipeline statistics."""
        if hasattr(self.vector_store, "get_stats"):
            return self.vector_store.get_stats(provider_id=provider_id)
        return {
            "count": 0,
            "dimension": "N/A",
            "provider_id": provider_id or "None",
            "store_type": "Unknown",
            "status": "Empty"
        }
