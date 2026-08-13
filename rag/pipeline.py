"""
RAG Pipeline Orchestrator.

Integrates Document Ingestion, Text Chunking, In-Memory Vector Indexing, Semantic Retrieval, and Provider Chat Generation.
"""

from typing import List, Dict, Any, Optional
from rag.loaders import DocumentLoader, Document
from rag.chunker import DocumentChunker, Chunk
from rag.vector_store import InMemoryVectorStore
from rag.retriever import SemanticRetriever
from rag.prompt import RAG_SYSTEM_PROMPT, format_rag_prompt
from providers.base import BaseAIProvider
from utils.logging import logger


class RAGPipeline:
    """Orchestrates end-to-end RAG workflow."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.vector_store = InMemoryVectorStore()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = SemanticRetriever(self.vector_store)

    def ingest_documents(self, documents: List[Document], provider: BaseAIProvider) -> Dict[str, Any]:
        """
        Ingests documents into RAG vector index.
        1. Chunks documents into text splits.
        2. Generates vector embeddings using active AI provider.
        3. Stores vectors in memory with metadata.
        """
        if not documents:
            return {"status": "error", "message": "No documents provided for ingestion."}

        if not provider.is_configured():
            return {
                "status": "error",
                "message": f"Provider '{provider.name}' is not configured with valid API credentials."
            }

        # Step 1: Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            return {"status": "warning", "message": "No text content extracted from documents."}

        # Step 2: Generate embeddings via active provider
        texts = [c.content for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunk(s) via provider '{provider.name}'")
        embeddings = provider.embed_documents(texts)

        # Step 3: Add chunks and embeddings to in-memory vector store
        added_count = self.vector_store.add_chunks(chunks, embeddings, provider.provider_id)

        return {
            "status": "success",
            "document_count": len(documents),
            "chunk_count": added_count,
            "provider": provider.name,
            "embedding_model": provider.get_embedding_model_name(),
            "vector_dimension": self.vector_store.get_embedding_dimension()
        }

    def clear_index(self):
        """Clears stored vector index."""
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

        if self.vector_store.count() == 0:
            return {
                "answer": "Vector store is empty. Please upload and index documents before asking questions.",
                "sources": []
            }

        # Step 1: Retrieve top-K relevant chunks via vector cosine similarity
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            provider=provider,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Step 2: Format prompt with context snippets
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

    def get_stats(self) -> Dict[str, Any]:
        """Returns current RAG pipeline statistics."""
        return self.vector_store.get_stats()
