"""
RAG Pipeline Orchestrator with Multi-User Document Lifecycle Management.

Integrates Document Ingestion, Text Chunking, Vector Indexing, Semantic Retrieval,
Document Lifecycle Management, and Provider Chat Generation.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
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
    """Orchestrates end-to-end RAG workflow with vector store and multi-user document isolation."""

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

    def ingest_documents(
        self,
        documents: List[Document],
        provider: BaseAIProvider,
        owner_id: str = "default_user",
    ) -> Dict[str, Any]:
        """
        Ingests documents into RAG vector index with owner binding.
        1. Chunks documents into text splits preserving doc_id and owner_id.
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
            logger.info(f"Generating embeddings for {len(texts)} chunk(s) via provider '{provider.name}' [owner='{owner_id}']")
            embeddings = provider.embed_documents(texts)

            # Step 3: Add chunks and embeddings to vector store
            added_count = self.vector_store.add_chunks(chunks, embeddings, provider.provider_id, owner_id=owner_id)

            spec = Config.get_provider_spec(provider.provider_id)
            collection_name = spec.collection_name if spec else "In-Memory"
            doc_ids = [d.metadata.get("doc_id") for d in documents if d.metadata.get("doc_id")]

            return {
                "status": "success",
                "document_count": len(documents),
                "chunk_count": added_count,
                "provider": provider.name,
                "provider_id": provider.provider_id,
                "embedding_model": provider.get_embedding_model_name(),
                "vector_dimension": self.vector_store.get_embedding_dimension() or (spec.dimension if spec else None),
                "collection_name": collection_name,
                "doc_ids": doc_ids,
                "owner_id": owner_id,
            }
        except Exception as e:
            clean_err = sanitize_error_message(e)
            logger.error(f"Ingestion error: {clean_err}")
            return {"status": "error", "message": clean_err}

    def list_documents(
        self,
        owner_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists active documents owned by the user."""
        if hasattr(self.vector_store, "list_documents"):
            return self.vector_store.list_documents(owner_id=owner_id, provider_id=provider_id)
        return []

    def delete_document(
        self,
        doc_id: str,
        owner_id: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> int:
        """Deletes a specific document and its vector chunks."""
        if hasattr(self.vector_store, "delete_document"):
            return self.vector_store.delete_document(doc_id=doc_id, owner_id=owner_id, provider_id=provider_id)
        return 0

    def clear_user_documents(
        self,
        owner_id: str,
        provider_id: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Clears all documents and chunks owned by a user."""
        if hasattr(self.vector_store, "clear_user_documents"):
            return self.vector_store.clear_user_documents(owner_id=owner_id, provider_id=provider_id)
        return 0, 0

    def clear_index(self, provider_id: Optional[str] = None):
        """Administrative wipe: Clears all vectors in the active/specified provider index."""
        if hasattr(self.vector_store, "clear_store"):
            self.vector_store.clear_store(provider_id=provider_id)

    def query(
        self,
        question: str,
        provider: Optional[BaseAIProvider] = None,
        owner_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.25,
        chat_provider: Optional[BaseAIProvider] = None,
        embedding_provider: Optional[BaseAIProvider] = None,
        chat_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes Question Answering against vector index with owner isolation.
        Supports decoupled chat generation and embedding providers/models.
        """
        if not question.strip():
            return {"answer": "Please enter a non-empty question.", "sources": []}

        # Resolve effective providers
        eff_chat_provider = chat_provider or provider
        eff_embed_provider = embedding_provider or provider

        if not eff_embed_provider:
            raise ValueError("An embedding provider must be specified for RAG query.")
        if not eff_chat_provider:
            raise ValueError("A chat provider must be specified for RAG answer generation.")

        if not eff_embed_provider.is_configured():
            return {
                "answer": f"Selected embedding provider '{eff_embed_provider.name}' is not configured with valid API credentials.",
                "sources": []
            }

        if not eff_chat_provider.is_configured():
            return {
                "answer": f"Selected chat provider '{eff_chat_provider.name}' is not configured with valid API credentials.",
                "sources": []
            }

        # Step 1: Retrieve top-K relevant chunks via vector similarity scoped to owner_id
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            provider=eff_embed_provider,
            owner_id=owner_id,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            model=embedding_model,
        )

        active_chat_model = chat_model or eff_chat_provider.get_chat_model_name()
        active_embed_model = embedding_model or eff_embed_provider.get_embedding_model_name()

        if not retrieved_chunks:
            return {
                "answer": "Based on the provided documents, I do not have sufficient information to answer this question.",
                "sources": [],
                "provider": eff_chat_provider.name,
                "chat_provider": eff_chat_provider.name,
                "chat_provider_id": eff_chat_provider.provider_id,
                "chat_model": active_chat_model,
                "embedding_provider": eff_embed_provider.name,
                "embedding_provider_id": eff_embed_provider.provider_id,
                "embedding_model": active_embed_model,
                "retrieved_count": 0,
            }

        # Step 2: Format prompt with context snippets
        formatted_prompt = format_rag_prompt(question, retrieved_chunks)

        # Step 3: Generate answer via active chat provider
        try:
            answer = eff_chat_provider.generate(
                prompt=formatted_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                model=chat_model,
            )
        except TypeError:
            answer = eff_chat_provider.generate(
                prompt=formatted_prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
            )

        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "provider": eff_chat_provider.name,
            "chat_provider": eff_chat_provider.name,
            "chat_provider_id": eff_chat_provider.provider_id,
            "chat_model": active_chat_model,
            "embedding_provider": eff_embed_provider.name,
            "embedding_provider_id": eff_embed_provider.provider_id,
            "embedding_model": active_embed_model,
            "retrieved_count": len(retrieved_chunks),
        }

    def get_stats(
        self,
        provider_id: Optional[str] = None,
        owner_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns current RAG pipeline statistics scoped to user."""
        if hasattr(self.vector_store, "get_stats"):
            return self.vector_store.get_stats(provider_id=provider_id, owner_id=owner_id)
        return {
            "count": 0,
            "dimension": "N/A",
            "provider_id": provider_id or "None",
            "store_type": "Unknown",
            "status": "Empty",
            "user_id": owner_id or "all",
        }
