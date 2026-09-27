"""
RAG Semantic Search, Telemetry, and Index Management routes.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from api.dependencies import APIError, get_rag_pipeline, require_authentication
from api.schemas import (
    ClearIndexResponse,
    QueryRequest,
    QueryResponse,
    RAGStatsResponse,
    SourceItem,
)
from config.settings import Config
from providers.factory import get_provider_by_id
from rag.pipeline import RAGPipeline
from utils.security import sanitize_error_message

router = APIRouter(
    prefix="/rag",
    tags=["RAG Operations"],
    dependencies=[Depends(require_authentication)],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute grounded semantic question answering",
    description=(
        "Retrieves relevant document chunks from the vector store and generates "
        "a grounded answer with source citations. Requires authentication."
    ),
)
async def query_rag(
    req: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Executes semantic retrieval and grounded answer generation."""
    # 1. Resolve and validate target AI provider
    target_provider_id = (req.provider_id or Config.get_ai_provider()).strip().lower()
    try:
        provider = get_provider_by_id(target_provider_id)
    except ValueError as e:
        raise APIError(
            status_code=400,
            code="INVALID_PROVIDER",
            message=str(e),
        )

    if not provider.is_configured():
        raise APIError(
            status_code=400,
            code="PROVIDER_NOT_CONFIGURED",
            message=f"Provider '{provider.name}' is not configured with valid API credentials.",
        )

    # 2. Check active vector index provider compatibility
    active_provider_id = pipeline.vector_store.get_active_provider_id()
    if active_provider_id is not None and active_provider_id != provider.provider_id:
        raise APIError(
            status_code=400,
            code="PROVIDER_MISMATCH",
            message=(
                f"Cannot execute query: Provider mismatch between active index ('{active_provider_id}') "
                f"and query provider ('{provider.name}'). Switch provider or clear index."
            ),
        )

    # 3. Execute RAG query
    try:
        res = pipeline.query(
            question=req.query,
            provider=provider,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
        )
    except Exception as e:
        clean_err = sanitize_error_message(e)
        raise APIError(
            status_code=500,
            code="QUERY_EXECUTION_ERROR",
            message=f"Query execution failed: {clean_err}",
        )

    sources = [
        SourceItem(
            chunk_id=src.get("chunk_id", f"c{i}"),
            content=src.get("content", ""),
            score=float(src.get("score", 0.0)),
            metadata=src.get("metadata", {}),
        )
        for i, src in enumerate(res.get("sources", []), 1)
    ]

    return QueryResponse(
        answer=res.get("answer", ""),
        sources=sources,
        provider=res.get("provider", provider.name),
        provider_id=provider.provider_id,
        chat_model=res.get("chat_model", provider.get_chat_model_name()),
        retrieved_count=res.get("retrieved_count", len(sources)),
    )


@router.get(
    "/stats",
    response_model=RAGStatsResponse,
    summary="Get RAG vector store and provider telemetry",
    description=(
        "Returns vector index counts, embedding dimensions, active store backend, "
        "and provider operational status. Requires authentication."
    ),
)
async def get_rag_stats(
    provider_id: Optional[str] = Query(None, description="Optional provider identifier ('openai', 'gemini', 'nvidia_nim')."),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGStatsResponse:
    """Returns provider-scoped vector store telemetry."""
    # Resolve target provider
    target_provider_id = (
        provider_id or pipeline.vector_store.get_active_provider_id() or Config.get_ai_provider()
    ).strip().lower()

    try:
        provider = get_provider_by_id(target_provider_id)
    except ValueError as e:
        raise APIError(
            status_code=400,
            code="INVALID_PROVIDER",
            message=str(e),
        )

    raw_stats = pipeline.get_stats(provider_id=provider.provider_id)
    spec = Config.get_provider_spec(provider.provider_id)
    col_name = spec.collection_name if spec else None

    raw_dim = raw_stats.get("dimension")
    effective_dim = (
        raw_dim
        if (raw_dim is not None and raw_dim != "N/A")
        else (spec.dimension if spec else "Dynamic")
    )

    return RAGStatsResponse(
        provider_id=provider.provider_id,
        provider_name=provider.name,
        count=raw_stats.get("count", 0),
        dimension=effective_dim,
        store_type=raw_stats.get("store_type", "In-Memory Vector Store"),
        collection_name=col_name,
        status=raw_stats.get("status", "Ready"),
        configured=provider.is_configured(),
    )


@router.delete(
    "/index",
    response_model=ClearIndexResponse,
    summary="Clear vector index for active provider",
    description=(
        "Clears stored vector embeddings for the specified or active AI provider. "
        "Preserves collection routing and isolation guarantees. Requires authentication."
    ),
)
async def clear_vector_index(
    provider_id: Optional[str] = Query(None, description="Optional provider identifier ('openai', 'gemini', 'nvidia_nim')."),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ClearIndexResponse:
    """Clears provider-scoped vector store points."""
    target_provider_id = (
        provider_id or pipeline.vector_store.get_active_provider_id() or Config.get_ai_provider()
    ).strip().lower()

    try:
        provider = get_provider_by_id(target_provider_id)
    except ValueError as e:
        raise APIError(
            status_code=400,
            code="INVALID_PROVIDER",
            message=str(e),
        )

    pipeline.clear_index(provider_id=provider.provider_id)

    return ClearIndexResponse(
        status="success",
        message=f"Vector index cleared successfully for provider '{provider.name}'.",
        provider_id=provider.provider_id,
    )
