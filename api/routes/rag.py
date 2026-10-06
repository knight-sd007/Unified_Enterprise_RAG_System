"""
RAG Semantic Search, Telemetry, and Administrative Index Management routes.
Supports decoupled chat generation and embedding models.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from api.dependencies import (
    APIError,
    UserSession,
    get_rag_pipeline,
    require_admin,
    require_workspace_access,
)
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
)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Execute grounded semantic question answering",
    description=(
        "Retrieves relevant document chunks owned by the authenticated user and generates "
        "a grounded answer with source citations. Supports independent chat and embedding models. Requires authentication."
    ),
)
async def query_rag(
    req: QueryRequest,
    session: UserSession = Depends(require_workspace_access),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Executes semantic retrieval with owner scoping and grounded answer generation."""
    # 1. Resolve and validate target AI providers
    default_p = Config.get_ai_provider().strip().lower()
    is_decoupled = bool(req.chat_provider_id or req.embedding_provider_id)
    chat_p_id = (req.chat_provider_id or req.provider_id or default_p).strip().lower()
    embed_p_id = (req.embedding_provider_id or req.provider_id or default_p).strip().lower()

    try:
        chat_provider = get_provider_by_id(chat_p_id)
    except ValueError as e:
        err_code = "INVALID_CHAT_PROVIDER" if is_decoupled else "INVALID_PROVIDER"
        raise APIError(status_code=400, code=err_code, message=str(e))

    try:
        embed_provider = get_provider_by_id(embed_p_id)
    except ValueError as e:
        err_code = "INVALID_EMBEDDING_PROVIDER" if is_decoupled else "INVALID_PROVIDER"
        raise APIError(status_code=400, code=err_code, message=str(e))

    if not chat_provider.is_configured():
        err_code = "CHAT_PROVIDER_NOT_CONFIGURED" if is_decoupled else "PROVIDER_NOT_CONFIGURED"
        raise APIError(
            status_code=400,
            code=err_code,
            message=f"Chat provider '{chat_provider.name}' is not configured with valid API credentials.",
        )

    if not embed_provider.is_configured():
        err_code = "EMBEDDING_PROVIDER_NOT_CONFIGURED" if is_decoupled else "PROVIDER_NOT_CONFIGURED"
        raise APIError(
            status_code=400,
            code=err_code,
            message=f"Embedding provider '{embed_provider.name}' is not configured with valid API credentials.",
        )

    # 2. Check active vector index provider compatibility against embedding provider
    active_provider_id = pipeline.vector_store.get_active_provider_id()
    if active_provider_id is not None and active_provider_id != embed_provider.provider_id:
        raise APIError(
            status_code=400,
            code="PROVIDER_MISMATCH",
            message=(
                f"Cannot execute query: Vector index was built with provider '{active_provider_id}', "
                f"but embedding provider is '{embed_provider.name}'. Switch embedding provider or re-index."
            ),
        )

    # 3. Execute RAG query scoped to caller's owner identity
    owner_scope = None if session.is_admin else session.user_id
    try:
        res = pipeline.query(
            question=req.query,
            chat_provider=chat_provider,
            embedding_provider=embed_provider,
            chat_model=req.chat_model,
            embedding_model=req.embedding_model,
            owner_id=owner_scope,
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
        provider=chat_provider.name,
        provider_id=chat_provider.provider_id,
        chat_model=res.get("chat_model", req.chat_model or chat_provider.get_chat_model_name()),
        chat_provider=chat_provider.name,
        chat_provider_id=chat_provider.provider_id,
        embedding_provider=embed_provider.name,
        embedding_provider_id=embed_provider.provider_id,
        embedding_model=res.get("embedding_model", req.embedding_model or embed_provider.get_embedding_model_name()),
        retrieved_count=res.get("retrieved_count", len(sources)),
    )


@router.get(
    "/stats",
    response_model=RAGStatsResponse,
    summary="Get RAG vector store and provider telemetry",
    description=(
        "Returns vector index counts scoped to the authenticated user, embedding dimensions, active store backend, "
        "and provider operational status. Requires authentication."
    ),
)
async def get_rag_stats(
    provider_id: Optional[str] = Query(None, description="Optional provider identifier ('openai', 'gemini', 'nvidia_nim')."),
    session: UserSession = Depends(require_workspace_access),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGStatsResponse:
    """Returns provider-scoped vector store telemetry for the current user."""
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

    owner_scope = None if session.is_admin else session.user_id
    raw_stats = pipeline.get_stats(provider_id=provider.provider_id, owner_id=owner_scope)
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
        user_id=session.user_id,
    )


@router.delete(
    "/index",
    response_model=ClearIndexResponse,
    summary="Admin: Clear entire vector index for active provider",
    description=(
        "Administrative full-index purge: Clears stored vector embeddings across all users for the specified or active AI provider. "
        "Requires administrative privileges. Ordinary users should use DELETE /api/v1/documents."
    ),
)
async def clear_vector_index(
    provider_id: Optional[str] = Query(None, description="Optional provider identifier ('openai', 'gemini', 'nvidia_nim')."),
    session: UserSession = Depends(require_admin),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> ClearIndexResponse:
    """Administrative full index clearing."""
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
        message=f"Administrator cleared vector index for provider '{provider.name}'.",
        provider_id=provider.provider_id,
    )
