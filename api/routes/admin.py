"""
Privileged Administrator Console Endpoints.
Guarded by require_admin authorization.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import require_admin, get_rag_pipeline, UserSession
from api.schemas import (
    AdminOverviewResponse,
    AdminDiagnosticsResponse,
    AdminVectorClearRequest,
    AdminVectorClearResponse,
    ProviderHealthItem,
)
from config.settings import Config
from providers.factory import get_supported_provider_ids, get_provider_by_id
from rag.pipeline import RAGPipeline
from rag.storage.metadata_db import get_metadata_repo
from utils.logging import logger

router = APIRouter(prefix="/api/v1/admin", tags=["Admin Console"])


@router.get(
    "/overview",
    response_model=AdminOverviewResponse,
    summary="Admin System Overview",
    description="Returns aggregated document, chunk, user, and storage statistics.",
)
async def get_admin_overview(
    admin: UserSession = Depends(require_admin),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> AdminOverviewResponse:
    """Aggregated global telemetry for administrative overview."""
    logger.info(f"Admin overview requested by '{admin.user_id}'")
    repo = get_metadata_repo()
    stats = repo.get_global_stats()

    store_type = "Qdrant Cloud" if Config.is_qdrant_configured() else "InMemoryVectorStore"

    return AdminOverviewResponse(
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        total_bytes=stats.get("total_bytes", 0),
        total_users=stats.get("total_users", 0),
        vector_store_backend=store_type,
        providers=stats.get("providers", []),
    )


@router.get(
    "/diagnostics",
    response_model=AdminDiagnosticsResponse,
    summary="Admin Diagnostics",
    description="Non-sensitive system diagnostics and live provider health verification.",
)
async def get_admin_diagnostics(
    admin: UserSession = Depends(require_admin),
) -> AdminDiagnosticsResponse:
    """Performs live connectivity checks and returns environment diagnostics."""
    logger.info(f"Admin diagnostics requested by '{admin.user_id}'")

    provider_health_items: List[ProviderHealthItem] = []
    for pid in get_supported_provider_ids():
        provider = get_provider_by_id(pid)
        check = provider.check_connectivity()
        provider_health_items.append(
            ProviderHealthItem(
                provider_id=pid,
                name=provider.name,
                configured=provider.is_configured(),
                status=check.get("status", "unknown"),
                message=check.get("message", ""),
                latency_ms=check.get("latency_ms"),
            )
        )

    return AdminDiagnosticsResponse(
        app_status="healthy",
        environment=Config.get_environment(),
        qdrant_configured=Config.is_qdrant_configured(),
        qdrant_host=Config.get_qdrant_host() if Config.is_qdrant_configured() else None,
        oauth_configured=Config.is_google_oauth_configured(),
        oauth_admin_emails=Config.get_google_admin_emails(),
        oauth_admin_subs=Config.get_google_admin_subs(),
        metadata_db_path=Config.get_metadata_db_path(),
        providers=provider_health_items,
    )


@router.post(
    "/vectors/clear",
    response_model=AdminVectorClearResponse,
    summary="Admin Global Vector Purge",
    description="Privileged operation to purge vectors across all users with explicit confirmation phrase.",
)
async def clear_global_vectors(
    payload: AdminVectorClearRequest,
    admin: UserSession = Depends(require_admin),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> AdminVectorClearResponse:
    """Purges vectors and metadata globally."""
    if payload.confirmation != "CONFIRM_ADMIN_GLOBAL_PURGE":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmation string must exactly match 'CONFIRM_ADMIN_GLOBAL_PURGE'.",
        )

    logger.warning(
        f"GLOBAL VECTOR PURGE initiated by admin '{admin.user_id}' (provider_id: {payload.provider_id})"
    )

    # 1. Clear vector store across all configured collections FIRST
    try:
        pipeline.clear_index(provider_id=payload.provider_id)
    except Exception as e:
        from utils.security import sanitize_error_message
        clean_err = sanitize_error_message(e)
        logger.error(f"Global vector purge failed during vector index clearance: {clean_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Global vector purge failed: {clean_err}",
        )

    # 2. Only after all vector collections succeed, delete SQLite metadata
    repo = get_metadata_repo()
    deleted_docs = repo.delete_all_global()
    purged_doc_count = len(deleted_docs)
    purged_chunk_count = sum(d.get("chunk_count", 0) for d in deleted_docs)

    return AdminVectorClearResponse(
        status="success",
        message=f"Global purge complete. Purged {purged_doc_count} document(s) and {purged_chunk_count} chunk(s).",
        purged_documents=purged_doc_count,
        purged_chunks=purged_chunk_count,
        provider_id=payload.provider_id,
    )
