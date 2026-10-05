"""
Health check routes for container liveness, orchestration probes, and provider diagnostics.
"""

from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter
from api.schemas import HealthResponse, ProvidersHealthResponse, ProviderHealthItem
from providers.factory import get_supported_provider_ids, get_provider_by_id

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check probe",
    description="Returns service health and operational status. Does not require authentication.",
)
async def health_check() -> HealthResponse:
    """Returns basic service health status."""
    return HealthResponse(
        status="healthy",
        service="p06-enterprise-rag",
        version="1.0.0",
    )


@router.get(
    "/health/providers",
    response_model=ProvidersHealthResponse,
    summary="Provider Health Diagnostics",
    description="Returns live health and connectivity status for all supported AI providers.",
)
async def providers_health_check() -> ProvidersHealthResponse:
    """Evaluates and returns configuration and live connectivity status for all providers."""
    items: List[ProviderHealthItem] = []
    for pid in get_supported_provider_ids():
        provider = get_provider_by_id(pid)
        check = provider.check_connectivity()
        items.append(
            ProviderHealthItem(
                provider_id=pid,
                name=provider.name,
                configured=provider.is_configured(),
                status=check.get("status", "unknown"),
                message=check.get("message", ""),
                latency_ms=check.get("latency_ms"),
            )
        )

    return ProvidersHealthResponse(
        status="healthy",
        providers=items,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
