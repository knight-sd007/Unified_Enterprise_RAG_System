"""
Health check route for container liveness and orchestration probes.
"""

from fastapi import APIRouter
from api.schemas import HealthResponse

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
