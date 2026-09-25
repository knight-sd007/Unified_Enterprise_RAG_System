"""
AI Provider metadata routes.
"""

from typing import List
from fastapi import APIRouter, Depends
from api.dependencies import require_authentication
from api.schemas import ProviderMetadata, ProvidersListResponse
from config.settings import Config
from providers.factory import get_provider_by_id, get_supported_provider_ids

router = APIRouter(
    prefix="/providers",
    tags=["Providers"],
    dependencies=[Depends(require_authentication)],
)


@router.get(
    "",
    response_model=ProvidersListResponse,
    summary="List supported AI providers and configuration status",
    description="Returns sanitized provider metadata including models and vector dimensions. Requires authentication.",
)
async def list_providers() -> ProvidersListResponse:
    """Returns safe metadata for all supported AI providers."""
    providers_list: List[ProviderMetadata] = []

    for provider_id in get_supported_provider_ids():
        provider = get_provider_by_id(provider_id)
        spec = Config.get_provider_spec(provider_id)
        dimension = spec.dimension if spec else 768

        providers_list.append(
            ProviderMetadata(
                provider_id=provider.provider_id,
                name=provider.name,
                configured=provider.is_configured(),
                chat_model=provider.get_chat_model_name(),
                embedding_model=provider.get_embedding_model_name(),
                dimension=dimension,
            )
        )

    return ProvidersListResponse(providers=providers_list)
