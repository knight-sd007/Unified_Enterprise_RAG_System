"""
AI Provider metadata routes.
"""

from typing import List
from fastapi import APIRouter, Depends
from api.dependencies import require_authentication
from api.schemas import ProviderMetadata, ProvidersListResponse
from config.settings import Config, SUPPORTED_CHAT_MODELS, SUPPORTED_EMBEDDING_MODELS
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
    supported_ids = get_supported_provider_ids()

    for provider_id in supported_ids:
        provider = get_provider_by_id(provider_id)
        spec = Config.get_provider_spec(provider_id)
        dimension = spec.dimension if spec else 768

        chat_models = SUPPORTED_CHAT_MODELS.get(provider_id, [provider.get_chat_model_name()])
        embed_models = SUPPORTED_EMBEDDING_MODELS.get(provider_id, [{"model": provider.get_embedding_model_name(), "dimension": dimension}])

        providers_list.append(
            ProviderMetadata(
                provider_id=provider.provider_id,
                name=provider.name,
                configured=provider.is_configured(),
                chat_model=provider.get_chat_model_name(),
                embedding_model=provider.get_embedding_model_name(),
                dimension=dimension,
                supported_chat_models=chat_models,
                supported_embedding_models=embed_models,
            )
        )

    configured_provider = Config.get_ai_provider().strip().lower()
    default_provider = (
        configured_provider
        if configured_provider in supported_ids
        else "openai"
    )

    return ProvidersListResponse(
        providers=providers_list,
        default_provider=default_provider,
    )
