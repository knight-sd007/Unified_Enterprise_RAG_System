"""
Unified AI Provider Factory.

Provides centralized, framework-agnostic resolution and instantiation
of supported AI providers (OpenAI, Google Gemini, NVIDIA NIM) without UI coupling.
"""

from typing import Dict, List, Type
from providers.base import BaseAIProvider
from providers.gemini_provider import GeminiProvider
from providers.nvidia_nim_provider import NvidiaNimProvider
from providers.openai_provider import OpenAIProvider

# Registry mapping canonical provider ID to provider implementation class
PROVIDER_REGISTRY: Dict[str, Type[BaseAIProvider]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "nvidia_nim": NvidiaNimProvider,
}


def get_supported_provider_ids() -> List[str]:
    """
    Returns the deterministic list of all supported provider identifier strings.
    """
    return list(PROVIDER_REGISTRY.keys())


def get_provider_by_id(provider_id: str) -> BaseAIProvider:
    """
    Instantiates and returns the AI provider corresponding to the given provider ID.

    Args:
        provider_id: Canonical identifier ('openai', 'gemini', 'nvidia_nim').

    Returns:
        BaseAIProvider: Instantiated provider implementing embedding & chat capabilities.

    Raises:
        ValueError: If provider_id is missing, invalid, or unsupported.
    """
    if not isinstance(provider_id, str):
        raise ValueError(
            f"Invalid provider_id type '{type(provider_id).__name__}'. Expected string."
        )

    normalized_id = provider_id.strip().lower()
    provider_cls = PROVIDER_REGISTRY.get(normalized_id)

    if provider_cls is None:
        supported = ", ".join(repr(p) for p in get_supported_provider_ids())
        raise ValueError(
            f"Unsupported or unknown AI provider '{provider_id}'. Supported providers: [{supported}]."
        )

    return provider_cls()
