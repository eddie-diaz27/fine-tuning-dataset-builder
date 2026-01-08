"""
Provider factory for creating LLM provider instances.
"""

from providers.base_provider import BaseProvider, ProviderSpec, ValidationResult, ValidationError
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.google_provider import GoogleProvider
from providers.mistral_provider import MistralProvider
from providers.llama_provider import LlamaProvider


# Provider mapping
PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "mistral": MistralProvider,
    "llama": LlamaProvider,
}


def get_provider(name: str) -> BaseProvider:
    """
    Factory function to get provider instance.

    Args:
        name: Provider name (case-insensitive)

    Returns:
        Provider instance

    Raises:
        ValueError: If provider not found

    Example:
        provider = get_provider("openai")
        spec = provider.get_spec()
    """
    name_lower = name.lower()

    if name_lower not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider: {name}\n"
            f"Available providers: {', '.join(PROVIDER_MAP.keys())}"
        )

    provider_class = PROVIDER_MAP[name_lower]
    return provider_class()


def list_providers() -> list:
    """
    Get list of available provider names.

    Returns:
        List of provider names
    """
    return list(PROVIDER_MAP.keys())


__all__ = [
    'BaseProvider',
    'ProviderSpec',
    'ValidationResult',
    'ValidationError',
    'OpenAIProvider',
    'AnthropicProvider',
    'GoogleProvider',
    'MistralProvider',
    'LlamaProvider',
    'get_provider',
    'list_providers'
]
