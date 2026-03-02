"""Provider registry."""

from __future__ import annotations

from tinyagent.providers.base import Provider
from tinyagent.providers.ollama_provider import OllamaProvider
from tinyagent.providers.anthropic_provider import AnthropicProvider
from tinyagent.providers.openai_provider import OpenAIProvider
from tinyagent.providers.google_provider import GoogleProvider


def build_registry() -> dict[str, Provider]:
    """Instantiate all providers and return a name -> Provider map."""
    providers: list[Provider] = [
        OllamaProvider(),
        AnthropicProvider(),
        OpenAIProvider(),
        GoogleProvider(),
    ]
    return {p.name: p for p in providers}


__all__ = [
    "Provider",
    "OllamaProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "build_registry",
]
