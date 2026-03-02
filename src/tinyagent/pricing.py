"""Per-model pricing lookup table.

Prices are per 1M tokens: (input_price, output_price).
"""

from __future__ import annotations

# (dollars per 1M input tokens, dollars per 1M output tokens)
PRICING: dict[tuple[str, str], tuple[float, float]] = {
    # Ollama — local, free
    ("ollama", "llama3"): (0.0, 0.0),
    ("ollama", "llama3.1"): (0.0, 0.0),
    ("ollama", "llama3.2"): (0.0, 0.0),
    ("ollama", "mistral"): (0.0, 0.0),
    ("ollama", "qwen2.5"): (0.0, 0.0),
    # Anthropic
    ("anthropic", "claude-sonnet-4-6"): (3.0, 15.0),
    ("anthropic", "claude-opus-4-6"): (15.0, 75.0),
    ("anthropic", "claude-haiku-4-5"): (0.80, 4.0),
    # OpenAI
    ("openai", "gpt-4o"): (2.50, 10.0),
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "o3"): (10.0, 40.0),
    # Google
    ("google", "gemini-2.5-flash"): (0.15, 0.60),
    ("google", "gemini-2.5-pro"): (1.25, 10.0),
}

# Fallback pricing for unknown models (conservative estimate)
DEFAULT_PRICING: tuple[float, float] = (5.0, 15.0)


def get_pricing(provider: str, model: str) -> tuple[float, float]:
    """Return (input_price_per_1M, output_price_per_1M) for a model."""
    return PRICING.get((provider, model), DEFAULT_PRICING)


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int = 2000,
    completion_tokens: int = 2000,
) -> float:
    """Estimate cost in dollars for a given token count."""
    input_price, output_price = get_pricing(provider, model)
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000


def compute_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Compute exact cost from actual token usage."""
    input_price, output_price = get_pricing(provider, model)
    return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
