"""Per-model pricing on AWS Bedrock (USD per million tokens, as of 2026-01)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float


_REGISTRY: dict[str, Pricing] = {
    # Anthropic models on Bedrock (mirror Anthropic direct pricing).
    "anthropic.claude-haiku-4-5": Pricing(0.80, 4.00),
    "anthropic.claude-sonnet-4-6": Pricing(3.00, 15.00),
    "anthropic.claude-opus-4-7": Pricing(15.00, 75.00),
    "anthropic.claude-3-5-haiku-20241022-v1:0": Pricing(0.80, 4.00),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": Pricing(3.00, 15.00),
    # Meta Llama models on Bedrock.
    "meta.llama3-1-8b-instruct-v1:0": Pricing(0.22, 0.22),
    "meta.llama3-1-70b-instruct-v1:0": Pricing(0.72, 0.72),
    "meta.llama3-3-70b-instruct-v1:0": Pricing(0.72, 0.72),
    # Amazon Titan and Mistral kept for testing examples.
    "amazon.titan-text-express-v1": Pricing(0.20, 0.60),
    "mistral.mistral-7b-instruct-v0:2": Pricing(0.15, 0.20),
    "mistral.mixtral-8x7b-instruct-v0:1": Pricing(0.45, 0.70),
    # Internal aliases used by the AnthropicAsBedrock / GroqAsBedrock adapters
    # so the router sees a single canonical ID per model regardless of provider.
    "_anthropic_haiku_45": Pricing(0.80, 4.00),
    "_anthropic_sonnet_46": Pricing(3.00, 15.00),
    "_groq_llama_318b": Pricing(0.22, 0.22),  # mirror Bedrock pricing
    "_groq_llama_3370b": Pricing(0.72, 0.72),
}


def register_pricing(model_id: str, pricing: Pricing) -> None:
    """Add or override a pricing entry. Useful for negotiated rates or new models."""
    _REGISTRY[model_id] = pricing


def get_pricing(model_id: str) -> Pricing | None:
    return _REGISTRY.get(model_id)


def cost_for(model_id: str, *, input_tokens: int = 0, output_tokens: int = 0) -> float:
    """Return the USD cost for one call. 0 if no pricing entry."""
    p = _REGISTRY.get(model_id)
    if p is None:
        return 0.0
    return input_tokens / 1_000_000 * p.input_per_mtok + output_tokens / 1_000_000 * p.output_per_mtok
