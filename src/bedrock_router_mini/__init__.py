"""Cost-aware multi-model router for AWS Bedrock with complexity-based routing.

Wraps any Bedrock-compatible client behind a single ``BedrockRouter`` that:

* selects a model per call using a configurable routing rule
* falls back through a chain of models on error
* tracks token cost using public Bedrock pricing
"""

from bedrock_router_mini.clients import (
    AnthropicAsBedrockClient,
    BedrockClient,
    BedrockResponse,
    Boto3BedrockClient,
    GroqAsBedrockClient,
    MockBedrockClient,
)
from bedrock_router_mini.pricing import Pricing, cost_for, get_pricing, register_pricing
from bedrock_router_mini.router import (
    BedrockRouter,
    Route,
    RouterResponse,
    by_char_length,
    by_keyword,
    by_token_count,
)

__version__ = "0.1.0"

__all__ = [
    # clients
    "BedrockClient",
    "BedrockResponse",
    "Boto3BedrockClient",
    "AnthropicAsBedrockClient",
    "GroqAsBedrockClient",
    "MockBedrockClient",
    # router
    "BedrockRouter",
    "Route",
    "RouterResponse",
    "by_char_length",
    "by_token_count",
    "by_keyword",
    # pricing
    "Pricing",
    "cost_for",
    "get_pricing",
    "register_pricing",
]
