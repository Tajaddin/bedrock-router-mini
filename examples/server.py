"""FastAPI server exposing a cost-aware Bedrock-style /invoke endpoint.

Run with::

    pip install -e ".[anthropic,server]"
    export ANTHROPIC_API_KEY=sk-ant-...
    uvicorn examples.server:app --port 8080

Then::

    curl -X POST http://localhost:8080/invoke -H 'content-type: application/json' \
        -d '{"prompt": "Hi"}'
    # Short prompt → Llama (cheap)

    curl -X POST http://localhost:8080/invoke -H 'content-type: application/json' \
        -d '{"prompt": "Please explain in detail the trade-offs ..."}'
    # Long or analysis prompt → Claude Haiku
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from bedrock_router_mini import (
    AnthropicAsBedrockClient,
    BedrockRouter,
    Route,
    by_char_length,
    by_keyword,
)


class InvokeRequest(BaseModel):
    prompt: str
    system: str | None = None
    max_tokens: int = 512


class InvokeResponse(BaseModel):
    text: str
    model_used: str
    cost_usd: float
    latency_ms: int
    fallback_triggered: bool
    matched_route_label: str | None


app = FastAPI(title="bedrock-router-mini")

# In production, you'd point this at boto3 Bedrock with Boto3BedrockClient.
# Here we use the AnthropicAsBedrock adapter so the example runs without AWS.
client = AnthropicAsBedrockClient()

router = BedrockRouter(
    client=client,
    routes=[
        # Long or analytical prompts → Haiku (the smart-but-affordable model).
        Route(by_keyword(["analyze", "explain", "compare", "trade-off"]),
              "anthropic.claude-haiku-4-5", "keyword-analytic"),
        Route(by_char_length(800), "anthropic.claude-haiku-4-5", "long-prompt"),
        # Everything else → cheap default. (Swap in a Bedrock Llama model once you have AWS creds.)
        Route(lambda _: True, "anthropic.claude-3-5-haiku-latest", "default"),
    ],
    fallback_chain=["anthropic.claude-3-5-haiku-latest"],
)


@app.post("/invoke", response_model=InvokeResponse)
def invoke(req: InvokeRequest) -> InvokeResponse:
    result = router.invoke(req.prompt, system=req.system, max_tokens=req.max_tokens)
    return InvokeResponse(
        text=result.text,
        model_used=result.model_used or "",
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        fallback_triggered=result.fallback_triggered,
        matched_route_label=result.matched_route_label,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_routes": len(router.routes)}
