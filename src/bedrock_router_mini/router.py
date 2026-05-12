"""BedrockRouter: pick a model per call, fall back on error, track cost.

The router is a thin orchestrator over any :class:`BedrockClient`. It scans a
list of :class:`Route` rules in order and routes to the first matching model;
if that model errors, it walks a fallback chain. Every step is recorded in the
returned :class:`RouterResponse` so a reviewer can audit the path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from bedrock_router_mini.clients import BedrockClient, BedrockResponse
from bedrock_router_mini.pricing import cost_for


@dataclass
class Route:
    """One routing rule. ``condition`` returns True when this route should fire."""

    condition: Callable[[str], bool]
    model_id: str
    label: str = ""


@dataclass
class AttemptRecord:
    model_id: str
    success: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None


@dataclass
class RouterResponse:
    """Full audit trail of one invocation."""

    text: str
    model_used: str | None
    success: bool
    cost_usd: float
    latency_ms: int
    fallback_triggered: bool
    attempts: list[AttemptRecord] = field(default_factory=list)
    matched_route_label: str | None = None
    raw_response: BedrockResponse | None = None


# ---------------------------------------------------------------------------
# Built-in routing conditions
# ---------------------------------------------------------------------------


def by_char_length(threshold: int) -> Callable[[str], bool]:
    """Returns a condition that fires for prompts strictly LONGER than ``threshold`` chars.

    Typical use::

        Route(by_char_length(500), "anthropic.claude-haiku-4-5", "long → claude")
        Route(lambda _: True,      "meta.llama3-1-8b-instruct-v1:0", "default → llama")
    """
    return lambda prompt: len(prompt) > threshold


def by_token_count(threshold: int, *, chars_per_token: float = 4.0) -> Callable[[str], bool]:
    """Returns a condition that fires for prompts whose estimated token count exceeds ``threshold``.

    Token count is approximated from character length (default ratio is 4
    chars/token, which is close to the ratio for English text under BPE).
    """
    def cond(prompt: str) -> bool:
        return (len(prompt) / chars_per_token) > threshold
    return cond


def by_keyword(keywords: list[str], *, case_sensitive: bool = False) -> Callable[[str], bool]:
    """Fires when any keyword is present in the prompt."""
    if not case_sensitive:
        kws = [k.lower() for k in keywords]

        def cond(prompt: str) -> bool:
            return any(k in prompt.lower() for k in kws)
        return cond
    return lambda prompt: any(k in prompt for k in keywords)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class BedrockRouter:
    """Pick a model from ``routes`` (first matching wins), invoke it via the
    underlying :class:`BedrockClient`, and fall back through ``fallback_chain``
    on error.

    Parameters
    ----------
    client:
        Any :class:`BedrockClient` adapter.
    routes:
        Routing rules evaluated in order. The first matching rule wins.
    default_model:
        Model used if no route matches. Required if any route's condition can
        return False.
    fallback_chain:
        Models to try in order if the chosen model errors. Cost is tracked
        across attempts so the response shows the real spend on the call.
    """

    def __init__(
        self,
        client: BedrockClient,
        routes: list[Route],
        *,
        default_model: str | None = None,
        fallback_chain: list[str] | None = None,
    ) -> None:
        self.client = client
        self.routes = list(routes)
        self.default_model = default_model
        self.fallback_chain = list(fallback_chain or [])

    def select_model(self, prompt: str) -> tuple[str, str | None]:
        """Return ``(model_id, matched_route_label)``."""
        for route in self.routes:
            if route.condition(prompt):
                return route.model_id, route.label
        if self.default_model is None:
            raise RuntimeError(
                "No route matched and no default_model is set. Add a catch-all route or default_model."
            )
        return self.default_model, None

    def invoke(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
    ) -> RouterResponse:
        primary, label = self.select_model(prompt)
        plan = [primary, *[m for m in self.fallback_chain if m != primary]]

        attempts: list[AttemptRecord] = []
        total_cost = 0.0
        total_latency = 0
        for idx, model_id in enumerate(plan):
            start = time.perf_counter()
            try:
                resp = self.client.invoke(
                    model_id, prompt, system=system, max_tokens=max_tokens
                )
                elapsed = int((time.perf_counter() - start) * 1000)
                cost = cost_for(
                    model_id,
                    input_tokens=resp.input_tokens,
                    output_tokens=resp.output_tokens,
                )
                attempts.append(
                    AttemptRecord(
                        model_id=model_id,
                        success=True,
                        latency_ms=elapsed,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        cost_usd=cost,
                    )
                )
                total_cost += cost
                total_latency += elapsed
                return RouterResponse(
                    text=resp.text,
                    model_used=model_id,
                    success=True,
                    cost_usd=total_cost,
                    latency_ms=total_latency,
                    fallback_triggered=idx > 0,
                    attempts=attempts,
                    matched_route_label=label,
                    raw_response=resp,
                )
            except Exception as exc:  # noqa: BLE001
                elapsed = int((time.perf_counter() - start) * 1000)
                attempts.append(
                    AttemptRecord(
                        model_id=model_id,
                        success=False,
                        latency_ms=elapsed,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        error=f"{type(exc).__name__}: {str(exc)[:200]}",
                    )
                )
                total_latency += elapsed
                continue

        return RouterResponse(
            text="",
            model_used=None,
            success=False,
            cost_usd=total_cost,
            latency_ms=total_latency,
            fallback_triggered=len(attempts) > 1,
            attempts=attempts,
            matched_route_label=label,
            raw_response=None,
        )
