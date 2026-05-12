"""Bedrock-compatible client interface and four adapters.

The router does not care whether the underlying service is AWS Bedrock,
Anthropic's direct API, Groq, or a deterministic mock — it talks to any
client implementing :class:`BedrockClient`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class BedrockResponse:
    """Normalized response shape across all adapters."""

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    raw: Any = None
    extra: dict = field(default_factory=dict)


@runtime_checkable
class BedrockClient(Protocol):
    """Minimal client surface: invoke a model with a prompt, get a typed response."""

    def invoke(self, model_id: str, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> BedrockResponse: ...


# ---------------------------------------------------------------------------
# MockBedrockClient — deterministic, used for tests and the offline demo
# ---------------------------------------------------------------------------


class MockBedrockClient:
    """In-memory client. Returns canned responses per model, optionally raises."""

    def __init__(
        self,
        canned: dict[str, str] | None = None,
        raise_for: set[str] | None = None,
        tokens_per_char: float = 0.25,
    ) -> None:
        self.canned = canned or {}
        self.raise_for = raise_for or set()
        self.tokens_per_char = tokens_per_char
        self.calls: list[tuple[str, str]] = []

    def invoke(
        self, model_id: str, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> BedrockResponse:
        self.calls.append((model_id, prompt))
        if model_id in self.raise_for:
            raise RuntimeError(f"MockBedrockClient configured to raise for {model_id}")
        text = self.canned.get(model_id, f"[mock {model_id}] echo: {prompt[:60]}")
        # Toy token counting: ~4 chars per token.
        input_tokens = max(1, int((len(prompt) + len(system or "")) * self.tokens_per_char))
        output_tokens = max(1, int(len(text) * self.tokens_per_char))
        return BedrockResponse(
            text=text,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw={"mock": True},
        )


# ---------------------------------------------------------------------------
# Boto3BedrockClient — real AWS Bedrock
# ---------------------------------------------------------------------------


class Boto3BedrockClient:
    """Wraps the boto3 ``bedrock-runtime`` client.

    Currently supports Anthropic and Meta Llama model families via the
    ``invoke_model`` API. Extend ``_format_prompt`` to add others.
    """

    def __init__(self, region_name: str | None = None, client: Any | None = None) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "boto3 is not installed. Install with `pip install bedrock-router-mini[bedrock]`."
                ) from exc
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region_name or os.environ.get("AWS_REGION", "us-east-1"),
            )
        else:
            self.client = client

    def invoke(
        self, model_id: str, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> BedrockResponse:
        import json as _json

        body, extract = _format_prompt(model_id, prompt, system, max_tokens)
        resp = self.client.invoke_model(
            modelId=model_id,
            body=_json.dumps(body).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        data = _json.loads(resp["body"].read())
        text, input_tokens, output_tokens = extract(data)
        return BedrockResponse(
            text=text,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=data,
        )


def _format_prompt(model_id: str, prompt: str, system: str | None, max_tokens: int):
    """Return ``(body_dict, extractor_fn)`` for a given Bedrock model family."""
    if model_id.startswith("anthropic."):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        def extract(data):
            text = "".join(b.get("text", "") for b in data.get("content", []))
            usage = data.get("usage", {})
            return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

        return body, extract

    if model_id.startswith("meta.llama"):
        full_prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>" if system else "<|begin_of_text|>") + f"<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        body = {
            "prompt": full_prompt,
            "max_gen_len": max_tokens,
            "temperature": 0.2,
        }

        def extract(data):
            return (
                data.get("generation", ""),
                data.get("prompt_token_count", 0),
                data.get("generation_token_count", 0),
            )

        return body, extract

    if model_id.startswith("amazon.titan"):
        body = {
            "inputText": prompt if not system else f"{system}\n\n{prompt}",
            "textGenerationConfig": {"maxTokenCount": max_tokens, "temperature": 0.2},
        }

        def extract(data):
            results = data.get("results", [])
            text = results[0].get("outputText", "") if results else ""
            return (
                text,
                data.get("inputTextTokenCount", 0),
                results[0].get("tokenCount", 0) if results else 0,
            )

        return body, extract

    raise ValueError(f"Unsupported Bedrock model family for: {model_id}")


# ---------------------------------------------------------------------------
# AnthropicAsBedrockClient — Anthropic SDK behind a Bedrock-shaped surface
# ---------------------------------------------------------------------------


class AnthropicAsBedrockClient:
    """Calls the Anthropic API directly but presents the BedrockClient shape.

    Useful for benchmarking the router with real Claude responses when you
    don't have AWS Bedrock credentials. Pricing alias map sends router
    bookkeeping to the right Bedrock model entry.
    """

    # Map Bedrock model IDs to Anthropic API model names.
    _DEFAULT_MAP: dict[str, str] = {
        "anthropic.claude-haiku-4-5": "claude-haiku-4-5-20251001",
        "anthropic.claude-sonnet-4-6": "claude-sonnet-4-6",
        "anthropic.claude-3-5-haiku-20241022-v1:0": "claude-3-5-haiku-latest",
        "anthropic.claude-3-5-sonnet-20241022-v2:0": "claude-3-5-sonnet-latest",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_id_map: dict[str, str] | None = None,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "anthropic is not installed. Install with `pip install bedrock-router-mini[anthropic]`."
            ) from exc
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.map = {**self._DEFAULT_MAP, **(model_id_map or {})}

    def invoke(
        self, model_id: str, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> BedrockResponse:
        anthropic_model = self.map.get(model_id, model_id)
        kwargs: dict = {
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return BedrockResponse(
            text=text,
            model_id=model_id,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            raw=resp,
        )


# ---------------------------------------------------------------------------
# GroqAsBedrockClient — Groq Llama behind a Bedrock-shaped surface
# ---------------------------------------------------------------------------


class GroqAsBedrockClient:
    """Wraps Groq's Llama models behind the BedrockClient surface.

    Useful when you want to benchmark "Bedrock with Llama" without AWS bills —
    Groq's pricing on Llama matches Bedrock's closely (~$0.22/Mtok).
    """

    _DEFAULT_MAP: dict[str, str] = {
        "meta.llama3-1-8b-instruct-v1:0": "llama-3.1-8b-instant",
        "meta.llama3-3-70b-instruct-v1:0": "llama-3.3-70b-versatile",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model_id_map: dict[str, str] | None = None,
        min_interval_secs: float = 2.1,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("groq is not installed.") from exc
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.map = {**self._DEFAULT_MAP, **(model_id_map or {})}
        self.min_interval_secs = min_interval_secs
        self._last_call_t: float = 0.0

    def _wait(self) -> None:
        import time

        now = time.monotonic()
        delta = now - self._last_call_t
        if delta < self.min_interval_secs:
            time.sleep(self.min_interval_secs - delta)
        self._last_call_t = time.monotonic()

    def invoke(
        self, model_id: str, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> BedrockResponse:
        groq_model = self.map.get(model_id, model_id)
        self._wait()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=groq_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or ""
        input_tokens = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0
        output_tokens = getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
        return BedrockResponse(
            text=text,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=resp,
        )
