"""Router behavior tests with MockBedrockClient."""

from __future__ import annotations

import pytest

from bedrock_router_mini import (
    BedrockRouter,
    MockBedrockClient,
    Route,
    by_char_length,
    by_keyword,
    by_token_count,
)


# ---------------------------------------------------------------------------
# routing condition unit tests
# ---------------------------------------------------------------------------


def test_by_char_length() -> None:
    cond = by_char_length(10)
    assert cond("hi") is False
    assert cond("a" * 100) is True
    assert cond("a" * 10) is False  # strictly greater


def test_by_token_count() -> None:
    cond = by_token_count(5, chars_per_token=4.0)
    assert cond("a" * 19) is False  # ~4.75 tokens
    assert cond("a" * 25) is True  # ~6.25 tokens


def test_by_keyword_case_insensitive() -> None:
    cond = by_keyword(["analyze", "compare"])
    assert cond("Please ANALYZE the data") is True
    assert cond("Just say hi") is False


def test_by_keyword_case_sensitive() -> None:
    cond = by_keyword(["Analyze"], case_sensitive=True)
    assert cond("Please Analyze") is True
    assert cond("please analyze") is False


# ---------------------------------------------------------------------------
# router selection
# ---------------------------------------------------------------------------


def test_router_picks_first_matching_route() -> None:
    client = MockBedrockClient()
    router = BedrockRouter(
        client,
        routes=[
            Route(by_keyword(["explain", "reasoning"]), "anthropic.claude-haiku-4-5"),
            Route(lambda _: True, "meta.llama3-1-8b-instruct-v1:0"),
        ],
    )
    model, _ = router.select_model("Please explain the proof")
    assert model == "anthropic.claude-haiku-4-5"
    model, _ = router.select_model("yes")
    assert model == "meta.llama3-1-8b-instruct-v1:0"


def test_router_uses_default_model_when_no_match() -> None:
    client = MockBedrockClient()
    router = BedrockRouter(
        client,
        routes=[Route(by_keyword(["analyze"]), "anthropic.claude-haiku-4-5")],
        default_model="meta.llama3-1-8b-instruct-v1:0",
    )
    resp = router.invoke("just a greeting")
    assert resp.success
    assert resp.model_used == "meta.llama3-1-8b-instruct-v1:0"


def test_router_raises_when_no_route_matches_and_no_default() -> None:
    client = MockBedrockClient()
    router = BedrockRouter(
        client,
        routes=[Route(by_keyword(["nope"]), "anthropic.claude-haiku-4-5")],
    )
    with pytest.raises(RuntimeError):
        router.invoke("hello world")


# ---------------------------------------------------------------------------
# fallback chain
# ---------------------------------------------------------------------------


def test_fallback_chain_on_error() -> None:
    client = MockBedrockClient(raise_for={"anthropic.claude-haiku-4-5"})
    router = BedrockRouter(
        client,
        routes=[Route(lambda _: True, "anthropic.claude-haiku-4-5")],
        fallback_chain=["meta.llama3-1-8b-instruct-v1:0"],
    )
    resp = router.invoke("test")
    assert resp.success is True
    assert resp.fallback_triggered is True
    assert resp.model_used == "meta.llama3-1-8b-instruct-v1:0"
    assert len(resp.attempts) == 2
    assert resp.attempts[0].success is False
    assert "RuntimeError" in (resp.attempts[0].error or "")


def test_fallback_chain_all_fail() -> None:
    client = MockBedrockClient(
        raise_for={"anthropic.claude-haiku-4-5", "meta.llama3-1-8b-instruct-v1:0"}
    )
    router = BedrockRouter(
        client,
        routes=[Route(lambda _: True, "anthropic.claude-haiku-4-5")],
        fallback_chain=["meta.llama3-1-8b-instruct-v1:0"],
    )
    resp = router.invoke("test")
    assert resp.success is False
    assert resp.model_used is None
    assert len(resp.attempts) == 2
    assert resp.text == ""


def test_fallback_skips_duplicate_of_primary() -> None:
    client = MockBedrockClient()
    router = BedrockRouter(
        client,
        routes=[Route(lambda _: True, "anthropic.claude-haiku-4-5")],
        fallback_chain=["anthropic.claude-haiku-4-5", "meta.llama3-1-8b-instruct-v1:0"],
    )
    resp = router.invoke("test")
    # Primary succeeded; only one attempt.
    assert len(resp.attempts) == 1


# ---------------------------------------------------------------------------
# cost + audit
# ---------------------------------------------------------------------------


def test_response_includes_cost_and_attempts() -> None:
    client = MockBedrockClient()
    router = BedrockRouter(
        client,
        routes=[Route(lambda _: True, "anthropic.claude-haiku-4-5")],
    )
    resp = router.invoke("hi")
    assert resp.cost_usd > 0
    assert resp.success
    assert resp.model_used == "anthropic.claude-haiku-4-5"
    assert resp.attempts[0].cost_usd == resp.cost_usd


def test_response_preserves_raw_response_text() -> None:
    canned = "the moon is round"
    client = MockBedrockClient(canned={"anthropic.claude-haiku-4-5": canned})
    router = BedrockRouter(
        client,
        routes=[Route(lambda _: True, "anthropic.claude-haiku-4-5")],
    )
    resp = router.invoke("?")
    assert resp.text == canned


def test_router_records_matched_route_label() -> None:
    client = MockBedrockClient()
    router = BedrockRouter(
        client,
        routes=[Route(by_char_length(5), "anthropic.claude-haiku-4-5", "long")],
        default_model="meta.llama3-1-8b-instruct-v1:0",
    )
    resp_long = router.invoke("a long prompt")
    resp_short = router.invoke("hi")
    assert resp_long.matched_route_label == "long"
    assert resp_short.matched_route_label is None
