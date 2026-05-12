"""Pricing tests."""

from __future__ import annotations

import pytest

from bedrock_router_mini import Pricing, cost_for, get_pricing, register_pricing


def test_haiku_cost_math() -> None:
    # 1M input × $0.80 + 500K output × $4.00 = $0.80 + $2.00 = $2.80
    cost = cost_for("anthropic.claude-haiku-4-5", input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(2.80, rel=1e-9)


def test_llama_cost_cheaper_than_haiku() -> None:
    haiku = cost_for("anthropic.claude-haiku-4-5", input_tokens=1000, output_tokens=200)
    llama = cost_for("meta.llama3-1-8b-instruct-v1:0", input_tokens=1000, output_tokens=200)
    assert llama < haiku


def test_unknown_model_returns_zero() -> None:
    assert cost_for("not-a-real-model", input_tokens=1_000_000) == 0.0


def test_register_pricing_override() -> None:
    register_pricing("custom.model.v1", Pricing(input_per_mtok=10.0, output_per_mtok=20.0))
    p = get_pricing("custom.model.v1")
    assert p is not None
    assert p.input_per_mtok == 10.0


def test_sonnet_more_expensive_than_haiku() -> None:
    haiku = cost_for("anthropic.claude-haiku-4-5", input_tokens=1000, output_tokens=200)
    sonnet = cost_for("anthropic.claude-sonnet-4-6", input_tokens=1000, output_tokens=200)
    assert sonnet > haiku * 3
