"""Pricing tests."""

from __future__ import annotations

import pytest

from bedrock_router_mini import Pricing, cost_for, get_pricing, register_pricing


def test_haiku_cost_math() -> None:
    # 1M input × $1.00 + 500K output × $5.00 = $1.00 + $2.50 = $3.50
    cost = cost_for("anthropic.claude-haiku-4-5", input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(3.50, rel=1e-9)


def test_opus_4_7_uses_current_price() -> None:
    # $5 / $25 per Mtok, not the Opus 4.1-era $15 / $75.
    cost = cost_for("anthropic.claude-opus-4-7", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(30.0, rel=1e-9)


def test_anthropic_adapter_alias_matches_bedrock_id() -> None:
    assert get_pricing("_anthropic_haiku_45") == get_pricing("anthropic.claude-haiku-4-5")


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


def test_sonnet_costs_three_times_haiku() -> None:
    haiku = cost_for("anthropic.claude-haiku-4-5", input_tokens=1000, output_tokens=200)
    sonnet = cost_for("anthropic.claude-sonnet-4-6", input_tokens=1000, output_tokens=200)
    assert sonnet == pytest.approx(haiku * 3, rel=1e-9)
