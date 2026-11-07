"""Held-out threshold selection in bench/boolq_router.py (offline, synthetic rows)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "boolq_router.py"


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("boolq_router", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["boolq_router"] = module
    spec.loader.exec_module(module)
    return module


def _rows(lengths: list[int], llama_right: list[bool]) -> tuple[list[dict], dict[str, dict]]:
    rows, haiku = [], {}
    for i, (length, ok) in enumerate(zip(lengths, llama_right, strict=True)):
        rows.append(
            {
                "id": str(i),
                "question": "q?",
                "passage": "p" * min(length, 400),
                "passage_full_len": length,
                "gold": True,
                "predicted": "YES" if ok else "NO",
            }
        )
        haiku[str(i)] = {"predicted": "YES", "input_tokens": 200, "output_tokens": 13}
    return rows, haiku


def test_split_is_seeded_disjoint_and_complete(bench: ModuleType) -> None:
    rows = [{"id": str(i)} for i in range(200)]
    dev, test = bench.split_dev_test(rows)
    assert (dev, test) == bench.split_dev_test(rows)
    ids = [r["id"] for r in dev + test]
    assert len(dev) == len(test) == 100
    assert sorted(ids, key=int) == [str(i) for i in range(200)]


def test_threshold_is_the_cheapest_that_matches_haiku_on_dev(bench: ModuleType) -> None:
    # Llama is right on short passages only; Haiku is always right.
    rows, haiku = _rows([100, 200, 300, 900, 1000], [True, True, True, False, False])
    assert bench.choose_threshold(rows, haiku) == 300


def test_threshold_sends_everything_to_llama_when_it_is_as_accurate(bench: ModuleType) -> None:
    rows, haiku = _rows([100, 600, 1200], [True, True, True])
    threshold = bench.choose_threshold(rows, haiku)
    routed = bench._evaluate(rows, "router", char_threshold=threshold, haiku_map=haiku)
    assert routed["route_counts"] == {"meta.llama3-1-8b-instruct-v1:0": 3}
