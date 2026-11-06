# bedrock-router-mini

> Cost-aware multi-model router for AWS Bedrock with complexity-based routing and graceful fallback. **46.3% cost savings** vs always-Haiku on a 197-question BoolQ benchmark, with only a 2.5-point accuracy delta.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

200-question BoolQ subset, three routing strategies, costs computed at AWS Bedrock on-demand pricing for us-east-1 from the AWS Price List API, checked 2026-11-06 (Llama-3.1-8B-Instruct @ $0.22 per Mtok in/out, Claude Haiku 4.5 @ $1.00 in / $5.00 out per Mtok, global endpoint).

| Strategy | n | Accuracy | Total cost (USD) | Cost / call | Models used |
|---|---:|---:|---:|---:|---|
| `always_llama` (baseline cheap) | 198 | 0.8333 | $0.00458 | $0.000023 | Llama-3.1-8B × 198 |
| `always_haiku` (baseline smart) | 197 | **0.8579** | $0.04430 | $0.000225 | Haiku-4.5 × 197 |
| **`router`** (passage > 500 chars → Haiku, else Llama) | 197 | 0.8325 | **$0.02380** | $0.000121 | Haiku × 93, Llama × 104 |

**Router vs always-Haiku: cost savings = 46.3%, accuracy delta = −2.5 pts.**

The router gives you 97% of Haiku's accuracy at 54% of the cost. You can dial the threshold to favor either side — push it higher (route more to Llama) to save more, lower (route more to Haiku) to preserve accuracy. Raw numbers in [`bench/router_results.json`](bench/router_results.json).

Real LLM calls were made via the Anthropic API (Haiku) and Groq (Llama-3.1-8B) using the project's adapters — there's no synthetic step. Cost numbers use Bedrock list pricing because that's where this router is designed to deploy.

Reproduce:
```bash
export ANTHROPIC_API_KEY=sk-ant-...   # for the Haiku tier
python bench/boolq_router.py --limit 200 --char-threshold 500
```

## What this is

A drop-in router for AWS Bedrock that picks a model per call from a configurable rule list and falls back through a chain of models on error. The router speaks Bedrock's `invoke_model` shape but works with **any** adapter implementing the `BedrockClient` protocol — real Bedrock via boto3, Anthropic API direct, Groq API direct, or a deterministic mock for tests.

## Architecture

```
           ┌──────────────────────────────────────┐
           │            BedrockRouter             │
           │                                      │
prompt ───▶│  1. select_model(prompt)             │
           │     - scan routes in order           │
           │     - first matching condition wins  │
           │                                      │
           │  2. invoke(primary)                  │
           │     ├─ ok  → return                  │
           │     └─ err → try fallback_chain[0]   │
           │             ├─ ok  → return          │
           │             └─ err → fallback[1]…    │
           │                                      │
           │  3. record cost, latency, attempts   │
           └──────────────────────────────────────┘
                            │
                            ▼
           ┌───────────────────────────────────┐
           │       BedrockClient (Protocol)    │
           ├───────────────────────────────────┤
           │  Boto3BedrockClient    (real)     │
           │  AnthropicAsBedrock    (proxy)    │
           │  GroqAsBedrock         (proxy)    │
           │  MockBedrockClient     (test)     │
           └───────────────────────────────────┘
```

Every call returns a `RouterResponse` with the model used, the attempt log (incl. failed ones), the matched route label, the cumulative cost in USD, and the raw underlying response.

## Quickstart

```bash
pip install -e ".[anthropic,groq,server]"
```

```python
from bedrock_router_mini import (
    BedrockRouter, Route, by_char_length, by_keyword, AnthropicAsBedrockClient,
)

router = BedrockRouter(
    client=AnthropicAsBedrockClient(),   # or Boto3BedrockClient() with AWS creds
    routes=[
        # Analytical / long prompts → Haiku
        Route(by_keyword(["analyze", "explain", "compare"]),
              "anthropic.claude-haiku-4-5", "analytic"),
        Route(by_char_length(800),
              "anthropic.claude-haiku-4-5", "long-prompt"),
        # Everything else → cheap default
        Route(lambda _: True,
              "anthropic.claude-3-5-haiku-latest", "default"),
    ],
    fallback_chain=["anthropic.claude-3-5-haiku-latest"],
)

result = router.invoke("Hi")
print(result.model_used)            # anthropic.claude-3-5-haiku-latest
print(result.cost_usd)              # 0.000018
print(result.matched_route_label)   # default
print(result.attempts)              # [AttemptRecord(...)]
```

## Three built-in routing conditions

```python
from bedrock_router_mini import by_char_length, by_token_count, by_keyword

by_char_length(800)                  # prompt has > 800 characters
by_token_count(200, chars_per_token=4.0)   # ~> 200 tokens
by_keyword(["explain", "compare"])   # case-insensitive substring match
```

Custom rules are any `Callable[[str], bool]`:

```python
import re

def has_code_block(prompt: str) -> bool:
    return bool(re.search(r"```", prompt))

Route(has_code_block, "anthropic.claude-sonnet-4-6", "code-task")
```

## Adapters

| Adapter | Backend | When to use |
|---|---|---|
| `Boto3BedrockClient` | `bedrock-runtime` via boto3 | Production. Supports Anthropic, Llama, Titan model families. |
| `AnthropicAsBedrockClient` | Anthropic API direct | When you want Claude responses but don't have AWS creds. Pricing aliased to Bedrock prices for honest cost comparisons. |
| `GroqAsBedrockClient` | Groq API direct (Llama, free tier) | When you want Llama responses without AWS. Rate-limited to 28 RPM by default. |
| `MockBedrockClient` | In-memory | Tests. Returns canned responses, can raise per-model errors. |

All four expose the same `invoke(model_id, prompt, system=..., max_tokens=...) -> BedrockResponse` surface.

## Fallback chain

```python
router = BedrockRouter(
    client=AnthropicAsBedrockClient(),
    routes=[
        Route(lambda _: True, "anthropic.claude-haiku-4-5", "primary"),
    ],
    fallback_chain=[
        "anthropic.claude-3-5-haiku-latest",   # backup
        "anthropic.claude-3-5-sonnet-latest",  # last-resort smart model
    ],
)

# When Haiku 4.5 errors (e.g. transient 5xx), the router retries with 3.5-haiku.
# If that also fails, it tries Sonnet. Every attempt is recorded.
result = router.invoke("hello")
if result.fallback_triggered:
    print("Primary failed:", result.attempts[0].error)
    print("Served by:", result.model_used)
```

Covered by `tests/test_router.py::test_fallback_chain_on_error`.

## FastAPI server

`examples/server.py` exposes `/invoke` and `/health` endpoints that proxy a real Bedrock-style call through the router:

```bash
pip install -e ".[anthropic,server]"
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn examples.server:app --port 8080
```

```bash
curl -X POST localhost:8080/invoke -H 'content-type: application/json' \
    -d '{"prompt": "Please analyze the trade-offs between..."}'
# → routes to claude-haiku-4-5 (keyword hit on "analyze")

curl -X POST localhost:8080/invoke -H 'content-type: application/json' \
    -d '{"prompt": "Hi"}'
# → routes to claude-3-5-haiku-latest (default, cheaper)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
20 passed in 0.09s
```

Covers each routing condition, route ordering, default-model behavior, fallback chain on error, fallback-all-fail, cost / pricing math, response shape preservation, attempt records.

## Pricing customization

Built-in pricing tracks AWS Bedrock on-demand prices for us-east-1, checked 2026-11-06 against the AWS Price List API (Claude prices are for global endpoints; regional endpoints cost 10% more). Override for enterprise rates or other regions:

```python
from bedrock_router_mini import Pricing, register_pricing

register_pricing(
    "anthropic.claude-haiku-4-5",
    Pricing(input_per_mtok=0.50, output_per_mtok=2.50),   # 50% off via your EDP
)
```

## Project layout

```
.
├── src/bedrock_router_mini/
│   ├── __init__.py
│   ├── clients.py        # Boto3 / AnthropicAsBedrock / GroqAsBedrock / Mock
│   ├── pricing.py        # Per-model Bedrock pricing table
│   └── router.py         # BedrockRouter, Route, RouterResponse
├── tests/                # 20 pytest cases
├── bench/
│   ├── boolq_router.py        # 200-question BoolQ routing benchmark
│   ├── boolq_200_augmented.jsonl  # cached BoolQ inputs with true passage lengths
│   ├── haiku_boolq.jsonl      # real Haiku 4.5 outputs
│   └── router_results.json    # per-strategy cost + accuracy
└── examples/
    └── server.py         # FastAPI /invoke server
```

## Limitations

**Cost savings depend on the workload.** On BoolQ, Llama-3.1-8B is already 83% accurate, so routing the easy half to Llama costs only 2.5 accuracy points. On tasks where Llama is noticeably worse than Claude (long-context reasoning, code generation, multi-step instructions), the same threshold may drop accuracy by more — measure before deploying.

**Routing rules are static.** The built-in conditions are length-, token-, and keyword-based. A natural v0.2 extension is a *learned* router — a tiny classifier that predicts which model will get this prompt right — but that adds an extra LLM call per prompt and may erode the savings on short prompts. Out of scope for v0.1.

**Streaming not yet wired.** `invoke_model` is supported; `invoke_model_with_response_stream` is not. Streaming + the cost / latency accounting need a unified hook that handles a chunked response. Tracked for v0.2.

**Single-region.** `Boto3BedrockClient` takes a `region_name`. Cross-region routing (e.g., fallback to a different region on throttling) is not built in but is one extra parameter away.

## License

MIT — see [LICENSE](LICENSE).
