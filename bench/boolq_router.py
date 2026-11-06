"""Routing benchmark on BoolQ.

Reuses the 1000 Llama-3.1-8B BoolQ outputs collected by the ``conf-calib``
project (committed at ``../conf-calib/bench/llama_boolq.jsonl``). Adds fresh
Haiku 4.5 outputs for the same passage/question pairs, then evaluates three
strategies:

* ``always_llama``   — every prompt to Llama-3.1-8B
* ``always_haiku``   — every prompt to Claude Haiku 4.5
* ``router``         — passage above ``char_threshold`` chars goes to Haiku,
  shorter ones go to Llama (the "cheap when easy, smart when complex" policy)

Cost is computed using public Bedrock pricing (mirroring the production
target this project's CV bullet describes). Accuracy is the fraction of
YES/NO predictions that match BoolQ's gold answer.

The full-set table scores ``--char-threshold`` on the same rows it reports.
The held-out section instead splits the rows 50/50 (seed 0), picks the
cheapest threshold whose dev accuracy matches always-Haiku's dev accuracy,
and scores every strategy on the test half.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bedrock_router_mini.pricing import cost_for

BENCH = Path(__file__).resolve().parent
LLAMA_CACHE = Path("../conf-calib/bench/llama_boolq.jsonl").resolve()
AUGMENTED_CACHE = BENCH / "boolq_200_augmented.jsonl"
HAIKU_CACHE = BENCH / "haiku_boolq.jsonl"
RESULTS = BENCH / "router_results.json"
HOLDOUT_SEED = 0
STRATEGIES = ["always_llama", "always_haiku", "router"]

SYSTEM = (
    "You answer factual yes/no questions grounded in a short passage. "
    "Reply with EXACTLY one line in the format:\n"
    "ANSWER: <YES or NO>\n"
    "CONFIDENCE: <integer 0-100>\n"
    "No other text."
)

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(YES|NO)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*:\s*([0-9]+)", re.IGNORECASE)


def _parse(text: str) -> tuple[str | None, float | None]:
    a = _ANSWER_RE.search(text or "")
    c = _CONF_RE.search(text or "")
    label = a.group(1).upper() if a else None
    conf = float(c.group(1)) / 100.0 if c else None
    return label, conf


def _load_llama(limit: int) -> list[dict]:
    """Load augmented rows (with full passage lengths) if present, else raw cache."""
    source = AUGMENTED_CACHE if AUGMENTED_CACHE.exists() else LLAMA_CACHE
    rows = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _routing_len(ex: dict) -> int:
    """Pick the best available passage length signal.

    Prefers the un-truncated ``passage_full_len`` from the augmented file;
    falls back to ``len(passage)`` (which is clipped at 400 in the raw cache)
    so the function works on either input.
    """
    return int(ex.get("passage_full_len") or len(ex.get("passage", "")))


def _existing(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" in obj:
                out[str(obj["id"])] = obj
    return out


def collect_haiku(rows: list[dict], model: str = "claude-haiku-4-5-20251001") -> None:
    """Append Haiku verdicts to ``haiku_boolq.jsonl``. Resumable."""
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    done = _existing(HAIKU_CACHE)
    with HAIKU_CACHE.open("a", encoding="utf-8") as out:
        for i, ex in enumerate(rows, 1):
            if str(ex["id"]) in done:
                continue
            prompt = f"Passage:\n{ex['passage']}\n\nQuestion: {ex['question']}"
            start = time.perf_counter()
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=40,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens
                err = None
            except Exception as exc:  # noqa: BLE001
                text = ""
                in_tok = out_tok = 0
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
            elapsed = int((time.perf_counter() - start) * 1000)
            label, conf = _parse(text)
            row = {
                "id": ex["id"],
                "predicted": label,
                "confidence": conf,
                "raw_text": text,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "elapsed_ms": elapsed,
                "error": err,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 20 == 0:
                print(f"  [{i}/{len(rows)}]")


def _evaluate(
    rows: list[dict],
    strategy: str,
    *,
    char_threshold: int = 400,
    haiku_map: dict[str, dict] | None = None,
) -> dict:
    """Return strategy summary dict."""
    if haiku_map is None:
        haiku_map = _existing(HAIKU_CACHE)

    LLAMA_MODEL = "meta.llama3-1-8b-instruct-v1:0"
    HAIKU_MODEL = "anthropic.claude-haiku-4-5"

    n = 0
    correct = 0
    total_cost = 0.0
    route_counts: dict[str, int] = {}

    for ex in rows:
        gold = bool(ex["gold"])
        plen = _routing_len(ex)
        haiku_row = haiku_map.get(str(ex["id"]))
        needs_haiku = strategy == "always_haiku" or (strategy == "router" and plen > char_threshold)
        if needs_haiku and haiku_row is None:
            continue  # required haiku verdict missing — skip

        if strategy == "always_llama":
            model = LLAMA_MODEL
            label = ex["predicted"]
            in_tok = int(len(ex["passage"]) / 4) + int(len(ex["question"]) / 4)
            out_tok = 5
        elif strategy == "always_haiku":
            model = HAIKU_MODEL
            label = haiku_row["predicted"]
            in_tok = haiku_row["input_tokens"]
            out_tok = haiku_row["output_tokens"]
        elif strategy == "router":
            if plen > char_threshold:
                model = HAIKU_MODEL
                label = haiku_row["predicted"]
                in_tok = haiku_row["input_tokens"]
                out_tok = haiku_row["output_tokens"]
            else:
                model = LLAMA_MODEL
                label = ex["predicted"]
                in_tok = int(len(ex["passage"]) / 4) + int(len(ex["question"]) / 4)
                out_tok = 5
        else:
            raise ValueError(strategy)

        if label is None:
            continue

        n += 1
        route_counts[model] = route_counts.get(model, 0) + 1
        total_cost += cost_for(model, input_tokens=in_tok, output_tokens=out_tok)
        predicted_yes = label == "YES"
        if predicted_yes == gold:
            correct += 1

    return {
        "strategy": strategy,
        "n": n,
        "accuracy": round(correct / max(n, 1), 4),
        "total_cost_usd": round(total_cost, 5),
        "cost_per_call_usd": round(total_cost / max(n, 1), 6),
        "route_counts": route_counts,
        "char_threshold": char_threshold if strategy == "router" else None,
    }


def split_dev_test(rows: list[dict], seed: int = HOLDOUT_SEED) -> tuple[list[dict], list[dict]]:
    """Seeded 50/50 split: the threshold is chosen on dev and scored on test."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    half = len(rows) // 2
    return [rows[i] for i in order[:half]], [rows[i] for i in order[half:]]


def choose_threshold(dev: list[dict], haiku_map: dict[str, dict]) -> int:
    """Cheapest threshold whose dev accuracy is at least always-Haiku's.

    Threshold 0 sends every row to Haiku, so it always qualifies; the largest
    dev passage length sends every dev row to Llama.
    """
    target = _evaluate(dev, "always_haiku", haiku_map=haiku_map)["accuracy"]
    best_threshold, best_cost = 0, float("inf")
    for threshold in sorted({0, *(_routing_len(r) for r in dev)}):
        s = _evaluate(dev, "router", char_threshold=threshold, haiku_map=haiku_map)
        if s["accuracy"] >= target and s["total_cost_usd"] < best_cost:
            best_threshold, best_cost = threshold, s["total_cost_usd"]
    return best_threshold


def _print_table(summaries: list[dict]) -> None:
    print("| Strategy | n | Accuracy | Cost USD | Cost/call | Models used |")
    print("|---|---:|---:|---:|---:|---|")
    for summary in summaries:
        rc = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(summary["route_counts"].items()))
        print(
            f"| {summary['strategy']} | {summary['n']} | {summary['accuracy']} | "
            f"{summary['total_cost_usd']} | {summary['cost_per_call_usd']} | {rc} |"
        )


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--char-threshold", type=int, default=400)
    p.add_argument("--skip-collect", action="store_true",
                   help="Use existing haiku_boolq.jsonl without making new calls")
    args = p.parse_args()

    rows = _load_llama(args.limit)
    print(f"Loaded {len(rows)} BoolQ rows with cached Llama verdicts.")

    if not args.skip_collect:
        # Only call Haiku on the items the router would actually route to Haiku.
        long_rows = [r for r in rows if _routing_len(r) > args.char_threshold]
        all_haiku_rows = rows  # for always-Haiku comparison
        target = all_haiku_rows
        done_count = len(_existing(HAIKU_CACHE))
        print(
            f"Long passages (route-to-Haiku): {len(long_rows)} / {len(rows)} "
            f"(threshold={args.char_threshold} chars)"
        )
        print(f"Existing Haiku cache: {done_count} rows. Collecting any missing for ALL {len(target)} rows...")
        collect_haiku(target)

    out = {"char_threshold": args.char_threshold, "n_target": args.limit, "summaries": []}
    out["summaries"] = [
        _evaluate(rows, s, char_threshold=args.char_threshold) for s in STRATEGIES
    ]
    print()
    _print_table(out["summaries"])

    # Headline lift: savings vs always-Haiku.
    by_strat = {s["strategy"]: s for s in out["summaries"]}
    a_haiku = by_strat["always_haiku"]
    router = by_strat["router"]
    if a_haiku["total_cost_usd"] > 0:
        savings = 1.0 - (router["total_cost_usd"] / a_haiku["total_cost_usd"])
        acc_delta = router["accuracy"] - a_haiku["accuracy"]
        print(
            f"\nRouter vs always-Haiku: cost savings = {savings:.1%}, "
            f"accuracy delta = {acc_delta:+.3f}"
        )
        out["savings_vs_haiku"] = round(savings, 4)
        out["accuracy_delta_vs_haiku"] = round(acc_delta, 4)

    haiku_map = _existing(HAIKU_CACHE)
    dev, test = split_dev_test(rows)
    chosen = choose_threshold(dev, haiku_map)
    out["holdout"] = {
        "seed": HOLDOUT_SEED,
        "n_dev": len(dev),
        "n_test": len(test),
        "objective": "cheapest char_threshold with dev accuracy >= always_haiku dev accuracy",
        "chosen_threshold": chosen,
        "test": [
            _evaluate(test, s, char_threshold=chosen, haiku_map=haiku_map) for s in STRATEGIES
        ],
    }
    print(f"\nHeld out: threshold {chosen} chosen on {len(dev)} dev rows, scored on {len(test)}:")
    _print_table(out["holdout"]["test"])

    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
