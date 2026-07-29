"""Live integration check against the GB10's Ollama server.

Not part of the unit suite -- this one really talks to the box. Run it after
any change to the LLM layer, and after any change to which models are pulled:

    .venv/bin/python ops/check_gb10.py

It verifies, in order: the server is reachable, both configured tiers exist,
each tier generates, and constrained JSON decoding actually produces the schema
the agent layer will depend on.
"""

from __future__ import annotations

import sys
import time

from spintrader.llm.router import LLMRouter, Tier

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["buy", "sell", "hold"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["action", "confidence", "rationale"],
}

PROMPT = (
    "Bitcoin is trading at $61,200, down 4.2% today, with the 50-day moving "
    "average at $63,800 and RSI at 31. Funding rates are slightly negative. "
    "Give a single trading decision as JSON."
)


def main() -> int:
    router = LLMRouter()
    failures: list[str] = []

    print("=" * 68)
    print("GB10 LLM integration check")
    print("=" * 68)

    health = router.health_check()
    print(f"  endpoint : {health['base_url']}")
    if not health["ok"]:
        print(f"  FAIL     : {health['error']}")
        return 1
    print(f"  quick    : {health['quick_model']}")
    print(f"  deep     : {health['deep_model']}")
    print()

    for tier in (Tier.QUICK, Tier.DEEP):
        label = f"{tier.value:>5} ({router.model_for(tier)})"
        started = time.monotonic()
        try:
            value, response = router.complete_json(
                PROMPT, schema=DECISION_SCHEMA, tier=tier, max_tokens=400,
            )
        except Exception as exc:                        # noqa: BLE001 - reported below
            print(f"  {label}: FAIL -- {type(exc).__name__}: {exc}")
            failures.append(tier.value)
            continue

        elapsed = time.monotonic() - started
        tok_s = response.completion_tokens / elapsed if elapsed > 0 else 0.0
        print(f"  {label}")
        print(f"      action     : {value.get('action')} @ {value.get('confidence')}")
        print(f"      rationale  : {str(value.get('rationale', ''))[:80]}...")
        print(f"      tokens     : {response.prompt_tokens} in / "
              f"{response.completion_tokens} out")
        print(f"      wall clock : {elapsed:.1f}s ({tok_s:.1f} tok/s)")

        # A schema violation here means the agent layer cannot trust
        # constrained decoding and needs a validation pass of its own.
        if value.get("action") not in {"buy", "sell", "hold"}:
            print(f"      WARN: action {value.get('action')!r} violates the schema enum")
            failures.append(f"{tier.value}-schema")
        print()

    print(f"  budget   : {router.budget_snapshot()}")
    print("=" * 68)

    if failures:
        print(f"RESULT: FAIL ({', '.join(failures)})")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
