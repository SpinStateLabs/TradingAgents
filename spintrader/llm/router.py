"""Tiered LLM routing against the GB10's Ollama server.

Why tiering exists
------------------
One trading decision fans out into dozens of LLM calls: per-analyst reports,
bull/bear debate rounds, risk arguments, then a final synthesis. Most of those
calls are routine -- summarise this news item, extract these figures, score this
signal -- and a 30B MoE handles them at a fraction of the latency. Only the
adjudication steps justify the 99 GB GLM-4.5-Air, which on a 121 GB
unified-memory box evicts everything else while it is resident.

Routing every call to the deep model would make a single cycle take hours and
starve the database and feature pipelines of memory. Routing every call to the
quick model would make the final decision noticeably worse. Hence two tiers,
chosen explicitly at each call site.

Tier batching is mandatory, not an optimisation
-----------------------------------------------
Measured on this GB10 (2026-07-29)::

    qwen3:30b-a3b            load  8.5 s   generation  91.0 tok/s
    GLM-4.5-Air:Q6_K         load 34.6 s   generation  14.6 tok/s

The two tiers total 117 GB against 121 GB of unified memory, so they cannot be
co-resident alongside Postgres and the feature pipelines. Ollama therefore
evicts one to load the other, and every tier switch costs a ~35 s reload.
Interleaving quick and deep calls turns a 12-minute cycle into an hour of
thrashing.

Callers must consequently group work by tier: run every ``Tier.QUICK`` call,
then every ``Tier.DEEP`` call. :meth:`LLMRouter.run_batched` enforces that
ordering. ``keep_alive`` is exposed so a batch can pin its model for the
duration and release it immediately afterwards.

Budgets
-------
A debate loop that fails to converge can otherwise burn the GPU indefinitely.
:class:`CycleBudget` caps both call count and token spend per decision cycle
and raises rather than silently truncating -- a partially-completed decision
must be visibly broken, not quietly acted upon.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping

import requests

from spintrader.core.config import LLMConfig

log = logging.getLogger(__name__)


class Tier(str, Enum):
    """Which model tier a call should be routed to."""
    QUICK = "quick"
    DEEP = "deep"


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Base for router failures."""


class BudgetExceeded(LLMError):
    """The decision cycle's call or token allowance is exhausted."""


class StructuredOutputError(LLMError):
    """The model did not return usable JSON after all repair attempts."""


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------

@dataclass(slots=True)
class LLMResponse:
    """One completion plus the accounting needed to reason about cost."""
    text: str
    model: str
    tier: Tier
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class CycleBudget:
    """Per-cycle call and token allowance.

    Thread-safe: analyst calls are fanned out across a thread pool, so the
    counters are mutated concurrently.
    """
    max_calls: int
    max_tokens: int
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reserve(self) -> None:
        """Claim one call slot, raising if the cycle is already exhausted.

        Checked *before* the request so a runaway loop is stopped at the
        boundary rather than after paying for another generation.
        """
        with self._lock:
            if self.calls >= self.max_calls:
                raise BudgetExceeded(
                    f"cycle call budget exhausted: {self.calls}/{self.max_calls} calls"
                )
            if self.total_tokens >= self.max_tokens:
                raise BudgetExceeded(
                    f"cycle token budget exhausted: {self.total_tokens}/{self.max_tokens} tokens"
                )
            self.calls += 1

    def record(self, response: LLMResponse) -> None:
        with self._lock:
            self.prompt_tokens += response.prompt_tokens
            self.completion_tokens += response.completion_tokens

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "max_calls": self.max_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "max_tokens": self.max_tokens,
            }

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Strip <think>...</think> blocks: qwen3 and other reasoning models emit them
# even when asked for pure JSON, and they frequently contain braces that
# confuse naive extraction.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Local models wrap JSON in prose, code fences and reasoning blocks no matter
    how firmly the prompt forbids it. Rather than fight that with prompt
    engineering alone, parse defensively in four escalating passes.
    """
    if not text or not text.strip():
        raise StructuredOutputError("model returned an empty response")

    cleaned = _THINK_RE.sub("", text).strip()

    # 1. The whole thing is already valid JSON.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2. It is inside a ```json fence.
    for match in _FENCE_RE.finditer(cleaned):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    # 3. Brace matching from the first opener, respecting strings and escapes,
    #    so a `{` inside a rationale string does not truncate the object.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break
        # fall through to the next bracket style

    # 4. Give up, but surface enough of the response to debug the prompt.
    preview = cleaned[:400] + ("..." if len(cleaned) > 400 else "")
    raise StructuredOutputError(f"no parseable JSON in model response: {preview!r}")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMRouter:
    """Routes completions to the appropriate GB10 model tier.

    Construct one per process and share it; the underlying ``requests.Session``
    pools connections and is safe to use from multiple threads.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        budget: CycleBudget | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.budget = budget or CycleBudget(
            max_calls=self.config.max_calls_per_cycle,
            max_tokens=self.config.max_tokens_per_cycle,
        )
        self._session = session or requests.Session()

    # -- model selection ---------------------------------------------------

    def model_for(self, tier: Tier) -> str:
        return self.config.deep_model if tier is Tier.DEEP else self.config.quick_model

    # -- core call ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        tier: Tier = Tier.QUICK,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_schema: Mapping[str, Any] | None = None,
        stop: list[str] | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        """Run a single completion.

        ``json_schema`` engages Ollama's constrained decoding, which is far more
        reliable than asking politely for JSON in the prompt -- the grammar
        makes malformed output impossible rather than merely unlikely.

        ``keep_alive`` controls how long Ollama keeps the model resident after
        the call (``"30m"``, ``"0"`` to evict immediately). Use it to pin a
        model across a batch and drop it afterwards, so the other tier is not
        forced to wait on a reload.

        ``think`` toggles reasoning-model thinking. Set it to ``False`` for
        structured calls: Ollama routes a reasoning model's output into a
        separate ``thinking`` field and applies the JSON grammar there, leaving
        ``response`` empty. Measured on qwen3:30b-a3b, disabling thinking also
        cut a decision call from 132 wasted tokens to 159 useful ones in 1.8 s.
        """
        self.budget.reserve()
        model = self.model_for(tier)

        options: dict[str, Any] = {
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = stop

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if json_schema is not None:
            payload["format"] = dict(json_schema)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think

        url = f"{self.config.base_url.rstrip('/')}/api/generate"
        response = self._post_with_retry(url, payload, model=model, tier=tier)
        self.budget.record(response)
        return response

    def complete_json(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any] | None = None,
        tier: Tier = Tier.QUICK,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        repair_attempts: int = 1,
        think: bool | None = False,
    ) -> tuple[Any, LLMResponse]:
        """Complete and parse JSON, retrying once with a repair prompt.

        Returns the parsed value alongside the raw response so callers can still
        log token spend and latency for attribution.

        ``think`` defaults to ``False`` here because reasoning and constrained
        decoding interact badly (see :meth:`complete`). Pass ``think=None`` to
        restore the model's own default when a call genuinely benefits from
        deliberation.
        """
        response = self.complete(
            prompt, tier=tier, system=system, temperature=temperature,
            max_tokens=max_tokens, json_schema=schema, think=think,
        )
        try:
            return extract_json(response.text), response
        except StructuredOutputError:
            if repair_attempts <= 0:
                raise

        # One repair pass at temperature 0: hand the model its own bad output
        # and ask for JSON only. Determinism matters more than creativity here.
        repair_prompt = (
            "Your previous response could not be parsed as JSON.\n"
            "Return ONLY the JSON value, with no prose, no explanation and no "
            "code fences.\n\n"
            f"--- your previous response ---\n{response.text[:4000]}\n--- end ---"
        )
        repaired = self.complete(
            repair_prompt, tier=tier, system=system, temperature=0.0,
            max_tokens=max_tokens, json_schema=schema, think=think,
        )
        return extract_json(repaired.text), repaired

    # -- batching ----------------------------------------------------------

    def run_batched(
        self,
        requests_by_tier: Mapping[Tier, list[dict[str, Any]]],
        *,
        keep_alive: str = "20m",
    ) -> dict[Tier, list[LLMResponse | Exception]]:
        """Run grouped work one tier at a time, quick tier first.

        Each value is a list of kwargs for :meth:`complete`. All QUICK work
        completes before the DEEP model is loaded, so the 99 GB model is paged
        in exactly once per cycle instead of once per interleaved call.

        Failures are returned in place rather than raised: one analyst timing
        out should not discard the other five reports. Callers decide whether
        the surviving set is enough to trade on.
        """
        results: dict[Tier, list[LLMResponse | Exception]] = {}

        for tier in (Tier.QUICK, Tier.DEEP):     # order is the whole point
            batch = requests_by_tier.get(tier) or []
            if not batch:
                continue
            tier_results: list[LLMResponse | Exception] = []
            for index, kwargs in enumerate(batch):
                call_kwargs = dict(kwargs)
                prompt = call_kwargs.pop("prompt")
                call_kwargs.pop("tier", None)
                # Release the model after the batch's final call so the next
                # tier does not contend for memory with an idle resident model.
                is_last = index == len(batch) - 1
                call_kwargs.setdefault("keep_alive", "0" if is_last else keep_alive)
                try:
                    tier_results.append(self.complete(prompt, tier=tier, **call_kwargs))
                except BudgetExceeded:
                    # Budget exhaustion is cycle-fatal, not per-call: continuing
                    # would just raise on every remaining item.
                    raise
                except LLMError as exc:
                    log.warning("batched %s call %d failed: %s", tier.value, index, exc)
                    tier_results.append(exc)
            results[tier] = tier_results

        return results

    # -- transport ---------------------------------------------------------

    def _post_with_retry(
        self, url: str, payload: dict[str, Any], *, model: str, tier: Tier
    ) -> LLMResponse:
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 1):
            started = time.monotonic()
            try:
                raw = self._session.post(
                    url, json=payload, timeout=self.config.request_timeout_s
                )
                if raw.status_code in _RETRYABLE_STATUS:
                    raise LLMError(f"ollama returned {raw.status_code}: {raw.text[:200]}")
                raw.raise_for_status()
                body = raw.json()

                # Reasoning models split their output: Ollama puts the chain of
                # thought in ``thinking`` and the answer in ``response``. With
                # constrained decoding the grammar is applied to the thinking
                # channel instead, so ``response`` comes back empty and the real
                # payload is in ``thinking``. Prefer ``response``, fall back
                # rather than surfacing an empty string as a valid completion.
                text = body.get("response") or ""
                if not text.strip() and body.get("thinking"):
                    log.debug(
                        "%s returned an empty response with a populated thinking "
                        "field; falling back to it (pass think=False to avoid this)",
                        model,
                    )
                    text = body["thinking"]

                return LLMResponse(
                    text=text,
                    model=model,
                    tier=tier,
                    prompt_tokens=int(body.get("prompt_eval_count", 0)),
                    completion_tokens=int(body.get("eval_count", 0)),
                    latency_s=time.monotonic() - started,
                    attempts=attempt,
                )
            except (requests.RequestException, LLMError, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                # Exponential backoff. The deep model in particular can take
                # minutes to page in from disk on a cold first call.
                delay = min(2.0 ** attempt, 30.0)
                log.warning(
                    "llm call to %s failed (attempt %d/%d): %s -- retrying in %.1fs",
                    model, attempt, self.config.max_retries, exc, delay,
                )
                time.sleep(delay)

        raise LLMError(
            f"llm call to {model} failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    # -- health ------------------------------------------------------------

    def available_models(self) -> list[str]:
        url = f"{self.config.base_url.rstrip('/')}/api/tags"
        raw = self._session.get(url, timeout=30)
        raw.raise_for_status()
        return [m["name"] for m in raw.json().get("models", [])]

    def health_check(self) -> dict[str, Any]:
        """Verify the server is reachable and both configured tiers exist.

        Called at startup: discovering that the deep model was never pulled is
        much cheaper here than four analyst reports into a live cycle.
        """
        result: dict[str, Any] = {"base_url": self.config.base_url, "ok": False}
        try:
            models = self.available_models()
        except requests.RequestException as exc:
            result["error"] = f"cannot reach ollama: {exc}"
            return result

        result["models"] = models
        missing = [
            m for m in (self.config.quick_model, self.config.deep_model)
            if m not in models
        ]
        if missing:
            result["error"] = f"configured models not pulled on the server: {missing}"
            return result

        result["ok"] = True
        result["quick_model"] = self.config.quick_model
        result["deep_model"] = self.config.deep_model
        return result

    def budget_snapshot(self) -> dict[str, int]:
        return self.budget.snapshot()

    def reset_budget(self) -> None:
        """Start a fresh decision cycle."""
        self.budget.reset()


__all__ = [
    "BudgetExceeded", "CycleBudget", "LLMError", "LLMResponse", "LLMRouter",
    "StructuredOutputError", "Tier", "extract_json",
]
