"""Tests for the tiered LLM router.

All network calls are mocked. The GB10 integration check lives in
``ops/check_gb10.py`` so that the unit suite stays runnable offline.

``extract_json`` gets disproportionate coverage because it is the layer that
absorbs local-model misbehaviour. Every case below is a real failure shape
observed from Ollama-served models: reasoning blocks, code fences, apologetic
preambles, and braces inside string values.
"""

from __future__ import annotations

import json
import unittest
from unittest import mock

import requests

from spintrader.core.config import LLMConfig
from spintrader.llm.router import (
    BudgetExceeded, CycleBudget, LLMError, LLMResponse, LLMRouter,
    StructuredOutputError, Tier, extract_json,
)


def ollama_body(text: str, *, prompt_tokens: int = 10, eval_tokens: int = 20) -> dict:
    return {
        "response": text,
        "prompt_eval_count": prompt_tokens,
        "eval_count": eval_tokens,
        "done": True,
    }


class FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def make_router(responses: list, **cfg_overrides) -> tuple[LLMRouter, mock.Mock]:
    """Router wired to a session that returns ``responses`` in order.

    An entry may be a ``FakeResponse`` or an exception instance to raise.
    """
    session = mock.Mock(spec=requests.Session)

    def side_effect(*_args, **_kwargs):
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    session.post.side_effect = side_effect
    config = LLMConfig(
        base_url="http://gb10.test:11434",
        quick_model="quick-m",
        deep_model="deep-m",
        max_retries=cfg_overrides.pop("max_retries", 3),
        **cfg_overrides,
    )
    return LLMRouter(config=config, session=session), session


class ExtractJsonTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"action": "buy"}'), {"action": "buy"})

    def test_plain_array(self):
        self.assertEqual(extract_json("[1, 2, 3]"), [1, 2, 3])

    def test_code_fence(self):
        raw = 'Here is my analysis:\n```json\n{"action": "sell", "conf": 0.7}\n```\nHope that helps!'
        self.assertEqual(extract_json(raw), {"action": "sell", "conf": 0.7})

    def test_bare_fence_without_language(self):
        self.assertEqual(extract_json('```\n{"a": 1}\n```'), {"a": 1})

    def test_strips_reasoning_block(self):
        # qwen3 emits <think> blocks even when told to return JSON only, and
        # those blocks routinely contain braces.
        raw = '<think>Let me consider {this} and {that} carefully.</think>{"action": "hold"}'
        self.assertEqual(extract_json(raw), {"action": "hold"})

    def test_multiline_reasoning_block(self):
        raw = '<think>\nline one {\nline two }\n</think>\n{"x": 1}'
        self.assertEqual(extract_json(raw), {"x": 1})

    def test_preamble_prose(self):
        raw = 'Sure! Based on the data, here is the result: {"action": "buy", "size": 0.1}'
        self.assertEqual(extract_json(raw), {"action": "buy", "size": 0.1})

    def test_brace_inside_string_value_does_not_truncate(self):
        # Naive "find the first }" extraction fails on exactly this.
        payload = {"rationale": "the setup looks like {a bull flag} to me", "action": "buy"}
        raw = f"Analysis follows.\n{json.dumps(payload)}\nDone."
        self.assertEqual(extract_json(raw), payload)

    def test_escaped_quote_inside_string(self):
        payload = {"rationale": 'he said "buy the dip" repeatedly'}
        self.assertEqual(extract_json(json.dumps(payload)), payload)

    def test_nested_structures(self):
        payload = {"votes": [{"agent": "burry", "action": "sell"}], "meta": {"n": 1}}
        self.assertEqual(extract_json(f"result: {json.dumps(payload)}"), payload)

    def test_trailing_text_after_object(self):
        self.assertEqual(extract_json('{"a": 1} -- that is my answer'), {"a": 1})

    def test_empty_response_raises(self):
        for raw in ("", "   ", "\n\n"):
            with self.subTest(raw=repr(raw)):
                with self.assertRaises(StructuredOutputError):
                    extract_json(raw)

    def test_prose_only_raises(self):
        with self.assertRaises(StructuredOutputError):
            extract_json("I cannot answer that question.")

    def test_error_message_includes_a_preview(self):
        with self.assertRaises(StructuredOutputError) as ctx:
            extract_json("totally unparseable output")
        self.assertIn("unparseable", str(ctx.exception))

    def test_unterminated_object_raises(self):
        with self.assertRaises(StructuredOutputError):
            extract_json('{"a": 1, "b":')


class BudgetTests(unittest.TestCase):
    def test_reserve_increments(self):
        b = CycleBudget(max_calls=3, max_tokens=1000)
        b.reserve()
        b.reserve()
        self.assertEqual(b.snapshot()["calls"], 2)

    def test_call_limit_enforced(self):
        b = CycleBudget(max_calls=2, max_tokens=1000)
        b.reserve()
        b.reserve()
        with self.assertRaises(BudgetExceeded) as ctx:
            b.reserve()
        self.assertIn("call budget", str(ctx.exception))

    def test_token_limit_enforced(self):
        b = CycleBudget(max_calls=100, max_tokens=50)
        b.record(LLMResponse("x", "m", Tier.QUICK, prompt_tokens=30, completion_tokens=25))
        with self.assertRaises(BudgetExceeded) as ctx:
            b.reserve()
        self.assertIn("token budget", str(ctx.exception))

    def test_reset_clears_counters(self):
        b = CycleBudget(max_calls=2, max_tokens=100)
        b.reserve()
        b.record(LLMResponse("x", "m", Tier.QUICK, prompt_tokens=10, completion_tokens=10))
        b.reset()
        snap = b.snapshot()
        self.assertEqual((snap["calls"], snap["prompt_tokens"]), (0, 0))

    def test_thread_safe_under_contention(self):
        import threading
        b = CycleBudget(max_calls=100, max_tokens=10**9)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(10):
                    b.reserve()
            except Exception as exc:      # noqa: BLE001 - recorded and asserted below
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(b.snapshot()["calls"], 100)   # exactly, no lost updates


class RoutingTests(unittest.TestCase):
    def test_tier_selects_model(self):
        router, _ = make_router([])
        self.assertEqual(router.model_for(Tier.QUICK), "quick-m")
        self.assertEqual(router.model_for(Tier.DEEP), "deep-m")

    def test_quick_tier_is_the_default(self):
        router, session = make_router([FakeResponse(ollama_body("ok"))])
        router.complete("hi")
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "quick-m")

    def test_deep_tier_requested_explicitly(self):
        router, session = make_router([FakeResponse(ollama_body("ok"))])
        router.complete("hi", tier=Tier.DEEP)
        self.assertEqual(session.post.call_args.kwargs["json"]["model"], "deep-m")

    def test_response_carries_token_accounting(self):
        router, _ = make_router([FakeResponse(ollama_body("hello", prompt_tokens=7, eval_tokens=13))])
        r = router.complete("hi")
        self.assertEqual((r.prompt_tokens, r.completion_tokens, r.total_tokens), (7, 13, 20))
        self.assertGreaterEqual(r.latency_s, 0.0)

    def test_budget_is_charged(self):
        router, _ = make_router([FakeResponse(ollama_body("x", prompt_tokens=5, eval_tokens=6))])
        router.complete("hi")
        snap = router.budget_snapshot()
        self.assertEqual((snap["calls"], snap["prompt_tokens"], snap["completion_tokens"]), (1, 5, 6))

    def test_system_prompt_forwarded(self):
        router, session = make_router([FakeResponse(ollama_body("x"))])
        router.complete("hi", system="you are a trader")
        self.assertEqual(session.post.call_args.kwargs["json"]["system"], "you are a trader")

    def test_schema_becomes_constrained_decoding_format(self):
        router, session = make_router([FakeResponse(ollama_body("{}"))])
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        router.complete("hi", json_schema=schema)
        self.assertEqual(session.post.call_args.kwargs["json"]["format"], schema)

    def test_keep_alive_forwarded(self):
        router, session = make_router([FakeResponse(ollama_body("x"))])
        router.complete("hi", keep_alive="0")
        self.assertEqual(session.post.call_args.kwargs["json"]["keep_alive"], "0")

    def test_options_carry_sampling_params(self):
        router, session = make_router([FakeResponse(ollama_body("x"))])
        router.complete("hi", temperature=0.0, max_tokens=256, stop=["END"])
        options = session.post.call_args.kwargs["json"]["options"]
        self.assertEqual(options["temperature"], 0.0)
        self.assertEqual(options["num_predict"], 256)
        self.assertEqual(options["stop"], ["END"])


class ReasoningModelTests(unittest.TestCase):
    """Regression tests for the qwen3 thinking-channel behaviour.

    Observed live on 2026-07-29: with a JSON schema attached, qwen3:30b-a3b
    returns ``response: ""`` and puts the schema-conforming JSON in
    ``thinking`` instead. Left unhandled, every quick-tier structured call
    fails with "model returned an empty response".
    """

    def test_falls_back_to_thinking_when_response_is_empty(self):
        body = ollama_body("")
        body["thinking"] = '{"action": "hold", "confidence": 0.7}'
        router, _ = make_router([FakeResponse(body)])
        self.assertEqual(router.complete("hi").text, '{"action": "hold", "confidence": 0.7}')

    def test_response_wins_when_both_are_populated(self):
        body = ollama_body('{"action": "buy"}')
        body["thinking"] = "let me consider the options"
        router, _ = make_router([FakeResponse(body)])
        self.assertEqual(router.complete("hi").text, '{"action": "buy"}')

    def test_whitespace_only_response_still_falls_back(self):
        body = ollama_body("   \n  ")
        body["thinking"] = '{"a": 1}'
        router, _ = make_router([FakeResponse(body)])
        self.assertEqual(router.complete("hi").text, '{"a": 1}')

    def test_empty_both_stays_empty(self):
        router, _ = make_router([FakeResponse(ollama_body(""))])
        self.assertEqual(router.complete("hi").text, "")

    def test_think_flag_forwarded(self):
        router, session = make_router([FakeResponse(ollama_body("x"))])
        router.complete("hi", think=False)
        self.assertIs(session.post.call_args.kwargs["json"]["think"], False)

    def test_think_omitted_when_unset(self):
        router, session = make_router([FakeResponse(ollama_body("x"))])
        router.complete("hi")
        self.assertNotIn("think", session.post.call_args.kwargs["json"])

    def test_complete_json_disables_thinking_by_default(self):
        router, session = make_router([FakeResponse(ollama_body('{"a": 1}'))])
        router.complete_json("decide")
        self.assertIs(session.post.call_args.kwargs["json"]["think"], False)

    def test_complete_json_can_restore_model_default(self):
        router, session = make_router([FakeResponse(ollama_body('{"a": 1}'))])
        router.complete_json("decide", think=None)
        self.assertNotIn("think", session.post.call_args.kwargs["json"])

    def test_end_to_end_thinking_channel_json_is_parsed(self):
        # The exact live failure: schema attached, JSON lands in `thinking`.
        body = ollama_body("")
        body["thinking"] = '{\n  "action": "hold",\n  "confidence": 0.7\n}'
        router, _ = make_router([FakeResponse(body)])
        value, _ = router.complete_json("decide", think=None)
        self.assertEqual(value, {"action": "hold", "confidence": 0.7})


class RetryTests(unittest.TestCase):
    @mock.patch("spintrader.llm.router.time.sleep")
    def test_retries_then_succeeds(self, sleep):
        router, _ = make_router([
            requests.ConnectionError("connection reset"),
            FakeResponse(ollama_body("recovered")),
        ])
        r = router.complete("hi")
        self.assertEqual(r.text, "recovered")
        self.assertEqual(r.attempts, 2)
        sleep.assert_called_once()

    @mock.patch("spintrader.llm.router.time.sleep")
    def test_retries_on_server_error_status(self, _sleep):
        router, _ = make_router([
            FakeResponse({}, status=503),
            FakeResponse(ollama_body("ok")),
        ])
        self.assertEqual(router.complete("hi").text, "ok")

    @mock.patch("spintrader.llm.router.time.sleep")
    def test_gives_up_after_max_retries(self, _sleep):
        router, _ = make_router(
            [requests.ConnectionError("down")] * 3, max_retries=3,
        )
        with self.assertRaises(LLMError) as ctx:
            router.complete("hi")
        self.assertIn("after 3 attempts", str(ctx.exception))

    @mock.patch("spintrader.llm.router.time.sleep")
    def test_backoff_is_bounded(self, sleep):
        router, _ = make_router([requests.ConnectionError("down")] * 6, max_retries=6)
        with self.assertRaises(LLMError):
            router.complete("hi")
        # Unbounded exponential backoff would stall a cycle for hours.
        self.assertTrue(all(call.args[0] <= 30.0 for call in sleep.call_args_list))


class CompleteJsonTests(unittest.TestCase):
    def test_parses_first_attempt(self):
        router, session = make_router([FakeResponse(ollama_body('{"action": "buy"}'))])
        value, response = router.complete_json("decide")
        self.assertEqual(value, {"action": "buy"})
        self.assertEqual(response.total_tokens, 30)
        self.assertEqual(session.post.call_count, 1)

    def test_repairs_unparseable_output(self):
        router, session = make_router([
            FakeResponse(ollama_body("I think you should buy.")),
            FakeResponse(ollama_body('{"action": "buy"}')),
        ])
        value, _ = router.complete_json("decide")
        self.assertEqual(value, {"action": "buy"})
        self.assertEqual(session.post.call_count, 2)
        # The repair pass must be deterministic.
        self.assertEqual(session.post.call_args.kwargs["json"]["options"]["temperature"], 0.0)

    def test_gives_up_when_repair_also_fails(self):
        router, _ = make_router([
            FakeResponse(ollama_body("nope")),
            FakeResponse(ollama_body("still nope")),
        ])
        with self.assertRaises(StructuredOutputError):
            router.complete_json("decide")

    def test_repair_can_be_disabled(self):
        router, session = make_router([FakeResponse(ollama_body("nope"))])
        with self.assertRaises(StructuredOutputError):
            router.complete_json("decide", repair_attempts=0)
        self.assertEqual(session.post.call_count, 1)


class BatchingTests(unittest.TestCase):
    def test_quick_tier_runs_entirely_before_deep(self):
        # The point of batching: the 99 GB deep model must be paged in once,
        # after all quick work is done.
        router, session = make_router([FakeResponse(ollama_body("r")) for _ in range(5)])
        router.run_batched({
            Tier.DEEP: [{"prompt": "d1"}, {"prompt": "d2"}],
            Tier.QUICK: [{"prompt": "q1"}, {"prompt": "q2"}, {"prompt": "q3"}],
        })
        models = [c.kwargs["json"]["model"] for c in session.post.call_args_list]
        self.assertEqual(models, ["quick-m"] * 3 + ["deep-m"] * 2)

    def test_last_call_in_each_batch_releases_the_model(self):
        router, session = make_router([FakeResponse(ollama_body("r")) for _ in range(3)])
        router.run_batched({Tier.QUICK: [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]})
        keep_alives = [c.kwargs["json"]["keep_alive"] for c in session.post.call_args_list]
        self.assertEqual(keep_alives[-1], "0")
        self.assertTrue(all(k != "0" for k in keep_alives[:-1]))

    @mock.patch("spintrader.llm.router.time.sleep")
    def test_one_failure_does_not_discard_the_batch(self, _sleep):
        router, _ = make_router([
            FakeResponse(ollama_body("first")),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            FakeResponse(ollama_body("third")),
        ], max_retries=3)
        results = router.run_batched({
            Tier.QUICK: [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}],
        })
        quick = results[Tier.QUICK]
        self.assertEqual(quick[0].text, "first")
        self.assertIsInstance(quick[1], LLMError)
        self.assertEqual(quick[2].text, "third")

    def test_budget_exhaustion_aborts_the_cycle(self):
        # Unlike a transport error, budget exhaustion is not recoverable by
        # continuing -- every remaining call would raise too.
        router, _ = make_router([FakeResponse(ollama_body("r"))] * 3)
        router.budget = CycleBudget(max_calls=1, max_tokens=10**9)
        with self.assertRaises(BudgetExceeded):
            router.run_batched({Tier.QUICK: [{"prompt": "a"}, {"prompt": "b"}]})

    def test_empty_tiers_are_skipped(self):
        router, session = make_router([FakeResponse(ollama_body("r"))])
        results = router.run_batched({Tier.QUICK: [{"prompt": "a"}], Tier.DEEP: []})
        self.assertNotIn(Tier.DEEP, results)
        self.assertEqual(session.post.call_count, 1)


class HealthCheckTests(unittest.TestCase):
    def _router_with_tags(self, models: list[str]) -> LLMRouter:
        session = mock.Mock(spec=requests.Session)
        session.get.return_value = FakeResponse({"models": [{"name": m} for m in models]})
        config = LLMConfig(base_url="http://gb10.test:11434",
                           quick_model="quick-m", deep_model="deep-m")
        return LLMRouter(config=config, session=session)

    def test_ok_when_both_tiers_present(self):
        result = self._router_with_tags(["quick-m", "deep-m", "other"]).health_check()
        self.assertTrue(result["ok"])

    def test_reports_missing_model(self):
        # Catching this at startup beats discovering it four analyst reports in.
        result = self._router_with_tags(["quick-m"]).health_check()
        self.assertFalse(result["ok"])
        self.assertIn("deep-m", result["error"])

    def test_reports_unreachable_server(self):
        session = mock.Mock(spec=requests.Session)
        session.get.side_effect = requests.ConnectionError("no route to host")
        router = LLMRouter(config=LLMConfig(base_url="http://gb10.test:11434"), session=session)
        result = router.health_check()
        self.assertFalse(result["ok"])
        self.assertIn("cannot reach ollama", result["error"])


if __name__ == "__main__":
    unittest.main()
