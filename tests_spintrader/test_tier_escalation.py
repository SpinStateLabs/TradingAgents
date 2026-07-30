"""Tests for adaptive tier escalation (task 18). No network, no LLM.

When a first-pass (quick-tier) panel verdict for an instrument is contested --
the panel split, or only a single lens spoke to it -- the slow loop re-runs just
that instrument on the deep model and replaces the verdict with the deep one.

These tests pin the orchestration with fakes:

* an escalating verdict triggers a deep pass over *exactly* the escalated
  instruments, and non-escalating ones are left untouched;
* the deep votes replace the quick verdict for the escalated instruments;
* all deep work is issued in a single ``adjudicate_all`` batch (the router's
  tier-batching constraint -- the 99 GB model must page in once, not per call);
* the bootstrap voter has no deep tier, so escalation is a graceful no-op;
* the real ``LLMVoter`` routes ``adjudicate_all`` to the deep tier and reuses
  the persona prompt plus the sharper adjudication directive.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from spintrader.agents.panel import PersonaPanel, PersonaVote
from spintrader.agents.personas.roster import default_roster
from spintrader.agents.personas.spec import Horizon, Lens
from spintrader.backtest.runner import backtest_instrument
from spintrader.core.types import Action, AssetClass
from spintrader.llm.router import LLMResponse, Tier
from spintrader.loop.context import build_context
from spintrader.loop.mandate import MandateService
from spintrader.loop.voting import (
    Adjudicator, BootstrapVoter, DEEP_ADJUDICATION_DIRECTIVE, LLMVoter, VoteItem,
)

D = Decimal


def ctx_for(symbol: str):
    """A minimal CRYPTO/INTRADAY context. The fake voter ignores its numbers;
    only asset_class, horizon and instrument key are read by the service."""
    inst = backtest_instrument(symbol, AssetClass.CRYPTO)
    return build_context(inst, [], "1m", Horizon.INTRADAY)


def pv(persona_key: str, action: Action, conf, lenses=()) -> PersonaVote:
    return PersonaVote(
        persona_key=persona_key, action=action, confidence=D(str(conf)),
        rationale="test", lenses=lenses,
    )


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeVoter:
    """Returns canned votes per instrument key; records adjudication calls.

    Implements ``adjudicate_all`` so it is a structural :class:`Adjudicator` --
    the capability the service checks for before escalating.
    """

    def __init__(self, quick_votes, deep_votes=None):
        self.quick_votes = quick_votes            # {key: [PersonaVote, ...]}
        self.deep_votes = deep_votes or {}
        self.vote_all_keys = None
        self.adjudicate_calls = 0
        self.adjudicated_keys = None              # None until a deep pass runs

    def vote_all(self, items):
        self.vote_all_keys = _distinct_keys(items)
        return {k: list(self.quick_votes.get(k, [])) for k in self.vote_all_keys}

    def adjudicate_all(self, items):
        self.adjudicate_calls += 1
        self.adjudicated_keys = _distinct_keys(items)
        return {k: list(self.deep_votes.get(k, [])) for k in self.adjudicated_keys}


class QuickOnlyVoter:
    """A voter with no deep tier at all -- not an Adjudicator."""

    def __init__(self, quick_votes):
        self.quick_votes = quick_votes

    def vote_all(self, items):
        return {k: list(self.quick_votes.get(k, [])) for k in _distinct_keys(items)}


def _distinct_keys(items):
    keys = []
    for item in items:
        if item.instrument_key not in keys:
            keys.append(item.instrument_key)
    return keys


class TierRecordingRouter:
    """Records each run_batched call and replays canned text per tier."""

    def __init__(self, replies_by_tier):
        self.replies_by_tier = replies_by_tier
        self.batches = []

    def run_batched(self, requests_by_tier, keep_alive="20m"):
        self.batches.append(requests_by_tier)
        out = {}
        for tier, reqs in requests_by_tier.items():
            replies = self.replies_by_tier.get(tier, [])
            tier_out = []
            for i in range(len(reqs)):
                text = replies[i] if i < len(replies) else '{"action":"hold","confidence":0}'
                tier_out.append(LLMResponse(text=text, model="stub", tier=tier))
            out[tier] = tier_out
        return out


# Canned quick/deep vote sets, chosen so the panel's escalate flag is
# deterministic (see module docstring for the aggregation rules).

def escalating_quick():
    # A lone single-lens BUY -> lens_diversity 1 -> panel sets escalate=True.
    return [pv("simons", Action.BUY, "0.7", lenses=(Lens.STATISTICAL,))]


def clean_quick():
    # Two BUYs agreeing across two lenses -> escalate=False.
    return [
        pv("simons", Action.BUY, "0.6", lenses=(Lens.STATISTICAL,)),
        pv("policy_headline", Action.BUY, "0.6"),      # lenses filled from spec
    ]


def deep_sell():
    # A diverse, decisive SELL: flips the direction and resolves the split.
    return [
        pv("simons", Action.SELL, "0.9", lenses=(Lens.STATISTICAL,)),
        pv("thorp", Action.SELL, "0.9"),
        pv("policy_headline", Action.SELL, "0.8"),
    ]


# --------------------------------------------------------------------------
# Sanity: the canned quick votes escalate as intended
# --------------------------------------------------------------------------

class PanelEscalationSanityTests(unittest.TestCase):
    def setUp(self):
        self.panel = PersonaPanel(default_roster())

    def test_single_lens_quick_verdict_escalates(self):
        v = self.panel.aggregate(escalating_quick(), AssetClass.CRYPTO, Horizon.INTRADAY)
        self.assertTrue(v.escalate)
        self.assertEqual(v.action, Action.BUY)

    def test_two_lens_agreement_does_not_escalate(self):
        v = self.panel.aggregate(clean_quick(), AssetClass.CRYPTO, Horizon.INTRADAY)
        self.assertFalse(v.escalate)
        self.assertEqual(v.action, Action.BUY)

    def test_deep_votes_resolve_to_a_sell(self):
        v = self.panel.aggregate(deep_sell(), AssetClass.CRYPTO, Horizon.INTRADAY)
        self.assertEqual(v.action, Action.SELL)
        self.assertFalse(v.escalate)


# --------------------------------------------------------------------------
# Adjudicator protocol
# --------------------------------------------------------------------------

class AdjudicatorProtocolTests(unittest.TestCase):
    def test_llmvoter_is_an_adjudicator(self):
        self.assertIsInstance(LLMVoter(TierRecordingRouter({})), Adjudicator)

    def test_bootstrap_voter_is_not_an_adjudicator(self):
        # The bootstrap voter has no deep model, so it must not advertise the
        # capability -- that is what makes escalation a graceful no-op.
        self.assertNotIsInstance(BootstrapVoter(), Adjudicator)

    def test_fake_voter_with_the_method_is_an_adjudicator(self):
        self.assertIsInstance(FakeVoter({}), Adjudicator)

    def test_quick_only_voter_is_not_an_adjudicator(self):
        self.assertNotIsInstance(QuickOnlyVoter({}), Adjudicator)


# --------------------------------------------------------------------------
# Escalation orchestration in MandateService
# --------------------------------------------------------------------------

class EscalationOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.a = ctx_for("AAA")          # will escalate
        self.b = ctx_for("BBB")          # will not
        self.contexts = {self.a.instrument_key: self.a, self.b.instrument_key: self.b}

    def _service(self, voter, **kw):
        return MandateService(default_roster(), voter=voter, **kw)

    def test_deep_pass_runs_over_exactly_the_escalated_instrument(self):
        voter = FakeVoter(
            quick_votes={self.a.instrument_key: escalating_quick(),
                         self.b.instrument_key: clean_quick()},
            deep_votes={self.a.instrument_key: deep_sell()},
        )
        delib = self._service(voter).deliberate(self.contexts)

        # (a) exactly the escalated instrument went to the deep pass.
        self.assertEqual(voter.adjudicate_calls, 1)
        self.assertEqual(set(voter.adjudicated_keys), {self.a.instrument_key})
        self.assertEqual(delib.escalated, frozenset({self.a.instrument_key}))

    def test_non_escalating_instrument_is_untouched(self):
        voter = FakeVoter(
            quick_votes={self.a.instrument_key: escalating_quick(),
                         self.b.instrument_key: clean_quick()},
            deep_votes={self.a.instrument_key: deep_sell()},
        )
        delib = self._service(voter).deliberate(self.contexts)

        # (b) the clean instrument never entered the deep pass and kept its BUY.
        self.assertNotIn(self.b.instrument_key, voter.adjudicated_keys)
        self.assertNotIn(self.b.instrument_key, delib.escalated)
        self.assertEqual(delib.verdicts[self.b.instrument_key].action, Action.BUY)

    def test_deep_votes_replace_the_quick_verdict(self):
        voter = FakeVoter(
            quick_votes={self.a.instrument_key: escalating_quick(),
                         self.b.instrument_key: clean_quick()},
            deep_votes={self.a.instrument_key: deep_sell()},
        )
        delib = self._service(voter).deliberate(self.contexts)

        # (c) the escalated instrument's verdict is now the deep SELL, not the
        # quick BUY, and the deep pass resolved the split (no longer escalating).
        verdict = delib.verdicts[self.a.instrument_key]
        self.assertEqual(verdict.action, Action.SELL)
        self.assertFalse(verdict.escalate)
        # The mandate reflects the replaced verdict: permitted, short-biased.
        self.assertIn(self.a.instrument_key, delib.mandate.permitted)
        self.assertLess(delib.mandate.bias_for(self.a.instrument_key), 0)

    def test_all_escalated_instruments_share_one_deep_batch(self):
        # Two escalating instruments must be adjudicated in a SINGLE deep pass,
        # not one pass each -- the tier-batching constraint.
        a2 = ctx_for("AAA")
        c = ctx_for("CCC")
        contexts = {a2.instrument_key: a2, c.instrument_key: c}
        voter = FakeVoter(
            quick_votes={a2.instrument_key: escalating_quick(),
                         c.instrument_key: escalating_quick()},
            deep_votes={a2.instrument_key: deep_sell(),
                        c.instrument_key: deep_sell()},
        )
        delib = self._service(voter).deliberate(contexts)
        self.assertEqual(voter.adjudicate_calls, 1)
        self.assertEqual(set(voter.adjudicated_keys),
                         {a2.instrument_key, c.instrument_key})
        self.assertEqual(delib.escalated,
                         frozenset({a2.instrument_key, c.instrument_key}))

    def test_disabled_escalation_never_adjudicates(self):
        voter = FakeVoter(
            quick_votes={self.a.instrument_key: escalating_quick(),
                         self.b.instrument_key: clean_quick()},
            deep_votes={self.a.instrument_key: deep_sell()},
        )
        delib = self._service(voter, escalate=False).deliberate(self.contexts)
        self.assertEqual(voter.adjudicate_calls, 0)
        self.assertEqual(delib.escalated, frozenset())
        # Verdict stays the quick BUY.
        self.assertEqual(delib.verdicts[self.a.instrument_key].action, Action.BUY)

    def test_empty_deep_response_keeps_the_quick_verdict(self):
        # If the deep pass returns nothing for an instrument, the quick verdict
        # is retained rather than blanked -- and it is not reported as escalated.
        voter = FakeVoter(
            quick_votes={self.a.instrument_key: escalating_quick(),
                         self.b.instrument_key: clean_quick()},
            deep_votes={},                     # deep model produced no votes
        )
        delib = self._service(voter).deliberate(self.contexts)
        self.assertEqual(voter.adjudicate_calls, 1)
        self.assertEqual(delib.escalated, frozenset())
        self.assertEqual(delib.verdicts[self.a.instrument_key].action, Action.BUY)

    def test_no_escalation_means_no_deep_pass(self):
        voter = FakeVoter(
            quick_votes={self.a.instrument_key: clean_quick(),
                         self.b.instrument_key: clean_quick()},
            deep_votes={self.a.instrument_key: deep_sell()},
        )
        delib = self._service(voter).deliberate(self.contexts)
        self.assertEqual(voter.adjudicate_calls, 0)
        self.assertEqual(delib.escalated, frozenset())


# --------------------------------------------------------------------------
# Bootstrap voter: escalation must degrade gracefully
# --------------------------------------------------------------------------

class BootstrapEscalationTests(unittest.TestCase):
    def test_bootstrap_voter_never_breaks_on_an_escalating_verdict(self):
        # The bootstrap voter emits a single-lens vote, so its verdicts escalate
        # by construction. With no deep tier, deliberate must still complete,
        # produce a mandate, and simply not adjudicate.
        from tests_spintrader.test_decision_loop import uptrend

        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        ctx = build_context(inst, uptrend(60, step=0.3), "1m", Horizon.INTRADAY,
                            fast_window=3, slow_window=10)
        service = MandateService(default_roster(), voter=BootstrapVoter())

        delib = service.deliberate({inst.key: ctx})
        self.assertIn(inst.key, delib.verdicts)
        self.assertTrue(delib.verdicts[inst.key].escalate)   # bootstrap escalates
        self.assertEqual(delib.escalated, frozenset())        # but nothing adjudicated
        self.assertIsNotNone(delib.mandate)


# --------------------------------------------------------------------------
# LLMVoter deep-adjudication entry point (stubbed router)
# --------------------------------------------------------------------------

class LLMVoterAdjudicationTests(unittest.TestCase):
    def _items(self, n=2):
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        ctx = build_context(inst, [], "1m", Horizon.INTRADAY)
        specs = default_roster().applicable(AssetClass.CRYPTO, Horizon.INTRADAY)[:n]
        return [VoteItem(s, ctx) for s in specs], inst.key, specs

    def test_adjudicate_runs_on_the_deep_tier_in_one_batch(self):
        items, key, _ = self._items(2)
        router = TierRecordingRouter({Tier.DEEP: [
            '{"action":"sell","confidence":0.9,"rationale":"deep"}',
            '{"action":"sell","confidence":0.8,"rationale":"deep"}',
        ]})
        votes = LLMVoter(router).adjudicate_all(items)[key]

        self.assertEqual(len(router.batches), 1)              # single batched pass
        self.assertIn(Tier.DEEP, router.batches[0])           # ... on the deep tier
        self.assertNotIn(Tier.QUICK, router.batches[0])
        self.assertEqual(len(router.batches[0][Tier.DEEP]), 2)
        self.assertEqual(votes[0].action, Action.SELL)
        self.assertEqual(votes[0].confidence, D("0.9"))

    def test_adjudicate_reuses_persona_prompt_plus_directive(self):
        items, key, specs = self._items(1)
        router = TierRecordingRouter({Tier.DEEP: [
            '{"action":"buy","confidence":0.7,"rationale":"x"}',
        ]})
        LLMVoter(router).adjudicate_all(items)
        system = router.batches[0][Tier.DEEP][0]["system"]
        # The persona's own methodology is still present ...
        self.assertIn(specs[0].attribution, system)
        # ... and the sharper re-adjudication directive is appended.
        self.assertIn(DEEP_ADJUDICATION_DIRECTIVE.strip(), system)

    def test_vote_all_still_runs_on_the_quick_tier(self):
        # Regression: the first pass is unchanged and does not carry the directive.
        items, key, _ = self._items(1)
        router = TierRecordingRouter({Tier.QUICK: [
            '{"action":"buy","confidence":0.6,"rationale":"x"}',
        ]})
        LLMVoter(router).vote_all(items)
        self.assertIn(Tier.QUICK, router.batches[0])
        self.assertNotIn(Tier.DEEP, router.batches[0])
        self.assertNotIn(DEEP_ADJUDICATION_DIRECTIVE.strip(),
                         router.batches[0][Tier.QUICK][0]["system"])

    def test_custom_adjudication_directive_is_used(self):
        items, key, _ = self._items(1)
        router = TierRecordingRouter({Tier.DEEP: [
            '{"action":"hold","confidence":0}',
        ]})
        LLMVoter(router, adjudication_directive="\nCUSTOM DIRECTIVE").adjudicate_all(items)
        self.assertIn("CUSTOM DIRECTIVE", router.batches[0][Tier.DEEP][0]["system"])


if __name__ == "__main__":
    unittest.main()
