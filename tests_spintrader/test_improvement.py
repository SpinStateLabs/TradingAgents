"""Tests for the self-improvement cycle (task 19). No network, no LLM, no DB.

The invariant under test, above all: every candidate that is *evaluated* is
*counted* as a trial, and candidates already evaluated are skipped rather than
re-counted. The promotion gate's whole defence against noise rests on that count
being honest, so the orchestration is tested with a fake backtester that lets
each candidate's quality be dialled precisely.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from decimal import Decimal

from spintrader.core.types import AssetClass
from spintrader.loop.improvement import ImprovementCycle
from spintrader.loop.promotion import PromotionGate, TrialLedger
from spintrader.research.factory import BASE_CONFIG, CandidateConfig, CandidateFactory
from spintrader.research.memory import ResearchMemory, TrialRecord

D = Decimal


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

class CandidateFactoryTests(unittest.TestCase):
    def test_all_configs_are_valid(self):
        for cfg in CandidateFactory().all_configs():
            self.assertGreater(int(cfg.params["slow_window"]), int(cfg.params["fast_window"]))
            self.assertGreaterEqual(int(cfg.params["fast_window"]), 2)

    def test_enumeration_is_deterministic(self):
        a = [c.key for c in CandidateFactory().all_configs()]
        b = [c.key for c in CandidateFactory().all_configs()]
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(set(a)))          # keys are unique

    def test_generate_skips_avoided_keys(self):
        factory = CandidateFactory()
        allc = factory.all_configs()
        avoid = {allc[0].key, allc[1].key}
        fresh = factory.generate(avoid=avoid)
        self.assertTrue(all(c.key not in avoid for c in fresh))
        self.assertEqual(len(fresh), len(allc) - 2)

    def test_generate_respects_n(self):
        self.assertEqual(len(CandidateFactory().generate(n=3)), 3)

    def test_invalid_grid_points_are_filtered(self):
        # slow must exceed fast; a grid where they can be equal drops those.
        factory = CandidateFactory(grid={"fast_window": (20, 40), "slow_window": (20, 60)})
        for cfg in factory.all_configs():
            self.assertGreater(int(cfg.params["slow_window"]), int(cfg.params["fast_window"]))

    def test_config_key_is_stable_and_typed(self):
        cfg = CandidateConfig({"fast_window": 20, "slow_window": 100})
        self.assertEqual(cfg.key, CandidateConfig({"slow_window": 100, "fast_window": 20}).key)
        self.assertEqual(cfg.to_dict()["fast_window"], 20)   # int preserved


# --------------------------------------------------------------------------
# Research memory
# --------------------------------------------------------------------------

class FakeCacheStore:
    def __init__(self):
        self.blobs = {}

    def cache_put(self, cache_key, source, payload, **kw):
        self.blobs[cache_key] = payload

    def cache_get(self, cache_key):
        return self.blobs.get(cache_key)


def rec(key, *, promoted=False, sharpe=1.0, obj="crypto"):
    return TrialRecord(objective=obj, config_key=key, config={"k": key},
                       promoted=promoted, deflated_sharpe=0.9, n_trials=1,
                       sharpe=sharpe, rejections=[] if promoted else ["not_significant"])


class ResearchMemoryTests(unittest.TestCase):
    def test_seen_and_records(self):
        m = ResearchMemory()
        m.record(rec("a"))
        self.assertTrue(m.seen("crypto", "a"))
        self.assertFalse(m.seen("crypto", "b"))
        self.assertEqual(len(m.records("crypto")), 1)

    def test_best_is_highest_sharpe_promoted(self):
        m = ResearchMemory()
        m.record(rec("a", promoted=True, sharpe=1.0))
        m.record(rec("b", promoted=True, sharpe=2.0))
        m.record(rec("c", promoted=False, sharpe=9.0))   # not promoted, ignored
        self.assertEqual(m.best("crypto").config_key, "b")

    def test_best_none_when_nothing_promoted(self):
        m = ResearchMemory()
        m.record(rec("a", promoted=False))
        self.assertIsNone(m.best("crypto"))

    def test_persistence_round_trips(self):
        store = FakeCacheStore()
        m = ResearchMemory(store=store)
        m.record(rec("a", promoted=True, sharpe=1.5))
        m.save("crypto")

        m2 = ResearchMemory(store=store)
        loaded = m2.load("crypto")
        self.assertEqual(loaded, 1)
        self.assertTrue(m2.seen("crypto", "a"))
        self.assertEqual(m2.best("crypto").sharpe, 1.5)

    def test_rejection_profile(self):
        m = ResearchMemory()
        m.record(rec("a"))              # not_significant
        m.record(rec("b"))
        self.assertEqual(m.rejection_profile("crypto"), {"not_significant": 2})


# --------------------------------------------------------------------------
# Fake backtester for the cycle
# --------------------------------------------------------------------------

@dataclass
class FakeCard:
    sharpe: float
    n_observations: int = 600
    skew: float = 0.0
    excess_kurtosis: float = 0.0
    periods_per_year: int = 365
    cost_drag: float = 0.1                 # edge is 10x fees -> passes cost gate
    max_drawdown: float = -0.05            # within the MODERATE 10% limit
    sharpe_stderr: float = 0.1
    total_return: float = 0.2


@dataclass
class FakeFold:
    scorecard: FakeCard


@dataclass
class FakeWF:
    combined: FakeCard | None
    n_folds: int = 4
    folds: list = field(default_factory=list)


@dataclass
class FakeResult:
    scorecard: FakeCard | None


@dataclass
class FakeRun:
    walk_forward: FakeWF | None
    result: FakeResult


def fake_backtester(sharpe_by_key, none_keys=()):
    """Return a backtest_fn whose result quality is keyed on the config."""
    def fn(strategy_cls, symbol, bars, *, asset_class, aggression,
           starting_cash, walk_forward, n_trials, strategy_kwargs):
        key = CandidateConfig(dict(strategy_kwargs)).key
        if key in none_keys:
            return FakeRun(walk_forward=None, result=FakeResult(None))
        s = sharpe_by_key.get(key, 0.2)     # default: mediocre, will be rejected
        card = FakeCard(sharpe=s)
        wf = FakeWF(combined=card, n_folds=4, folds=[FakeFold(FakeCard(sharpe=s))] * 4)
        return FakeRun(walk_forward=wf, result=FakeResult(card))
    return fn


def small_factory():
    # Vary one axis -> three distinct candidates, enough to test selection.
    return CandidateFactory(grid={"trail_pct": ("0.05", "0.08", "0.12")})


def keys_for(factory):
    return [c.key for c in factory.all_configs()]


# --------------------------------------------------------------------------
# Improvement cycle
# --------------------------------------------------------------------------

class ImprovementCycleTests(unittest.TestCase):
    def _cycle(self, sharpe_by_key, none_keys=(), memory=None, gate=None):
        factory = small_factory()
        gate = gate or PromotionGate(ledger=TrialLedger())
        memory = memory or ResearchMemory()
        cycle = ImprovementCycle(
            gate=gate, memory=memory, factory=factory,
            backtest_fn=fake_backtester(sharpe_by_key, none_keys),
        )
        return cycle, factory, gate, memory

    def test_every_evaluated_candidate_is_counted_as_a_trial(self):
        factory = small_factory()
        ks = keys_for(factory)
        cycle, _, gate, _ = self._cycle({k: 0.2 for k in ks})
        cycle.factory = factory
        result = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        # Three fresh candidates -> three trials counted, exactly.
        self.assertEqual(result.evaluated, 3)
        self.assertEqual(gate.ledger.trials("crypto"), 3)

    def test_winner_is_the_highest_sharpe_promoted_candidate(self):
        factory = small_factory()
        k0, k1, k2 = keys_for(factory)
        cycle, _, _, memory = self._cycle({k0: 3.0, k1: 2.5, k2: 0.2})
        cycle.factory = factory
        result = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertTrue(result.promoted)
        self.assertEqual(result.promoted_key, k0)          # 3.0 beats 2.5
        self.assertEqual(memory.best("crypto").config_key, k0)

    def test_nothing_promoted_when_all_mediocre(self):
        factory = small_factory()
        ks = keys_for(factory)
        cycle, _, _, _ = self._cycle({k: 0.2 for k in ks})
        cycle.factory = factory
        result = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertFalse(result.promoted)

    def test_second_round_skips_already_tried_without_recounting(self):
        factory = small_factory()
        ks = keys_for(factory)
        gate = PromotionGate(ledger=TrialLedger())
        memory = ResearchMemory()
        cycle, _, _, _ = self._cycle({k: 0.2 for k in ks}, memory=memory, gate=gate)
        cycle.factory = factory
        cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertEqual(gate.ledger.trials("crypto"), 3)

        second = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertEqual(second.evaluated, 0)
        self.assertEqual(second.skipped, 3)
        self.assertEqual(gate.ledger.trials("crypto"), 3)   # NOT re-counted

    def test_insufficient_data_is_recorded_but_not_counted(self):
        factory = small_factory()
        k0, k1, k2 = keys_for(factory)
        cycle, _, gate, memory = self._cycle({k0: 3.0, k1: 3.0, k2: 3.0},
                                             none_keys={k1})
        cycle.factory = factory
        result = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        # k1 had no walk-forward: recorded (so it is not re-tested) but NOT a trial.
        self.assertEqual(result.evaluated, 2)
        self.assertEqual(gate.ledger.trials("crypto"), 2)
        self.assertTrue(memory.seen("crypto", k1))

    def test_champion_becomes_the_incumbent_next_round(self):
        # Round 1 promotes k0. Round 2 (with a fresh factory point) must treat k0
        # as the incumbent and re-backtest it as the reference.
        factory = CandidateFactory(grid={"trail_pct": ("0.05", "0.08")})
        k0, k1 = keys_for(factory)
        gate = PromotionGate(ledger=TrialLedger())
        memory = ResearchMemory()
        # Round 1: only k0 exists as a candidate (k1 avoided) so it becomes champion.
        cycle = ImprovementCycle(
            gate=gate, memory=memory,
            factory=CandidateFactory(grid={"trail_pct": ("0.05",)}),
            backtest_fn=fake_backtester({k0: 3.0}),
        )
        r1 = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertEqual(r1.promoted_key, k0)
        self.assertIsNone(r1.incumbent_key)                 # no incumbent in round 1

        # Round 2: k1 is the fresh candidate; k0 is the champion/incumbent.
        cycle.factory = factory
        cycle.backtest_fn = fake_backtester({k0: 3.0, k1: 0.2})
        r2 = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertEqual(r2.incumbent_key, k0)
        self.assertEqual(r2.evaluated, 1)                   # only k1 is fresh
        self.assertFalse(r2.promoted)                       # 0.2 cannot beat the champion

    def test_challenger_must_beat_incumbent_by_margin(self):
        # A challenger only marginally better than the champion must be refused.
        factory = CandidateFactory(grid={"trail_pct": ("0.05", "0.08")})
        k0, k1 = keys_for(factory)
        gate = PromotionGate(ledger=TrialLedger())
        memory = ResearchMemory()
        cycle = ImprovementCycle(
            gate=gate, memory=memory,
            factory=CandidateFactory(grid={"trail_pct": ("0.05",)}),
            backtest_fn=fake_backtester({k0: 3.0}),
        )
        cycle.run_round("crypto", "BTC-USD", [None] * 10)   # k0 champion at Sharpe 3.0

        cycle.factory = factory
        # k1 at 3.05 -- better, but well within one standard error (0.1) of 3.0.
        cycle.backtest_fn = fake_backtester({k0: 3.0, k1: 3.05})
        r2 = cycle.run_round("crypto", "BTC-USD", [None] * 10)
        self.assertFalse(r2.promoted)
        self.assertTrue(any("margin" in " ".join(v.notes).lower() or
                            "incumbent" in " ".join(v.notes).lower()
                            for v in r2.verdicts))


if __name__ == "__main__":
    unittest.main()
