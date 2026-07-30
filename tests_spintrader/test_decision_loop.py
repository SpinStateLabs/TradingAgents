"""Tests for the two-tier decision loop. No network, no LLM, no database.

The behaviours that matter most, and are asserted here:

* the fast loop routes an intent through the real RiskEngine / PaperVenue /
  Ledger and a fill actually moves the book;
* an expired or empty mandate forbids *opening* a position;
* but an EXIT still fires under an expired mandate (the "exits always reachable"
  decision) -- while a tripped kill switch still halts everything (the boundary);
* the loop cannot bypass the live gate.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.agents.personas.baseline_trend import BaselineTrendAgent
from spintrader.agents.personas.spec import Horizon
from spintrader.agents.panel import PanelVerdict, PersonaVote
from spintrader.backtest.runner import backtest_instrument, costs_for
from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    Action, AssetClass, Bar, Side, TradingMode, to_decimal,
)
from spintrader.llm.router import LLMResponse, Tier
from spintrader.loop.context import build_context, gather_contexts
from spintrader.loop.decision_loop import DecisionLoop, LiveCursor, _synth_quote
from spintrader.loop.mandate import MandateService, build_mandate
from spintrader.loop.voting import BootstrapVoter, LLMVoter, VoteItem
from spintrader.risk.engine import Mandate, RiskEngine
from spintrader.venues.paper import PaperVenue

D = Decimal
UTC = timezone.utc
KEY = "paper:BTC-USD"
T0 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# Bar builders
# --------------------------------------------------------------------------

def bar(i: int, close: float, key: str = KEY, interval: str = "1m") -> Bar:
    c = close
    return Bar(
        instrument_key=key, ts=T0 + timedelta(minutes=i + 1), interval=interval,
        open=D(str(c - 0.02)), high=D(str(c + 0.2)), low=D(str(c - 0.2)),
        close=D(str(c)), volume=D("1"),
    )


def uptrend(n: int = 30, start: float = 100.0, step: float = 0.05) -> list[Bar]:
    # Gentle rise with a small alternating wobble, so realised vol is nonzero
    # (the agent refuses to open when vol is ~0) but modest.
    out = []
    for i in range(n):
        wobble = 0.03 if i % 2 else -0.03
        out.append(bar(i, start + step * i + wobble))
    return out


def crash(base_index: int, n: int = 6, start: float = 100.0) -> list[Bar]:
    # A sharp decline that breaks any trailing average.
    return [bar(base_index + i, start * (1 - 0.03 * (i + 1))) for i in range(n)]


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeStore:
    """Serves a fixed bar list per key; records persisted decisions."""

    def __init__(self, bars_by_key):
        self.bars_by_key = {k: list(v) for k, v in bars_by_key.items()}
        self.decisions = []

    def read_bars(self, instrument_key, interval, start=None, end=None, limit=None):
        bars = self.bars_by_key.get(instrument_key, [])
        return bars[-limit:] if limit else list(bars)

    def write_decision(self, decision, mode):
        self.decisions.append((decision, mode))


def lenient_agent(**kw):
    """A trend agent tuned so the loop's mechanics, not the signal thresholds,
    are what the test exercises. Signal thresholds are covered elsewhere."""
    params = dict(
        fast_window=3, slow_window=10, vol_window=5, interval="1m",
        continuous=True, vol_ceiling="100", min_annual_vol="0.00001",
        stop_pct="0.05", trail_pct="0.05", spread_bps="3",
    )
    params.update(kw)
    return BaselineTrendAgent(**params)


class AlwaysBuyStrategy:
    """Emits a BUY intent every bar, ignoring the mandate.

    Used to test the RiskEngine's mandate gate directly: BaselineTrendAgent
    self-censors (returns () when the mandate forbids), so with it no intent ever
    reaches the engine and a deleted engine gate would go unnoticed. This stub
    forces the intent through so the engine's permit/bias rejection is exercised.
    """
    name = "always_buy"
    warmup_bars = 2

    def on_bar(self, cursor, instrument, mandate):
        from spintrader.risk.engine import TradeIntent
        return [TradeIntent(
            instrument=instrument, side=Side.BUY, edge=D("0.05"),
            confidence=D("0.9"), volatility=D("0.20"),
            quote=cursor.quote(D("3")), strategy="stub",
        )]

    def fit(self, bars):
        pass


def make_loop(bars, *, aggression=Aggression.MODERATE, mode=TradingMode.PAPER,
              starting_cash="10000", strategy=None, live_gate=None):
    """Wire a single-instrument paper loop over a fixed bar series."""
    instrument = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
    store = FakeStore({instrument.key: bars})
    settings = Settings(
        mode=mode, aggression=aggression, base_currency="USD",
        live=live_gate or LiveGate(enabled=False), enforce_cash_account=True,
    )
    costs = costs_for(AssetClass.CRYPTO)

    holder = {}
    venue = PaperVenue(
        quote_source=lambda inst: holder["loop"].market_quote(inst),
        starting_cash=to_decimal(starting_cash), currency="USD",
        settings=settings, slippage=costs.slippage, settlement_days=1,
    )
    venue.register(instrument)
    venue.connect()

    from spintrader.portfolio.ledger import Ledger
    ledger = Ledger(base_currency="USD", mode=mode, settlement_days=1,
                    opening_cash={"USD": to_decimal(starting_cash)})
    risk = RiskEngine(settings=settings)
    strat = strategy or lenient_agent()

    loop = DecisionLoop(
        settings=settings, store=store, venue=venue, ledger=ledger, risk=risk,
        strategies={instrument.key: strat}, mandate_service=None,  # not used here
        instruments=[instrument], interval="1m", spread_bps=costs.spread_bps,
        horizon=Horizon.INTRADAY,
    )
    holder["loop"] = loop
    return loop, instrument, ledger, store


def permissive_mandate(key=KEY, hours=1):
    now = datetime.now(UTC)
    return Mandate(issued_at=now, expires_at=now + timedelta(hours=hours),
                   permitted=frozenset({key}))


# --------------------------------------------------------------------------
# LiveCursor
# --------------------------------------------------------------------------

class LiveCursorTests(unittest.TestCase):
    def test_positioned_at_last_bar(self):
        bars = uptrend(20)
        cur = LiveCursor(bars, _synth_quote(bars[-1], D("3")))
        self.assertEqual(cur.current, bars[-1])
        self.assertEqual(cur.now, bars[-1].ts)

    def test_history_returns_trailing_window(self):
        bars = uptrend(20)
        cur = LiveCursor(bars, _synth_quote(bars[-1], D("3")))
        self.assertEqual(cur.history(5), bars[-5:])
        self.assertEqual(cur.history(), bars)

    def test_quote_is_the_supplied_one(self):
        bars = uptrend(5)
        q = _synth_quote(bars[-1], D("10"))
        cur = LiveCursor(bars, q)
        self.assertIs(cur.quote(D("999")), q)      # spread arg ignored

    def test_empty_bars_rejected(self):
        with self.assertRaises(ValueError):
            LiveCursor([], _synth_quote(bar(0, 100), D("3")))


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------

class ContextTests(unittest.TestCase):
    def test_uptrend_has_positive_trend(self):
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        ctx = build_context(inst, uptrend(60), "1m", Horizon.INTRADAY,
                            fast_window=5, slow_window=30)
        self.assertGreater(ctx.trend_strength, 0)
        self.assertGreater(ctx.annual_vol, 0)
        self.assertEqual(ctx.last_close, uptrend(60)[-1].close)

    def test_assess_regime_is_benign_when_the_model_cannot_fit(self):
        # The "regime unavailable" contract is (None, 0, 0) -- explicitly benign,
        # never a crisis default. Forced deterministically via a fit failure so
        # the result does not depend on whether hmmlearn is installed on the box.
        from unittest import mock
        from spintrader.loop import context as ctxmod
        from spintrader.quant.regime import RegimeError
        with mock.patch("spintrader.quant.regime.RegimeModel.fit",
                        side_effect=RegimeError("hmmlearn is not installed")):
            label, risk, conf = ctxmod.assess_regime(uptrend(120), "1m", True)
        self.assertIsNone(label)
        self.assertEqual(risk, D("0"))
        self.assertEqual(conf, 0.0)

    def test_build_context_propagates_benign_regime_when_unavailable(self):
        from unittest import mock
        from spintrader.quant.regime import RegimeError
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        with mock.patch("spintrader.quant.regime.RegimeModel.fit",
                        side_effect=RegimeError("hmmlearn is not installed")):
            ctx = build_context(inst, uptrend(120), "1m", Horizon.INTRADAY,
                                with_regime=True)
        self.assertIsNone(ctx.regime_label)
        self.assertEqual(ctx.regime_risk, D("0"))

    def test_gather_reads_each_instrument(self):
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        store = FakeStore({inst.key: uptrend(40)})
        ctxs = gather_contexts(store, [inst], "1m", Horizon.INTRADAY, lookback=40)
        self.assertIn(inst.key, ctxs)
        self.assertEqual(len(ctxs[inst.key].bars), 40)


# --------------------------------------------------------------------------
# Bootstrap voter
# --------------------------------------------------------------------------

class BootstrapVoterTests(unittest.TestCase):
    def _ctx(self, bars, regime_risk="0"):
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        ctx = build_context(inst, bars, "1m", Horizon.INTRADAY,
                            fast_window=3, slow_window=10)
        ctx.regime_risk = D(regime_risk)
        return ctx

    def _spec(self):
        from spintrader.agents.personas.roster import default_roster
        return default_roster().applicable(AssetClass.CRYPTO, Horizon.INTRADAY)[0]

    def test_uptrend_votes_buy(self):
        item = VoteItem(self._spec(), self._ctx(uptrend(40, step=0.2)))
        votes = BootstrapVoter().vote_all([item])
        vote = votes[item.instrument_key][0]
        self.assertEqual(vote.action, Action.BUY)
        self.assertGreater(vote.confidence, D("0"))

    def test_downtrend_votes_sell(self):
        down = [bar(i, 130 - 0.3 * i) for i in range(40)]
        item = VoteItem(self._spec(), self._ctx(down))
        vote = BootstrapVoter().vote_all([item])[item.instrument_key][0]
        self.assertEqual(vote.action, Action.SELL)

    def test_single_lens_so_panel_penalises_diversity(self):
        from spintrader.agents.personas.spec import Lens
        item = VoteItem(self._spec(), self._ctx(uptrend(40, step=0.2)))
        vote = BootstrapVoter().vote_all([item])[item.instrument_key][0]
        self.assertEqual(vote.lenses, (Lens.STATISTICAL,))

    def test_regime_risk_damps_confidence(self):
        calm = VoteItem(self._spec(), self._ctx(uptrend(40, step=0.2), "0"))
        crisis = VoteItem(self._spec(), self._ctx(uptrend(40, step=0.2), "1"))
        c_calm = BootstrapVoter().vote_all([calm])[calm.instrument_key][0].confidence
        c_crisis = BootstrapVoter().vote_all([crisis])[crisis.instrument_key][0].confidence
        self.assertGreater(c_calm, c_crisis)


# --------------------------------------------------------------------------
# LLM voter (stubbed router)
# --------------------------------------------------------------------------

class StubRouter:
    def __init__(self, replies):
        self._replies = replies      # list of str | Exception
        self.batches = []

    def run_batched(self, requests_by_tier, keep_alive="20m"):
        self.batches.append(requests_by_tier)
        tier = next(iter(requests_by_tier))
        out = []
        for reply in self._replies:
            if isinstance(reply, Exception):
                out.append(reply)
            else:
                out.append(LLMResponse(text=reply, model="stub", tier=tier))
        return {tier: out}


class LLMVoterTests(unittest.TestCase):
    def _items(self, n=2):
        from spintrader.agents.personas.roster import default_roster
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        ctx = build_context(inst, uptrend(40), "1m", Horizon.INTRADAY)
        specs = default_roster().applicable(AssetClass.CRYPTO, Horizon.INTRADAY)[:n]
        return [VoteItem(s, ctx) for s in specs], inst.key

    def test_parses_json_into_votes(self):
        items, key = self._items(2)
        router = StubRouter([
            '{"action":"buy","confidence":0.8,"rationale":"trend","changed_by":"break"}',
            '{"action":"hold","confidence":0.0,"rationale":"no view"}',
        ])
        votes = LLMVoter(router).vote_all(items)[key]
        self.assertEqual(votes[0].action, Action.BUY)
        self.assertEqual(votes[0].confidence, D("0.8"))
        self.assertTrue(votes[1].abstained)

    def test_llm_error_becomes_abstention(self):
        items, key = self._items(1)
        router = StubRouter([RuntimeError("timeout")])
        vote = LLMVoter(router).vote_all(items)[key][0]
        self.assertTrue(vote.abstained)
        self.assertIn("llm error", vote.rationale)

    def test_unparseable_reply_becomes_abstention(self):
        items, key = self._items(1)
        vote = LLMVoter(StubRouter(["not json at all"])).vote_all(items)[key][0]
        self.assertTrue(vote.abstained)

    def test_all_calls_run_in_one_quick_batch(self):
        items, key = self._items(2)
        router = StubRouter([
            '{"action":"buy","confidence":0.7,"rationale":"a"}',
            '{"action":"buy","confidence":0.6,"rationale":"b"}',
        ])
        LLMVoter(router).vote_all(items)
        self.assertEqual(len(router.batches), 1)              # one batched pass
        self.assertIn(Tier.QUICK, router.batches[0])
        self.assertEqual(len(router.batches[0][Tier.QUICK]), 2)


# --------------------------------------------------------------------------
# build_mandate
# --------------------------------------------------------------------------

def verdict(action, confidence, net=None):
    net = to_decimal(net if net is not None else (confidence if action is Action.BUY else -confidence))
    return PanelVerdict(action=action, confidence=to_decimal(confidence), net_direction=net)


class BuildMandateTests(unittest.TestCase):
    def test_directional_verdict_permits(self):
        m = build_mandate({KEY: verdict(Action.BUY, "0.7")}, {KEY: D("0")},
                          ttl=timedelta(hours=1))
        self.assertIn(KEY, m.permitted)
        self.assertGreater(m.bias_for(KEY), 0)

    def test_hold_does_not_permit(self):
        m = build_mandate({KEY: verdict(Action.HOLD, "0.9", net="0")}, {KEY: D("0")},
                          ttl=timedelta(hours=1))
        self.assertNotIn(KEY, m.permitted)

    def test_low_confidence_does_not_permit(self):
        m = build_mandate({KEY: verdict(Action.BUY, "0.05")}, {KEY: D("0")},
                          ttl=timedelta(hours=1))
        self.assertNotIn(KEY, m.permitted)

    def test_empty_verdicts_permit_nothing(self):
        m = build_mandate({}, {}, ttl=timedelta(hours=1))
        self.assertEqual(m.permitted, frozenset())

    def test_regime_risk_is_the_max(self):
        m = build_mandate(
            {"paper:A": verdict(Action.BUY, "0.7"), "paper:B": verdict(Action.BUY, "0.7")},
            {"paper:A": D("0.2"), "paper:B": D("0.9")},
            ttl=timedelta(hours=1),
        )
        self.assertEqual(m.regime_risk, D("0.9"))

    def test_expiry_follows_ttl(self):
        now = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
        m = build_mandate({KEY: verdict(Action.BUY, "0.7")}, {KEY: D("0")},
                          ttl=timedelta(minutes=90), now=now)
        self.assertEqual(m.expires_at, now + timedelta(minutes=90))


class MandateServiceTests(unittest.TestCase):
    def test_bootstrap_deliberation_permits_uptrend(self):
        from spintrader.agents.personas.roster import default_roster
        inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
        ctx = build_context(inst, uptrend(60, step=0.3), "1m", Horizon.INTRADAY,
                            fast_window=3, slow_window=10)
        service = MandateService(default_roster(), voter=BootstrapVoter())
        delib = service.deliberate({inst.key: ctx})
        self.assertIn(inst.key, delib.mandate.permitted)
        self.assertIn(inst.key, delib.verdicts)


# --------------------------------------------------------------------------
# Fast loop
# --------------------------------------------------------------------------

class FastLoopEntryTests(unittest.TestCase):
    def test_entry_fills_and_moves_the_book(self):
        loop, inst, ledger, store = make_loop(uptrend(40, step=0.3))
        results = loop.fast_tick(permissive_mandate())
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].approved)
        self.assertTrue(results[0].submitted)
        self.assertGreater(results[0].filled_qty, D("0"))
        self.assertGreater(ledger.position(inst.key).qty, D("0"))
        self.assertTrue(store.decisions)                 # decision was persisted

    def test_expired_mandate_blocks_entry(self):
        loop, inst, ledger, _ = make_loop(uptrend(40, step=0.3))
        past = datetime.now(UTC) - timedelta(hours=2)
        expired = Mandate(issued_at=past, expires_at=past, permitted=frozenset({inst.key}))
        results = loop.fast_tick(expired)
        self.assertTrue(all(not r.approved for r in results))
        self.assertEqual(ledger.position(inst.key).qty, D("0"))

    def test_unpermitted_instrument_blocks_entry_at_the_engine(self):
        # Force an intent through (AlwaysBuyStrategy) so the RiskEngine's permit
        # gate is what does the rejecting, not the strategy declining to emit.
        loop, inst, ledger, _ = make_loop(uptrend(40, step=0.3),
                                          strategy=AlwaysBuyStrategy())
        now = datetime.now(UTC)
        empty = Mandate(issued_at=now, expires_at=now + timedelta(hours=1),
                        permitted=frozenset())          # empty permits nothing
        results = loop.fast_tick(empty)
        self.assertTrue(results, "the stub strategy should have produced an intent")
        self.assertTrue(all(not r.approved for r in results))
        self.assertTrue(any("not in the current mandate" in " ".join(r.risk.reasons)
                            for r in results))
        self.assertEqual(ledger.position(inst.key).qty, D("0"))

    def test_short_bias_blocks_a_long_entry_at_the_engine(self):
        loop, inst, ledger, _ = make_loop(uptrend(40, step=0.3),
                                          strategy=AlwaysBuyStrategy())
        now = datetime.now(UTC)
        bearish = Mandate(issued_at=now, expires_at=now + timedelta(hours=1),
                          permitted=frozenset({inst.key}),   # permitted, but short-biased
                          directional_bias={inst.key: D("-0.5")})
        results = loop.fast_tick(bearish)
        self.assertTrue(results)
        self.assertTrue(all(not r.approved for r in results))
        self.assertTrue(any("bias" in " ".join(r.risk.reasons) for r in results))
        self.assertEqual(ledger.position(inst.key).qty, D("0"))

    def test_paper_mode_orders_are_not_blocked_by_the_gate(self):
        # Sanity: the closed gate is a no-op for simulated modes.
        loop, inst, ledger, _ = make_loop(uptrend(40, step=0.3))
        self.assertIs(loop.settings.mode, TradingMode.PAPER)
        loop.fast_tick(permissive_mandate())
        self.assertGreater(ledger.position(inst.key).qty, D("0"))


class LiveGateSafetyTests(unittest.TestCase):
    def test_live_mode_without_arming_is_refused(self):
        # The loop must not be able to route a real order while disarmed. The
        # gate is enforced twice -- in the risk engine AND in Venue.submit -- so
        # a disarmed order is rejected before it is even sized. Either layer
        # refusing is the property that matters: nothing goes out.
        loop, inst, ledger, _ = make_loop(
            uptrend(40, step=0.3), mode=TradingMode.LIVE,
            live_gate=LiveGate(enabled=False),
        )
        results = loop.fast_tick(permissive_mandate())
        self.assertTrue(results, "the strategy should still have produced an intent")
        self.assertTrue(all(not r.submitted for r in results))
        # The refusal names the disarmed gate, from whichever layer caught it.
        self.assertTrue(any(
            ("disarm" in " ".join(r.risk.reasons).lower())
            or (r.error and "disarm" in r.error.lower())
            for r in results
        ))
        self.assertEqual(ledger.position(inst.key).qty, D("0"))


class ExitReachabilityTests(unittest.TestCase):
    """The decision: a held position can be exited under a stale mandate."""

    def _enter_then(self, exit_bars, mandate_for_exit):
        strat = lenient_agent()
        loop, inst, ledger, _ = make_loop(uptrend(40, step=0.3), strategy=strat)
        loop.fast_tick(permissive_mandate())
        held = ledger.position(inst.key).qty
        self.assertGreater(held, D("0"), "entry should have opened a position")

        # Append a crash so the strategy wants out, then tick with the given mandate.
        all_bars = uptrend(40, step=0.3) + crash(40, n=8, start=100 + 0.3 * 39)
        loop.store.bars_by_key[inst.key] = all_bars
        results = loop.fast_tick(mandate_for_exit)
        return loop, inst, ledger, results

    def test_exit_fires_under_expired_mandate(self):
        past = datetime.now(UTC) - timedelta(hours=2)
        expired = Mandate(issued_at=past, expires_at=past, permitted=frozenset())
        loop, inst, ledger, results = self._enter_then(crash, expired)
        # The exit was a reduction and went through despite the expired mandate.
        self.assertTrue(any(r.reducing and r.submitted for r in results))
        self.assertEqual(ledger.position(inst.key).qty, D("0"))

    def test_exit_fires_when_instrument_not_permitted(self):
        now = datetime.now(UTC)
        unpermitted = Mandate(issued_at=now, expires_at=now + timedelta(hours=1),
                              permitted=frozenset())     # holds nothing in play
        loop, inst, ledger, results = self._enter_then(crash, unpermitted)
        self.assertEqual(ledger.position(inst.key).qty, D("0"))

    def test_tripped_kill_switch_still_halts_exits(self):
        # The boundary: exits bypass the mandate, but NOT the kill switch, which
        # halts everything pending a human reset by design.
        past = datetime.now(UTC) - timedelta(hours=2)
        expired = Mandate(issued_at=past, expires_at=past, permitted=frozenset())
        strat = lenient_agent()
        loop, inst, ledger, _ = make_loop(uptrend(40, step=0.3), strategy=strat)
        loop.fast_tick(permissive_mandate())
        self.assertGreater(ledger.position(inst.key).qty, D("0"))

        loop.risk.kill_switch._trip("manual trip for test")
        all_bars = uptrend(40, step=0.3) + crash(40, n=8, start=100 + 0.3 * 39)
        loop.store.bars_by_key[inst.key] = all_bars
        results = loop.fast_tick(expired)
        self.assertTrue(all(not r.submitted for r in results))
        self.assertGreater(ledger.position(inst.key).qty, D("0"))   # trapped, by design


class DriverTests(unittest.TestCase):
    def test_run_refreshes_mandate_then_ticks(self):
        from spintrader.agents.personas.roster import default_roster
        loop, inst, ledger, _ = make_loop(uptrend(60, step=0.3))
        loop.mandate_service = MandateService(default_roster(), voter=BootstrapVoter())
        out = loop.run(fast_interval_s=0, slow_interval_s=10_000, max_ticks=1,
                       install_signal_handlers=False)
        self.assertEqual(out["ticks"], 1)
        self.assertIsNotNone(loop.deliberation)
        # The bootstrap mandate permitted the uptrend, so the tick opened a position.
        self.assertGreater(ledger.position(inst.key).qty, D("0"))


if __name__ == "__main__":
    unittest.main()
