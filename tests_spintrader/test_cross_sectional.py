"""Tests for the cross-sectional backtester (diversified-alpha follow-on).

Five properties carry the weight.

**Alignment.** The universe is replayed on the union of all instruments'
timestamps, and a name missing at an instant is simply not a candidate then --
never forward-filled with a stale price into a decision.

**Top-K selection.** Each bar the strongest signals by conviction-weighted edge
are the ones funded.

**max_positions.** The book-wide concurrent-position cap is respected even when
more names would otherwise be entered.

**Single-instrument reproduction.** With one name the cross-sectional engine must
reproduce :meth:`BacktestEngine.run` exactly -- same code paths, same numbers --
so the new loop cannot have quietly forked sizing, fills or P&L.

**No lookahead (truncation invariance).** Replaying a universe truncated at ``t``
yields byte-identical decisions and equity up to ``t``. If a future bar could
leak into an earlier decision, this is where it shows.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.agents.personas.mean_reversion import MeanReversionAgent
from spintrader.backtest.cross_sectional import (
    CrossSectionalEngine, CrossSectionalRun, rank_key, run_cross_sectional_backtest,
)
from spintrader.backtest.engine import BacktestEngine
from spintrader.backtest.runner import (
    backtest_instrument, backtest_settings, costs_for,
)
from spintrader.core.types import AssetClass, Bar, Instrument, Side
from spintrader.risk.engine import TradeIntent

D = Decimal
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
CRYPTO = AssetClass.CRYPTO


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def instrument(symbol: str) -> Instrument:
    return backtest_instrument(symbol, CRYPTO, costs_for(CRYPTO))


def bar(inst: Instrument, day: int, close: float, interval: str = "1d") -> Bar:
    price = float(close)
    return Bar(
        instrument_key=inst.key, ts=T0 + timedelta(days=day), interval=interval,
        open=D(str(round(price, 6))), high=D(str(round(price * 1.005, 6))),
        low=D(str(round(price * 0.995, 6))), close=D(str(round(price, 6))),
        volume=D("100"),
    )


def series(inst: Instrument, closes, start_day: int = 0) -> list[Bar]:
    return [bar(inst, start_day + i, c) for i, c in enumerate(closes)]


def engine(top_k=None, cash="1000000", enforce_cash_account=False):
    costs = costs_for(CRYPTO)
    return CrossSectionalEngine(
        settings=backtest_settings(enforce_cash_account=enforce_cash_account),
        starting_cash=D(cash), spread_bps=costs.spread_bps,
        slippage=costs.slippage, periods_per_year=365, top_k=top_k,
    )


class StubStrategy:
    """A deterministic ranker: buys each name once with a per-name edge.

    Real personas depend on price patterns to fire; this one fires on the first
    bar it sees with an edge read straight off the instrument, so top-K and the
    position cap can be tested without coaxing a signal out of synthetic prices.
    """
    name = "stub"
    warmup_bars = 1

    def __init__(self, edge_by_symbol, confidence="0.9", volatility="1.0",
                 entry_index_by_symbol=None):
        self.edge_by_symbol = edge_by_symbol
        self.confidence = D(str(confidence))
        self.volatility = D(str(volatility))
        # Optional per-name entry bar, to stagger entries across days (so the
        # daily-trade cap does not stand in for the position cap under test).
        self.entry_index_by_symbol = entry_index_by_symbol or {}
        self.done = False

    def fit(self, bars):
        self.done = False

    def on_bar(self, cursor, instrument, mandate):
        if self.done:
            return ()
        if cursor.index < self.entry_index_by_symbol.get(instrument.symbol, 0):
            return ()
        edge = self.edge_by_symbol.get(instrument.symbol)
        if edge is None:
            return ()
        self.done = True
        return (TradeIntent(
            instrument=instrument, side=Side.BUY, edge=D(str(edge)),
            confidence=self.confidence, volatility=self.volatility,
            quote=cursor.quote(D("5")), strategy=self.name,
        ),)


def held_keys(eng) -> set[str]:
    return {k for k, p in eng._last_ledger.positions.items() if not p.is_flat}


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

class AlignmentTests(unittest.TestCase):
    def test_union_timeline_when_a_name_is_missing_early(self):
        a = instrument("AAA-USD")
        b = instrument("BBB-USD")
        # A exists days 0..9; B only appears on day 5.
        members = [
            (a, series(a, [100 + i for i in range(10)], start_day=0)),
            (b, series(b, [100 + i for i in range(5)], start_day=5)),
        ]
        eng = engine(top_k=5)
        edges = {"AAA-USD": 0.05, "BBB-USD": 0.05}
        result = eng.run_cross_sectional(lambda: StubStrategy(edges), members)

        # One equity point per instant on the UNION of timestamps (10), not the
        # intersection and not A's count alone.
        self.assertEqual(len(result.equity_curve), 10)
        self.assertEqual(result.timestamps[0], T0)
        self.assertEqual(result.timestamps[-1], T0 + timedelta(days=9))

    def test_missing_name_is_not_traded_before_it_exists(self):
        a = instrument("AAA-USD")
        b = instrument("BBB-USD")
        members = [
            (a, series(a, [100 + i for i in range(10)], start_day=0)),
            (b, series(b, [100 + i for i in range(5)], start_day=5)),
        ]
        eng = engine(top_k=5)
        edges = {"AAA-USD": 0.05, "BBB-USD": 0.05}
        eng.run_cross_sectional(lambda: StubStrategy(edges), members)

        fills = eng._last_venue.fills
        a_first = min(f.ts for f in fills if f.instrument_key == a.key)
        b_first = min(f.ts for f in fills if f.instrument_key == b.key)
        # A trades on day 0; B cannot trade until the day it first exists (5).
        self.assertEqual(a_first, T0)
        self.assertEqual(b_first, T0 + timedelta(days=5))


# --------------------------------------------------------------------------
# Top-K selection and position cap
# --------------------------------------------------------------------------

class RankingTests(unittest.TestCase):
    def _members(self, symbols):
        out = []
        for s in symbols:
            inst = instrument(s)
            out.append((inst, series(inst, [100 + 0.1 * i for i in range(30)])))
        return out

    def test_only_top_k_names_are_funded(self):
        symbols = ["AAA-USD", "BBB-USD", "CCC-USD", "DDD-USD", "EEE-USD"]
        edges = {"AAA-USD": 0.05, "BBB-USD": 0.04, "CCC-USD": 0.03,
                 "DDD-USD": 0.02, "EEE-USD": 0.01}
        eng = engine(top_k=2)
        eng.run_cross_sectional(lambda: StubStrategy(edges), self._members(symbols))

        held = held_keys(eng)
        self.assertEqual(len(held), 2)
        # The two strongest signals by edge (confidence is equal) win the slots.
        self.assertEqual(
            held, {instrument("AAA-USD").key, instrument("BBB-USD").key},
        )

    def test_max_positions_caps_concurrent_holdings(self):
        # More candidates than the profile allows concurrently, with top_k wide
        # enough to attempt them all -- the book cap, not top_k, must be binding.
        cap = engine().settings.risk.max_positions
        # One more name than the cap allows. Each enters on its own day and never
        # exits, so holdings accumulate one per day: neither the daily-trade cap
        # nor (with tiny, high-vol positions) gross exposure binds first -- only
        # the concurrent-position count does.
        symbols = [f"C{i}-USD" for i in range(cap + 1)]
        edges = {s: 0.10 for s in symbols}
        entry = {s: i for i, s in enumerate(symbols)}
        members = [
            (instrument(s), series(instrument(s), [100 + 0.1 * i for i in range(cap + 2)]))
            for s in symbols
        ]
        eng = engine(top_k=cap + 1)
        result = eng.run_cross_sectional(
            lambda: StubStrategy(edges, volatility="8.0", entry_index_by_symbol=entry),
            members,
        )

        self.assertEqual(len(held_keys(eng)), cap)        # == profile.max_positions
        self.assertGreaterEqual(result.rejections.get("max_positions", 0), 1)
        # The last name to try to enter is the one the cap turns away.
        self.assertNotIn(instrument(symbols[-1]).key, held_keys(eng))

    def test_rank_key_is_edge_times_confidence(self):
        inst = instrument("AAA-USD")
        strong_tentative = TradeIntent(
            instrument=inst, side=Side.BUY, edge=D("0.10"), confidence=D("0.5"),
            volatility=D("1"), quote=None, strategy="x",
        )
        weak_certain = TradeIntent(
            instrument=inst, side=Side.BUY, edge=D("0.06"), confidence=D("0.95"),
            volatility=D("1"), quote=None, strategy="x",
        )
        self.assertEqual(rank_key(strong_tentative), D("0.05"))
        self.assertGreater(rank_key(weak_certain), rank_key(strong_tentative))


# --------------------------------------------------------------------------
# Single-instrument reproduction
# --------------------------------------------------------------------------

class ReproductionTests(unittest.TestCase):
    def test_one_name_reproduces_the_single_instrument_engine(self):
        inst = instrument("BTC-USD")
        closes = [100 + math.sin(i / 10) * 8 for i in range(300)]
        bars = series(inst, closes)
        cfg = dict(lookback=20, interval="1d", continuous=True,
                   spread_bps=costs_for(CRYPTO).spread_bps)
        st = backtest_settings(enforce_cash_account=False)
        costs = costs_for(CRYPTO)

        single = BacktestEngine(
            settings=st, starting_cash=D("100000"), spread_bps=costs.spread_bps,
            slippage=costs.slippage, periods_per_year=365,
        ).run(MeanReversionAgent(**cfg), inst, bars)

        cross = CrossSectionalEngine(
            settings=st, starting_cash=D("100000"), spread_bps=costs.spread_bps,
            slippage=costs.slippage, periods_per_year=365,
        ).run_cross_sectional(lambda: MeanReversionAgent(**cfg), [(inst, bars)])

        self.assertEqual(single.n_orders, cross.n_orders)
        self.assertGreater(single.n_orders, 0)            # it actually traded
        self.assertEqual(len(single.equity_curve), len(cross.equity_curve))
        for x, y in zip(single.equity_curve, cross.equity_curve):
            self.assertTrue(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6))
        for x, y in zip(single.benchmark_curve, cross.benchmark_curve):
            self.assertTrue(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6))
        self.assertTrue(math.isclose(
            single.scorecard.sharpe, cross.scorecard.sharpe,
            rel_tol=1e-9, abs_tol=1e-9,
        ))


# --------------------------------------------------------------------------
# No lookahead
# --------------------------------------------------------------------------

class LookaheadTests(unittest.TestCase):
    def _universe(self):
        a = instrument("AAA-USD")
        b = instrument("BBB-USD")
        ca = [100 + math.sin(i / 9) * 8 for i in range(200)]
        cb = [100 + math.cos(i / 11) * 7 for i in range(200)]
        return [(a, series(a, ca)), (b, series(b, cb))]

    def test_truncation_invariance(self):
        cfg = dict(lookback=20, interval="1d", continuous=True,
                   spread_bps=costs_for(CRYPTO).spread_bps)
        members = self._universe()

        full = engine().run_cross_sectional(
            lambda: MeanReversionAgent(**cfg), members,
        )

        cutoff = T0 + timedelta(days=119)                 # first 120 timestamps
        truncated_members = [
            (inst, [b for b in bars if b.ts <= cutoff]) for inst, bars in members
        ]
        truncated = engine().run_cross_sectional(
            lambda: MeanReversionAgent(**cfg), truncated_members,
        )

        self.assertEqual(len(truncated.equity_curve), 120)
        # Every decision and mark up to the cutoff is identical whether or not the
        # future bars were present in the input.
        for x, y in zip(full.equity_curve[:120], truncated.equity_curve):
            self.assertTrue(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6))
        for x, y in zip(full.benchmark_curve[:120], truncated.benchmark_curve):
            self.assertTrue(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6))


# --------------------------------------------------------------------------
# Runner entry + walk-forward
# --------------------------------------------------------------------------

class RunnerTests(unittest.TestCase):
    def test_runner_produces_a_scored_walk_forward(self):
        universe = {}
        for s, phase in (("AAA-USD", 0.0), ("BBB-USD", 1.5), ("CCC-USD", 3.0)):
            inst = instrument(s)
            closes = [100 + math.sin(i / 9 + phase) * 8 for i in range(200)]
            universe[s] = series(inst, closes)

        run = run_cross_sectional_backtest(
            MeanReversionAgent, universe, asset_class=CRYPTO,
            starting_cash="100000", walk_forward=True,
            train_size=40, test_size=40, embargo=5,
            strategy_kwargs={"lookback": 20},
        )

        self.assertIsInstance(run, CrossSectionalRun)
        self.assertEqual(run.n_bars, 200)
        self.assertIsNotNone(run.walk_forward)
        self.assertGreaterEqual(run.walk_forward.n_folds, 2)
        # Same Scorecard shape the promotion gate consumes.
        self.assertIsNotNone(run.result.scorecard)


if __name__ == "__main__":
    unittest.main()
