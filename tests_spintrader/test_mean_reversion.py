"""Tests for the mean-reversion persona. No network.

The behaviours that matter: it buys a dip (not a rally), it forecasts the gap
back to the mean as its edge, it exits on reversion or a stop, and -- like the
baseline -- it evaluates exits before any guard that could suppress them and only
ever requests its trailing window.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.agents.personas.mean_reversion import MeanReversionAgent
from spintrader.backtest.engine import ReplayCursor
from spintrader.backtest.runner import backtest_instrument
from spintrader.core.types import AssetClass, Bar, Side
from spintrader.risk.engine import Mandate

D = Decimal
UTC = timezone.utc
INST = backtest_instrument("X-USD", AssetClass.CRYPTO)


def bars(closes, interval="1d"):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for i, c in enumerate(closes):
        c = float(c)
        out.append(Bar(instrument_key=INST.key, ts=t0 + timedelta(days=i + 1),
                       interval=interval, open=D(str(c)), high=D(str(c + 0.5)),
                       low=D(str(c - 0.5)), close=D(str(c)), volume=D("1")))
    return out


def agent(**kw):
    params = dict(lookback=20, entry_z="1.5", exit_z="0.0", vol_ceiling="0.40",
                 interval="1d", continuous=False, min_annual_vol="0.001")
    params.update(kw)
    return MeanReversionAgent(**params)


def run_series(a, closes):
    """Feed closes bar by bar; return (index, side, strategy) for each intent."""
    series = bars(closes)
    cursor = ReplayCursor(series)
    events = []
    while True:
        for intent in a.on_bar(cursor, INST, Mandate.open_mandate([INST.key], hours=10**6)):
            events.append((cursor.index, intent.side, intent.strategy))
        if not cursor.advance():
            break
    return events


class SignalTests(unittest.TestCase):
    def test_warmup_is_lookback_plus_two(self):
        self.assertEqual(agent(lookback=20).warmup_bars, 22)

    def test_buys_a_dip(self):
        # Flat at 100, then a drop well below the trailing mean.
        events = run_series(agent(), [100.0] * 30 + [95.0])
        buys = [e for e in events if e[1] is Side.BUY]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0][0], 30)                 # the dip bar

    def test_does_not_buy_a_rally(self):
        # Price above the mean -> nothing to fade for a long-only reverter.
        events = run_series(agent(), [100.0] * 30 + [105.0])
        self.assertEqual([e for e in events if e[1] is Side.BUY], [])

    def test_flat_market_never_trades(self):
        self.assertEqual(run_series(agent(), [100.0] * 40), [])

    def test_exits_on_reversion(self):
        events = run_series(agent(), [100.0] * 30 + [95.0, 98.0, 101.0])
        sides = [e[1] for e in events]
        self.assertIn(Side.BUY, sides)
        self.assertIn(Side.SELL, sides)
        # The sell is an exit, tagged with the reason.
        sell = next(e for e in events if e[1] is Side.SELL)
        self.assertTrue(sell[2].startswith("mean_reversion_v1:"))

    def test_edge_is_the_gap_to_the_mean(self):
        a = agent()
        cursor = ReplayCursor(bars([100.0] * 30 + [95.0]))
        while cursor.advance():
            pass
        intents = a.on_bar(cursor, INST, Mandate.open_mandate([INST.key], hours=10**6))
        self.assertEqual(len(intents), 1)
        # Gap back to ~99.75 mean from 95 is ~5%, clipped to the 6% ceiling.
        self.assertGreater(intents[0].edge, D("0.03"))
        self.assertLessEqual(intents[0].edge, D("0.06"))

    def test_fit_resets_state(self):
        a = agent()
        run_series(a, [100.0] * 30 + [95.0])
        self.assertTrue(a.is_long)
        a.fit([])
        self.assertFalse(a.is_long)

    def test_only_requests_the_warmup_window(self):
        # A recording cursor: the agent must never ask for more than warmup_bars.
        class RecordingCursor(ReplayCursor):
            def __init__(self, b):
                super().__init__(b)
                self.max_lookback = 0

            def history(self, lookback=None):
                if lookback is not None:
                    self.max_lookback = max(self.max_lookback, lookback)
                return super().history(lookback)

        a = agent()
        cur = RecordingCursor(bars([100.0] * 30 + [95.0]))
        while True:
            a.on_bar(cur, INST, Mandate.open_mandate([INST.key], hours=10**6))
            if not cur.advance():
                break
        self.assertLessEqual(cur.max_lookback, a.warmup_bars)


if __name__ == "__main__":
    unittest.main()
