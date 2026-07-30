"""Tests for the baseline trend persona.

Bias throughout: where a behaviour could plausibly be loosened by a later
refactor in a way that makes the persona trade *more*, there is a test pinning
it shut. The failure mode that costs money is a baseline that quietly starts
taking risk it was never sized for -- entering before warm-up, entering into a
volatility spike, or re-entering every bar and grinding the account down in
fees. Each of those has a test below.

The causality test is the important one. Every other property can be inspected
by reading the code; lookahead cannot, because it is a property of how the code
is *called*.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.agents.personas.baseline_trend import (
    BaselineTrendAgent, TrendReading,
)
from spintrader.backtest.engine import ReplayCursor
from spintrader.core.types import (
    AssetClass, Bar, Instrument, Quote, Side, VenueId,
)
from spintrader.risk.engine import Mandate

D = Decimal
T0 = datetime(2024, 1, 1, 21, 0, tzinfo=timezone.utc)

SPY = Instrument(
    symbol="SPY", asset_class=AssetClass.EQUITY, venue=VenueId.PAPER,
    venue_symbol="SPY", quote_currency="USD",
    price_increment=D("0.01"), qty_increment=D("0.0001"),
    min_qty=D("0"), min_notional=D("1"), taker_fee=D("0.00005"),
)


def bars(closes, interval="1d"):
    out = []
    for i, close in enumerate(closes):
        price = float(close)
        out.append(Bar(
            instrument_key=SPY.key, ts=T0 + timedelta(days=i), interval=interval,
            open=D(str(price)), high=D(str(price * 1.002)),
            low=D(str(price * 0.998)), close=D(str(price)), volume=D("1000"),
        ))
    return out


def agent(**kw) -> BaselineTrendAgent:
    """Small windows so tests stay readable; semantics are window-agnostic."""
    params = dict(fast_window=5, slow_window=20, vol_window=5, interval="1d",
                  continuous=False)
    params.update(kw)
    return BaselineTrendAgent(**params)


def mandate(**kw) -> Mandate:
    params = dict(issued_at=T0, expires_at=T0 + timedelta(days=3650),
                  permitted=frozenset({SPY.key}))
    params.update(kw)
    return Mandate(**params)


def drive(strategy, series, mand=None):
    """Feed a whole series through the persona, collecting emitted intents."""
    mand = mand or mandate()
    cursor = ReplayCursor(series)
    emitted = []
    while True:
        for intent in strategy.on_bar(cursor, SPY, mand) or ():
            emitted.append((cursor.index, intent))
        if not cursor.advance():
            break
    return emitted


def rising(n=60, start=100.0, step=0.004, jitter=0.0015):
    """A gently rising series with enough wobble to have non-zero volatility.

    A perfectly straight line has zero realised volatility, which the persona
    refuses to size against -- so a test series must actually move.
    """
    out, price = [], start
    for i in range(n):
        wiggle = jitter if i % 2 else -jitter
        price *= (1.0 + step + wiggle)
        out.append(price)
    return out


def falling(n=60, start=100.0, step=0.004, jitter=0.0015):
    out, price = [], start
    for i in range(n):
        wiggle = jitter if i % 2 else -jitter
        price *= (1.0 - step + wiggle)
        out.append(price)
    return out


class ConstructionTests(unittest.TestCase):

    def test_slow_window_must_exceed_fast(self):
        with self.assertRaises(ValueError):
            BaselineTrendAgent(fast_window=20, slow_window=20)
        with self.assertRaises(ValueError):
            BaselineTrendAgent(fast_window=30, slow_window=20)

    def test_degenerate_windows_rejected(self):
        with self.assertRaises(ValueError):
            BaselineTrendAgent(fast_window=1, slow_window=10)
        with self.assertRaises(ValueError):
            BaselineTrendAgent(fast_window=5, slow_window=10, vol_window=1)

    def test_unknown_interval_fails_at_construction_not_mid_run(self):
        # An unknown interval silently defaulting to a wrong annualisation
        # factor would mis-scale every volatility for the whole run.
        with self.assertRaises(ValueError):
            BaselineTrendAgent(interval="3d")

    def test_equity_and_crypto_annualisation_differ(self):
        equity = BaselineTrendAgent(interval="1d", continuous=False)
        crypto = BaselineTrendAgent(interval="1d", continuous=True)
        self.assertLess(equity._ann_scale, crypto._ann_scale)

    def test_warmup_covers_the_longest_window(self):
        strategy = agent(fast_window=5, slow_window=20, vol_window=50)
        self.assertGreaterEqual(strategy.warmup_bars, 50)

    def test_describe_round_trips_config(self):
        strategy = agent(stop_pct="0.07")
        described = strategy.describe()
        self.assertEqual(described["stop_pct"], "0.07")
        self.assertEqual(described["warmup_bars"], strategy.warmup_bars)


class WarmupTests(unittest.TestCase):

    def test_silent_before_warmup(self):
        strategy = agent()
        series = bars(rising(strategy.warmup_bars - 1))
        self.assertEqual(drive(strategy, series), [])
        self.assertFalse(strategy.is_long)

    def test_no_reading_recorded_before_warmup(self):
        strategy = agent()
        drive(strategy, bars(rising(strategy.warmup_bars - 1)))
        self.assertIsNone(strategy.last_reading)

    def test_first_possible_entry_is_at_warmup(self):
        strategy = agent()
        series = bars(rising(80))
        emitted = drive(strategy, series)
        self.assertTrue(emitted)
        first_index = emitted[0][0]
        self.assertGreaterEqual(first_index, strategy.warmup_bars - 1)


class EntryTests(unittest.TestCase):

    def test_enters_a_clean_uptrend(self):
        strategy = agent()
        emitted = drive(strategy, bars(rising(80)))
        self.assertTrue(emitted)
        _, first = emitted[0]
        self.assertIs(first.side, Side.BUY)
        self.assertGreater(first.confidence, D("0"))
        self.assertGreater(first.volatility, D("0"))

    def test_never_enters_a_downtrend(self):
        strategy = agent()
        emitted = drive(strategy, bars(falling(80)))
        buys = [i for _, i in emitted if i.side is Side.BUY]
        self.assertEqual(buys, [])
        self.assertFalse(strategy.is_long)

    def test_volatility_ceiling_blocks_entry(self):
        # Same rising direction, but violent enough to breach the ceiling.
        wild = []
        price = 100.0
        for i in range(80):
            price *= (1.10 if i % 2 else 0.94)
            wild.append(price)
        strategy = agent(vol_ceiling="0.40")
        emitted = drive(strategy, bars(wild))
        self.assertEqual([i for _, i in emitted if i.side is Side.BUY], [])

    def test_flat_series_produces_no_intent(self):
        # Zero volatility would make Kelly and vol-targeting divide by zero and
        # demand an unbounded position.
        strategy = agent()
        emitted = drive(strategy, bars([100.0] * 80))
        self.assertEqual(emitted, [])

    def test_one_entry_per_transition_not_per_bar(self):
        strategy = agent()
        emitted = drive(strategy, bars(rising(120)))
        buys = [i for _, i in emitted if i.side is Side.BUY]
        # A persona that re-entered every bullish bar would emit ~100 buys and
        # grind the account away in fees.
        self.assertLessEqual(len(buys), 3)
        self.assertGreaterEqual(len(buys), 1)

    def test_edge_is_clipped_into_the_configured_band(self):
        strategy = agent(edge_floor="0.01", edge_ceiling="0.03")
        emitted = drive(strategy, bars(rising(120, step=0.02)))
        for _, intent in emitted:
            if intent.side is Side.BUY:
                self.assertGreaterEqual(intent.edge, D("0.01"))
                self.assertLessEqual(intent.edge, D("0.03"))

    def test_confidence_stays_a_valid_probability(self):
        strategy = agent()
        for series in (rising(120), falling(120), rising(120, step=0.02)):
            strategy.reset()
            for _, intent in drive(strategy, bars(series)):
                self.assertGreaterEqual(intent.confidence, D("0"))
                self.assertLessEqual(intent.confidence, D("1"))

    def test_marginal_signal_lands_below_the_moderate_floor(self):
        # The persona defers filtering to the risk engine's min_confidence
        # rather than duplicating a threshold. That only works if a weak signal
        # actually scores below it.
        strategy = agent()
        weak = strategy._confidence(trend_strength=0.0, annual_vol=0.40)
        self.assertLess(weak, 0.62)


class MandateTests(unittest.TestCase):

    def test_instrument_outside_the_mandate_is_not_traded(self):
        strategy = agent()
        emitted = drive(strategy, bars(rising(80)),
                        mand=mandate(permitted=frozenset({"paper:QQQ"})))
        self.assertEqual(emitted, [])

    def test_empty_mandate_permits_nothing(self):
        strategy = agent()
        emitted = drive(strategy, bars(rising(80)),
                        mand=mandate(permitted=frozenset()))
        self.assertEqual(emitted, [])

    def test_short_bias_suppresses_a_long_only_persona(self):
        strategy = agent()
        emitted = drive(strategy, bars(rising(80)),
                        mand=mandate(directional_bias={SPY.key: D("-1")}))
        self.assertEqual(emitted, [])


class ExitTests(unittest.TestCase):

    def _long_then(self, tail):
        strategy = agent()
        series = bars(rising(60) + tail)
        emitted = drive(strategy, series)
        return strategy, emitted

    def test_exits_on_trend_break(self):
        strategy, emitted = self._long_then(falling(40))
        sells = [i for _, i in emitted if i.side is Side.SELL]
        self.assertTrue(sells, "a persona that cannot exit cannot limit a loss")
        self.assertIn("trend_break", " ".join(i.strategy for i in sells))
        self.assertFalse(strategy.is_long)

    def test_exit_carries_full_conviction(self):
        # min_confidence must never be able to veto a stop.
        _, emitted = self._long_then(falling(40))
        for _, intent in emitted:
            if intent.side is Side.SELL:
                self.assertEqual(intent.confidence, D("1"))

    def test_exit_edge_survives_kelly_sizing(self):
        # Defence in depth: if the engine's reduction path were bypassed, the
        # exit must still be sized large enough to close the whole position.
        _, emitted = self._long_then(falling(40))
        sells = [i for _, i in emitted if i.side is Side.SELL]
        self.assertTrue(sells)
        for intent in sells:
            self.assertGreaterEqual(intent.edge, intent.volatility ** 2)

    def test_pure_uptrend_never_exits(self):
        strategy = agent()
        emitted = drive(strategy, bars(rising(120)))
        self.assertTrue(any(i.side is Side.BUY for _, i in emitted))
        self.assertEqual([i for _, i in emitted if i.side is Side.SELL], [])
        self.assertTrue(strategy.is_long)

    def test_retracement_from_a_high_triggers_an_exit(self):
        strategy = agent(trail_pct="0.05", stop_pct="0.50")
        climb = rising(60, step=0.01)
        peak = climb[-1]
        retrace = [peak * (1.0 - 0.012 * i) for i in range(1, 8)]
        emitted = drive(strategy, bars(climb + retrace))
        sells = [i for _, i in emitted if i.side is Side.SELL]
        self.assertTrue(sells, "a 8% retracement from the high must close the "
                               "position under a 5% trailing stop")


class ExitReasonTests(unittest.TestCase):
    """Each exit trigger in isolation, and their precedence.

    Driven through ``_exit_reason`` rather than a price series because the four
    triggers overlap: any drop deep enough to break a hard stop usually breaks
    the trend average too, so a series-level test cannot isolate them and would
    pass while three of the four triggers were dead code.
    """

    @staticmethod
    def reading(close, *, above_slow=True, vol_ok=True, annual_vol=0.16):
        return TrendReading(
            close=close, fast=close, slow=close * 0.9,
            trend_strength=0.02, annual_vol=annual_vol,
            above_slow=above_slow, vol_ok=vol_ok, bullish=above_slow and vol_ok,
            edge=0.02, confidence=0.8,
        )

    def armed(self, entry=100.0, peak=None, **kw):
        strategy = agent(**kw)
        strategy._long = True
        strategy._entry_mark = entry
        strategy._peak_mark = peak if peak is not None else entry
        return strategy

    def test_no_exit_when_nothing_is_breached(self):
        strategy = self.armed(stop_pct="0.05", trail_pct="0.08")
        self.assertIsNone(strategy._exit_reason(self.reading(101.0)))

    def test_trend_break(self):
        strategy = self.armed()
        self.assertEqual(
            strategy._exit_reason(self.reading(99.0, above_slow=False)),
            "trend_break",
        )

    def test_vol_spike(self):
        strategy = self.armed()
        self.assertEqual(
            strategy._exit_reason(self.reading(101.0, vol_ok=False)),
            "vol_spike",
        )

    def test_hard_stop(self):
        strategy = self.armed(entry=100.0, peak=100.0,
                              stop_pct="0.05", trail_pct="0.50")
        self.assertIsNone(strategy._exit_reason(self.reading(95.5)))
        self.assertEqual(strategy._exit_reason(self.reading(95.0)), "hard_stop")
        self.assertEqual(strategy._exit_reason(self.reading(90.0)), "hard_stop")

    def test_trailing_stop(self):
        strategy = self.armed(entry=100.0, peak=140.0,
                              stop_pct="0.50", trail_pct="0.08")
        self.assertIsNone(strategy._exit_reason(self.reading(129.0)))
        self.assertEqual(
            strategy._exit_reason(self.reading(128.8)), "trailing_stop",
        )

    def test_trailing_stop_can_fire_while_still_in_profit(self):
        # This is the whole point of a trailing stop: it protects gains, not
        # just capital. A version that only fired below the entry price would
        # give back every unrealised gain on the way down.
        strategy = self.armed(entry=100.0, peak=140.0,
                              stop_pct="0.05", trail_pct="0.08")
        reason = strategy._exit_reason(self.reading(125.0))
        self.assertEqual(reason, "trailing_stop")

    def test_trend_break_takes_precedence_over_stops(self):
        strategy = self.armed(entry=100.0, peak=140.0,
                              stop_pct="0.05", trail_pct="0.08")
        self.assertEqual(
            strategy._exit_reason(self.reading(80.0, above_slow=False)),
            "trend_break",
        )

    def test_a_volatility_collapse_does_not_trap_the_position(self):
        # Regression. The minimum-volatility guard exists to stop an unbounded
        # position being *opened* when the denominator in Kelly goes to zero. It
        # was evaluated before the exit branch, so a quiet, orderly decline --
        # exactly the shape a slow bleed takes -- suppressed the exit entirely
        # and the position could not be closed at all.
        strategy = agent(trail_pct="0.05", stop_pct="0.50", min_annual_vol="0.01")
        climb = rising(60, step=0.01)
        peak = climb[-1]
        # A perfectly linear decline has near-zero realised volatility.
        bleed = [peak * (1.0 - 0.012 * i) for i in range(1, 10)]
        emitted = drive(strategy, bars(climb + bleed))
        sells = [i for _, i in emitted if i.side is Side.SELL]
        self.assertTrue(sells, "an orderly decline suppressed the exit")
        self.assertFalse(strategy.is_long)

    def test_exit_volatility_is_floored_not_zero(self):
        strategy = self.armed(entry=100.0, peak=100.0, min_annual_vol="0.02")
        intents = []

        class FakeCursor:
            @staticmethod
            def history(lookback=None):
                return bars(rising(strategy.warmup_bars + 5))[-(lookback or 0):]

            @staticmethod
            def quote(spread=None):
                return Quote(SPY.key, T0, bid=D("99"), ask=D("101"))

        intent = strategy._exit_intent(
            SPY, FakeCursor(), max(D("0"), strategy.min_annual_vol), "hard_stop",
        )
        self.assertGreater(intent.volatility, D("0"))

    def test_peak_tracks_upward_only(self):
        strategy = agent(trail_pct="0.08")
        drive(strategy, bars(rising(80, step=0.01)))
        self.assertTrue(strategy.is_long)
        peak = strategy._peak_mark
        self.assertIsNotNone(peak)
        self.assertGreaterEqual(peak, strategy._entry_mark)

    def test_no_sell_emitted_while_flat(self):
        # A sell with no position is an attempt to open a short, which the risk
        # engine rejects under a cash account. Emitting it pollutes the
        # rejection histogram and hides real problems.
        strategy = agent()
        emitted = drive(strategy, bars(falling(80)))
        self.assertEqual([i for _, i in emitted if i.side is Side.SELL], [])

    def test_intents_alternate_buy_sell(self):
        strategy = agent()
        series = bars(rising(60) + falling(40) + rising(60) + falling(40))
        sides = [i.side for _, i in drive(strategy, series)]
        self.assertTrue(sides)
        for previous, current in zip(sides, sides[1:]):
            self.assertIsNot(previous, current,
                             "two entries or two exits in a row means the "
                             "persona lost track of its own exposure")


class StateTests(unittest.TestCase):

    def test_fit_is_a_no_op_that_resets(self):
        strategy = agent()
        drive(strategy, bars(rising(80)))
        self.assertTrue(strategy.is_long)
        strategy.fit(bars(rising(80)))
        self.assertFalse(strategy.is_long)
        self.assertIsNone(strategy.last_reading)

    def test_reset_clears_stop_levels(self):
        strategy = agent()
        drive(strategy, bars(rising(80)))
        strategy.reset()
        self.assertIsNone(strategy._entry_mark)
        self.assertIsNone(strategy._peak_mark)

    def test_reading_is_recorded_for_audit(self):
        strategy = agent()
        drive(strategy, bars(rising(80)))
        reading = strategy.last_reading
        self.assertIsInstance(reading, TrendReading)
        self.assertGreater(reading.annual_vol, 0.0)
        self.assertAlmostEqual(
            reading.trend_strength, reading.fast / reading.slow - 1.0, places=12,
        )


class CausalityTests(unittest.TestCase):
    """The property that cannot be verified by reading the code."""

    def _decisions(self, series):
        strategy = agent()
        cursor = ReplayCursor(series)
        mand = mandate()
        out = []
        while True:
            out.append(tuple(
                (i.side.value, str(i.edge), str(i.confidence))
                for i in strategy.on_bar(cursor, SPY, mand) or ()
            ))
            if not cursor.advance():
                break
        return out

    def test_truncating_the_future_never_changes_the_past(self):
        series = bars(rising(60) + falling(30) + rising(50) + falling(30))
        full = self._decisions(series)
        for cut in (40, 70, 110, len(series) - 1):
            partial = self._decisions(series[:cut])
            self.assertEqual(
                full[:len(partial)], partial,
                f"decisions changed when the series was cut at {cut}: the "
                f"persona is reading data it should not have",
            )

    def test_only_the_trailing_window_is_requested(self):
        # A persona that asked for the whole series could not be audited for
        # lookahead by inspection, and would make every run O(n^2).
        strategy = agent()
        series = bars(rising(200))
        cursor = ReplayCursor(series)
        seen: list[int | None] = []
        original = cursor.history

        def spy(lookback=None):
            seen.append(lookback)
            return original(lookback)

        cursor.history = spy  # type: ignore[method-assign]
        for _ in range(120):
            strategy.on_bar(cursor, SPY, mandate())
            if not cursor.advance():
                break
        self.assertTrue(seen)
        self.assertTrue(
            all(value == strategy.warmup_bars for value in seen),
            f"expected every history() call to request exactly "
            f"{strategy.warmup_bars} bars, got {sorted(set(seen))}",
        )


if __name__ == "__main__":
    unittest.main()
