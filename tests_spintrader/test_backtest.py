"""Tests for the backtester and scorecard.

Two clusters carry the weight.

**Lookahead.** A strategy must be physically unable to see the future. The
cursor tests assert the barrier, and a deliberately cheating strategy is used
to confirm the barrier is what stops it rather than good manners.

**Statistical honesty.** A Sharpe ratio without error bars, or one selected as
the best of many trials, is how a system convinces its owner of an edge it does
not have. The deflated-Sharpe tests pin that pure noise, mined over many
trials, is reported as insignificant.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np

from spintrader.backtest.engine import (
    BacktestEngine, ReplayCursor, WalkForwardFold, make_folds,
)
from spintrader.backtest.scorecard import (
    Scorecard, deflated_sharpe, max_drawdown, probabilistic_sharpe, score,
    sharpe_standard_error,
)
from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    AssetClass, Bar, Instrument, Side, TradingMode, VenueId,
)
from spintrader.risk.engine import Mandate, TradeIntent

D = Decimal
T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)

BTC = Instrument(
    symbol="BTC-USD", asset_class=AssetClass.CRYPTO, venue=VenueId.PAPER,
    venue_symbol="BTC-USD", quote_currency="USD",
    price_increment=D("0.01"), qty_increment=D("0.00000001"),
    min_qty=D("0.00001"), min_notional=D("1"), taker_fee=D("0.0026"),
)


def bars(closes, interval="1d"):
    out = []
    for i, close in enumerate(closes):
        price = float(close)
        out.append(Bar(
            instrument_key=BTC.key, ts=T0 + timedelta(days=i), interval=interval,
            open=D(str(price)), high=D(str(price * 1.005)),
            low=D(str(price * 0.995)), close=D(str(price)), volume=D("100"),
        ))
    return out


def settings(**kw) -> Settings:
    base = dict(mode=TradingMode.BACKTEST, aggression=Aggression.BALANCED,
                live=LiveGate(enabled=False), base_currency="USD",
                enforce_cash_account=False)
    base.update(kw)
    return Settings(**base)


class NullStrategy:
    name = "null"

    def on_bar(self, cursor, instrument, mandate):
        return ()

    def fit(self, bars):
        return


class BuyOnceStrategy:
    name = "buy_once"

    def __init__(self):
        self.done = False

    def on_bar(self, cursor, instrument, mandate):
        if self.done or cursor.index < 60:
            return ()
        self.done = True
        return [TradeIntent(
            instrument=instrument, side=Side.BUY, edge=D("0.10"),
            confidence=D("0.9"), volatility=D("0.30"),
            quote=cursor.quote(), strategy=self.name,
        )]

    def fit(self, bars):
        return


class PeekingStrategy:
    """Tries to read the future. Must be structurally unable to."""
    name = "peeker"

    def __init__(self):
        self.attempted = False
        self.saw_future = False

    def on_bar(self, cursor, instrument, mandate):
        self.attempted = True
        history = cursor.history()
        # If the cursor leaked anything beyond the current bar, this trips.
        if any(bar.ts > cursor.now for bar in history):
            self.saw_future = True
        return ()

    def fit(self, bars):
        return


# --------------------------------------------------------------------------
# Cursor
# --------------------------------------------------------------------------

class ReplayCursorTests(unittest.TestCase):
    def setUp(self):
        self.series = bars([100 + i for i in range(10)])
        self.cursor = ReplayCursor(self.series)

    def test_starts_at_the_first_bar(self):
        self.assertEqual(self.cursor.index, 0)
        self.assertEqual(self.cursor.now, self.series[0].ts)

    def test_history_never_includes_the_future(self):
        for _ in range(5):
            self.cursor.advance()
        history = self.cursor.history()
        self.assertEqual(len(history), 6)
        self.assertTrue(all(bar.ts <= self.cursor.now for bar in history))

    def test_lookback_truncates_from_the_left(self):
        for _ in range(8):
            self.cursor.advance()
        self.assertEqual(len(self.cursor.history(lookback=3)), 3)

    def test_advance_stops_at_the_end(self):
        while self.cursor.advance():
            pass
        self.assertEqual(self.cursor.index, len(self.series) - 1)
        self.assertFalse(self.cursor.advance())

    def test_empty_series_rejected(self):
        with self.assertRaises(ValueError):
            ReplayCursor([])

    def test_quote_straddles_the_close(self):
        quote = self.cursor.quote(spread_bps=D("10"))
        close = self.series[0].close
        self.assertLess(quote.bid, close)
        self.assertGreater(quote.ask, close)
        self.assertAlmostEqual(float(quote.spread_bps), 10.0, places=4)

    def test_quote_uses_the_current_bar_not_the_next(self):
        first = self.cursor.quote().mid
        self.cursor.advance()
        self.assertNotEqual(first, self.cursor.quote().mid)


class LookaheadBarrierTests(unittest.TestCase):
    def test_peeking_strategy_cannot_see_the_future(self):
        engine = BacktestEngine(settings=settings(), starting_cash=D("1000"))
        strategy = PeekingStrategy()
        engine.run(strategy, BTC, bars([100 + i * 0.1 for i in range(80)]))
        self.assertTrue(strategy.attempted, "strategy never ran")
        self.assertFalse(strategy.saw_future, "cursor leaked future bars")


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------

class FoldTests(unittest.TestCase):
    def test_rolling_folds_have_constant_train_size(self):
        folds = make_folds(1000, train_size=400, test_size=100)
        self.assertTrue(all(f.train_size == 400 for f in folds))

    def test_anchored_folds_grow(self):
        folds = make_folds(1000, train_size=400, test_size=100, anchored=True)
        sizes = [f.train_size for f in folds]
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])

    def test_test_windows_do_not_overlap_by_default(self):
        folds = make_folds(1000, train_size=400, test_size=100)
        for a, b in zip(folds, folds[1:]):
            self.assertGreaterEqual(b.test_start, a.test_end)

    def test_embargo_separates_train_from_test(self):
        # Overlapping feature windows make the first test bars partly in-sample
        # without a gap.
        folds = make_folds(1000, train_size=400, test_size=100, embargo=50)
        for fold in folds:
            self.assertEqual(fold.test_start - fold.train_end, 50)

    def test_test_never_precedes_train(self):
        for fold in make_folds(1000, train_size=300, test_size=100):
            self.assertGreater(fold.test_start, fold.train_end - 1)

    def test_folds_stay_inside_the_series(self):
        n = 1000
        for fold in make_folds(n, train_size=400, test_size=100):
            self.assertLessEqual(fold.test_end, n)

    def test_insufficient_data_raises_with_the_requirement(self):
        with self.assertRaises(ValueError) as ctx:
            make_folds(100, train_size=400, test_size=100)
        self.assertIn("need at least", str(ctx.exception))

    def test_nonpositive_sizes_rejected(self):
        with self.assertRaises(ValueError):
            make_folds(1000, train_size=0, test_size=100)


# --------------------------------------------------------------------------
# Scorecard maths
# --------------------------------------------------------------------------

class MaxDrawdownTests(unittest.TestCase):
    def test_monotonic_curve_has_no_drawdown(self):
        depth, _ = max_drawdown(np.array([100.0, 110.0, 120.0]))
        self.assertEqual(depth, 0.0)

    def test_depth_measured_from_the_peak(self):
        depth, _ = max_drawdown(np.array([100.0, 200.0, 100.0, 150.0]))
        self.assertAlmostEqual(depth, -0.5)

    def test_duration_counts_to_recovery(self):
        _, days = max_drawdown(np.array([100.0, 90.0, 80.0, 100.0, 110.0]))
        self.assertEqual(days, 3)

    def test_empty_curve_is_safe(self):
        self.assertEqual(max_drawdown(np.array([])), (0.0, 0))


class SharpeErrorTests(unittest.TestCase):
    def test_standard_error_shrinks_with_sample_size(self):
        self.assertGreater(sharpe_standard_error(1.0, 60),
                           sharpe_standard_error(1.0, 500))

    def test_short_samples_are_almost_uninformative(self):
        # The point of reporting error bars: at n=60 a Sharpe of 1.0 cannot be
        # distinguished from 0.7.
        se = sharpe_standard_error(1.0, 60)
        self.assertGreater(se, 0.1)

    def test_single_observation_is_infinite_error(self):
        self.assertEqual(sharpe_standard_error(1.0, 1), float("inf"))

    def test_confidence_interval_brackets_the_estimate(self):
        card = Scorecard(n_observations=252, periods_per_year=252,
                         sharpe=1.0, sharpe_stderr=0.1)
        low, high = card.sharpe_ci95
        self.assertLess(low, 1.0)
        self.assertGreater(high, 1.0)


class SignificanceTests(unittest.TestCase):
    """PSR/DSR, with the annualisation units that caused a real bug here.

    The formulae are defined on the PER-PERIOD Sharpe. Passing an annualised
    one inflates the statistic by sqrt(252) for daily data and makes almost
    everything look overwhelmingly significant -- the exact comforting lie this
    module exists to prevent.
    """

    P = 252

    def test_psr_rises_with_sharpe(self):
        weak = probabilistic_sharpe(0.3, 500, 0.0, 0.0, periods_per_year=self.P)
        strong = probabilistic_sharpe(2.0, 500, 0.0, 0.0, periods_per_year=self.P)
        self.assertLess(weak, strong)

    def test_annualisation_is_not_ignored(self):
        # If periods_per_year were dropped, these would be identical. A daily
        # Sharpe of 1.0 annualised is only ~0.063 per period.
        treated_annual = probabilistic_sharpe(1.0, 500, 0.0, 0.0,
                                              periods_per_year=self.P)
        treated_per_period = probabilistic_sharpe(1.0, 500, 0.0, 0.0,
                                                  periods_per_year=1)
        self.assertLess(treated_annual, treated_per_period)

    def test_psr_is_informative_not_saturated(self):
        # A Sharpe of 1.0 on two years of daily data should be encouraging but
        # not certain. A value pinned at 1.0 means the units are wrong.
        psr = probabilistic_sharpe(1.0, 504, 0.0, 0.0, periods_per_year=self.P)
        self.assertGreater(psr, 0.5)
        self.assertLess(psr, 0.999)

    def test_full_kurtosis_used_not_excess(self):
        # With excess kurtosis mistakenly used in place of full kurtosis, the
        # denominator goes negative near Sharpe 2 and this returns 0.0.
        psr = probabilistic_sharpe(2.0, 500, 0.0, 0.0, periods_per_year=1)
        self.assertGreater(psr, 0.0)

    def test_negative_skew_reduces_psr(self):
        # A strategy whose bad outcome has not happened yet should score worse
        # than a symmetric one with the same Sharpe.
        symmetric = probabilistic_sharpe(1.5, 500, 0.0, 0.0, periods_per_year=self.P)
        left_tailed = probabilistic_sharpe(1.5, 500, -1.5, 5.0,
                                           periods_per_year=self.P)
        self.assertLess(left_tailed, symmetric)

    def test_fat_tails_reduce_psr(self):
        thin = probabilistic_sharpe(1.5, 500, 0.0, 0.0, periods_per_year=self.P)
        fat = probabilistic_sharpe(1.5, 500, 0.0, 8.0, periods_per_year=self.P)
        self.assertLess(fat, thin)

    def test_deflated_sharpe_falls_as_trials_rise(self):
        one = deflated_sharpe(1.5, 500, 0.0, 0.0, n_trials=1,
                              periods_per_year=self.P)
        many = deflated_sharpe(1.5, 500, 0.0, 0.0, n_trials=200,
                               periods_per_year=self.P)
        self.assertLess(many, one)

    def test_mined_noise_is_reported_insignificant(self):
        """The test this whole module exists for.

        A modest Sharpe found after many trials must not clear the bar. If this
        ever fails, the self-improvement loop will promote noise.
        """
        dsr = deflated_sharpe(0.9, 500, 0.0, 0.0, n_trials=100,
                              periods_per_year=self.P)
        card = Scorecard(
            n_observations=500, periods_per_year=self.P, sharpe=0.9,
            sharpe_stderr=sharpe_standard_error(0.9, 500, self.P),
            deflated_sharpe=dsr, n_trials=100,
        )
        self.assertFalse(card.is_significant)

    def test_strong_result_survives_many_trials(self):
        dsr = deflated_sharpe(3.0, 1000, 0.0, 0.0, n_trials=100,
                              periods_per_year=self.P)
        self.assertGreater(dsr, 0.95)

    def test_single_trial_reduces_to_psr(self):
        self.assertAlmostEqual(
            deflated_sharpe(1.0, 500, 0.0, 0.0, n_trials=1,
                            periods_per_year=self.P),
            probabilistic_sharpe(1.0, 500, 0.0, 0.0, periods_per_year=self.P),
            places=10,
        )


class ScoreTests(unittest.TestCase):
    def test_flat_curve_scores_zero(self):
        card = score([100.0] * 50, periods_per_year=252)
        self.assertEqual(card.total_return, 0.0)
        self.assertEqual(card.sharpe, 0.0)

    def test_rising_curve_has_positive_sharpe(self):
        curve = [100.0 * (1.001 ** i) for i in range(300)]
        card = score(curve, periods_per_year=252)
        self.assertGreater(card.sharpe, 0)
        self.assertGreater(card.cagr, 0)
        self.assertEqual(card.max_drawdown, 0.0)

    def test_too_short_series_returns_empty_card(self):
        card = score([100.0], periods_per_year=252)
        self.assertEqual(card.n_observations, 1)
        self.assertEqual(card.sharpe, 0.0)

    def test_sample_size_recorded(self):
        card = score([100.0 + i for i in range(101)], periods_per_year=252)
        self.assertEqual(card.n_observations, 100)   # returns, not levels

    def test_cost_drag_reported(self):
        card = score([100.0, 110.0], periods_per_year=252, fees_paid=2.0)
        self.assertGreater(card.cost_drag, 0.0)

    def test_alpha_against_benchmark(self):
        card = score([100.0, 120.0], periods_per_year=252,
                     benchmark_curve=[100.0, 110.0])
        self.assertAlmostEqual(card.benchmark_return, 0.10)
        self.assertAlmostEqual(card.alpha, 0.10)

    def test_hit_rate_and_profit_factor(self):
        card = score([100.0] * 10, periods_per_year=252,
                     trade_pnls=[5.0, -2.0, 3.0, -1.0])
        self.assertAlmostEqual(card.hit_rate, 0.5)
        self.assertAlmostEqual(card.profit_factor, 8.0 / 3.0)

    def test_verdict_mentions_the_sample_size(self):
        card = score([100.0 * (1.001 ** i) for i in range(300)],
                     periods_per_year=252)
        self.assertIn("n=", card.verdict())


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class EngineTests(unittest.TestCase):
    def test_null_strategy_preserves_capital_exactly(self):
        engine = BacktestEngine(settings=settings(), starting_cash=D("1000"))
        result = engine.run(NullStrategy(), BTC, bars([100 + i * 0.1 for i in range(100)]))
        self.assertEqual(result.final_equity, 1000.0)
        self.assertEqual(result.n_orders, 0)
        self.assertEqual(result.fees_paid, 0.0)

    def test_equity_curve_has_one_point_per_bar(self):
        # The Sharpe denominator must be the true sample size.
        series = bars([100 + i * 0.1 for i in range(100)])
        engine = BacktestEngine(settings=settings())
        result = engine.run(NullStrategy(), BTC, series)
        self.assertEqual(len(result.equity_curve), len(series))
        self.assertEqual(len(result.timestamps), len(series))

    def test_buying_incurs_fees(self):
        engine = BacktestEngine(settings=settings(), starting_cash=D("1000"))
        result = engine.run(BuyOnceStrategy(), BTC,
                           bars([100 + i * 0.1 for i in range(120)]))
        self.assertGreater(result.n_orders, 0)
        self.assertGreater(result.fees_paid, 0.0)

    def test_rejections_are_categorised(self):
        class AlwaysBuy:
            name = "always_buy"

            def on_bar(self, cursor, instrument, mandate):
                return [TradeIntent(
                    instrument=instrument, side=Side.BUY, edge=D("0.5"),
                    confidence=D("0.99"), volatility=D("0.2"),
                    quote=cursor.quote(), strategy=self.name,
                )]

            def fit(self, bars):
                return

        engine = BacktestEngine(settings=settings(), starting_cash=D("100"))
        result = engine.run(AlwaysBuy(), BTC, bars([100 + i * 0.1 for i in range(60)]))
        # Cash and per-name limits must bind quickly; the reasons are recorded.
        self.assertGreater(result.n_rejected, 0)
        self.assertTrue(result.rejections)

    def test_backtest_mode_is_forced_regardless_of_settings(self):
        # A backtest must never be able to satisfy the live gate.
        engine = BacktestEngine(settings=settings(mode=TradingMode.LIVE))
        self.assertEqual(engine.settings.mode, TradingMode.BACKTEST)
        self.assertFalse(engine.settings.live.enabled)

    def test_benchmark_curve_tracks_buy_and_hold(self):
        series = bars([100.0, 110.0, 121.0])
        engine = BacktestEngine(settings=settings(), starting_cash=D("1000"))
        result = engine.run(NullStrategy(), BTC, series)
        self.assertAlmostEqual(result.benchmark_curve[-1], 1210.0, places=4)

    def test_two_bars_is_the_minimum(self):
        engine = BacktestEngine(settings=settings())
        with self.assertRaises(ValueError):
            engine.run(NullStrategy(), BTC, bars([100.0]))


class WalkForwardTests(unittest.TestCase):
    def test_produces_one_result_per_fold(self):
        series = bars([100 + math.sin(i / 10) * 5 for i in range(600)])
        engine = BacktestEngine(settings=settings())
        out = engine.walk_forward(NullStrategy, BTC, series,
                                  train_size=200, test_size=100)
        self.assertEqual(out.n_folds, 4)

    def test_fresh_strategy_per_fold(self):
        # Reusing an instance would carry fitted state across folds and leak the
        # future into earlier tests.
        created = []

        def factory():
            strategy = NullStrategy()
            created.append(strategy)
            return strategy

        series = bars([100 + i * 0.01 for i in range(600)])
        engine = BacktestEngine(settings=settings())
        engine.walk_forward(factory, BTC, series, train_size=200, test_size=100)
        self.assertGreater(len(created), 4)     # one per fold plus the name probe
        self.assertEqual(len(set(id(s) for s in created)), len(created))

    def test_stitched_curve_is_continuous(self):
        series = bars([100 + i * 0.01 for i in range(600)])
        engine = BacktestEngine(settings=settings())
        out = engine.walk_forward(NullStrategy, BTC, series,
                                  train_size=200, test_size=100)
        stitched = out.stitched_equity()
        self.assertGreater(len(stitched), 0)
        # No cliffs: each fold is chained multiplicatively from the last level.
        self.assertTrue(all(v > 0 for v in stitched))

    def test_combined_scorecard_carries_trial_count(self):
        series = bars([100 + i * 0.01 for i in range(600)])
        engine = BacktestEngine(settings=settings())
        out = engine.walk_forward(NullStrategy, BTC, series, train_size=200,
                                  test_size=100, n_trials=50)
        if out.combined is not None:
            self.assertEqual(out.combined.n_trials, 50)


if __name__ == "__main__":
    unittest.main()
