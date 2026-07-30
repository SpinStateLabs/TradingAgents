"""Tests for the tail-fragility gauge.

The gauge makes a strong promise -- causal, bounded, and monotone in the things
that actually mean fragility -- and each of those is a property a lookahead or
scaling bug would silently break without raising. The tests target the promises
directly:

* **Monotonicity.** A fat-tailed, high-vol, deep-drawdown series must score
  higher than a calm one, or the gauge measures nothing.
* **Causality / truncation invariance.** A past score must not move when future
  bars are removed, exactly as :func:`spintrader.quant.features.assert_causal`
  checks for the rolling statistics.
* **Graceful EVT degradation.** A short or degenerate window must yield
  ``EVTFit(ok=False)``, not an exception and not a fabricated tail.
* **Boundedness.** The score is in ``[0, 1]`` on adversarial inputs.

Synthetic series are generated with known properties so behaviour is checked
against ground truth, not against the gauge's own output.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import numpy as np

from spintrader.quant.tail import (
    EVTFit, TailRiskGauge, combine_regime_risk, evaluate_false_alarms,
    forward_drawdown,
)


def _closes_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    """Price path from log returns, seeded at ``start``."""
    return start * np.exp(np.cumsum(np.concatenate([[0.0], returns])))


def _calm_series(seed: int = 0, n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _closes_from_returns(rng.normal(0.0004, 0.006, n))


def _fragile_series(seed: int = 1, n: int = 400) -> np.ndarray:
    """High vol, left-skewed, fat-tailed, and ending in a deep drawdown."""
    rng = np.random.default_rng(seed)
    # Student-t gives fat tails; a negative shift and a late cluster of large
    # losses supply the left skew and the drawdown.
    body = rng.standard_t(3, n) * 0.02 - 0.001
    body[-30:] -= np.abs(rng.normal(0.0, 0.05, 30))     # a late liquidation
    return _closes_from_returns(body)


class MonotonicityTests(unittest.TestCase):
    """Fragile conditions must outscore calm ones, component by component."""

    def setUp(self):
        self.gauge = TailRiskGauge()

    def test_fragile_scores_higher_than_calm(self):
        calm = self.gauge.black_swan_score(_calm_series())
        fragile = self.gauge.black_swan_score(_fragile_series())
        # The load-bearing promise is separation, not an absolute level: the
        # gauge is deliberately conservative (a synthetic "fragile" tape is only
        # elevated, not a five-alarm crisis), so we assert a clear gap rather
        # than a magic 0.5 the calibration was never pinned to.
        self.assertGreater(fragile, calm)
        self.assertGreater(fragile - calm, 0.15)
        self.assertGreater(fragile, 0.4)
        self.assertLess(calm, 0.35)

    def test_holds_across_many_seeds(self):
        for seed in range(6):
            calm = self.gauge.black_swan_score(_calm_series(seed=seed))
            fragile = self.gauge.black_swan_score(_fragile_series(seed=seed + 100))
            self.assertGreater(
                fragile, calm, f"fragile <= calm on seed {seed}",
            )

    def test_deeper_drawdown_raises_the_drawdown_component(self):
        rng = np.random.default_rng(7)
        base = _closes_from_returns(rng.normal(0.0, 0.01, 400))
        # Force a fresh 20% slide onto the tail of an otherwise identical path.
        slid = base.copy()
        slid[-1] = slid[-1] * 0.8
        r_base = self.gauge.reading(base)
        r_slid = self.gauge.reading(slid)
        self.assertLess(r_slid.drawdown, r_base.drawdown)             # more negative
        self.assertGreaterEqual(
            r_slid.subscores["drawdown"], r_base.subscores["drawdown"],
        )

    def test_rising_vol_ratio_raises_its_component(self):
        rng = np.random.default_rng(9)
        quiet = rng.normal(0.0, 0.004, 300)
        loud = rng.normal(0.0, 0.02, 40)                 # recent vol jump
        closes = _closes_from_returns(np.concatenate([quiet, loud]))
        reading = self.gauge.reading(closes)
        self.assertGreater(reading.vol_ratio, 1.0)
        self.assertGreater(reading.subscores["vol_ratio"], 0.0)


class CausalityTests(unittest.TestCase):
    """Truncating the future must not change a past score."""

    def test_series_is_truncation_invariant(self):
        gauge = TailRiskGauge(window=120)
        closes = _fragile_series(seed=3, n=500)
        full = gauge.series(closes)
        for cut in (200, 350, 499):
            partial = gauge.series(closes[:cut])
            np.testing.assert_allclose(
                full[:cut], partial, rtol=1e-9, atol=1e-9,
                err_msg=f"score at t<{cut} moved when the future was removed",
            )

    def test_reading_uses_only_the_trailing_window(self):
        # Appending future bars beyond the window must not change today's score
        # when today is the last bar of both inputs -- but here we assert the
        # inverse: the reading depends only on the last `window` closes.
        gauge = TailRiskGauge(window=150)
        closes = _fragile_series(seed=4, n=400)
        tail_only = gauge.black_swan_score(closes[-150:])
        full = gauge.black_swan_score(closes)
        self.assertAlmostEqual(tail_only, full, places=9)


class EVTDegradationTests(unittest.TestCase):
    """The GPD fit must fail soft on inputs it cannot honestly fit."""

    def setUp(self):
        self.gauge = TailRiskGauge()

    def test_short_window_does_not_fit(self):
        fit = self.gauge.fit_evt(np.random.default_rng(0).normal(0, 0.01, 10))
        self.assertIsInstance(fit, EVTFit)
        self.assertFalse(fit.ok)

    def test_constant_series_does_not_fit_or_raise(self):
        fit = self.gauge.fit_evt(np.zeros(300))
        self.assertFalse(fit.ok)
        # And the whole reading still succeeds and stays benign-ish on the tail.
        reading = self.gauge.reading(np.full(300, 100.0))
        self.assertFalse(reading.evt.ok)
        self.assertNotIn("evt", reading.subscores)

    def test_one_sided_gains_have_no_loss_tail(self):
        # Monotonically rising prices: no losses to exceed the threshold.
        closes = _closes_from_returns(np.full(300, 0.002))
        fit = self.gauge.fit_evt(np.full(299, 0.002))
        self.assertFalse(fit.ok)
        self.assertTrue(0.0 <= self.gauge.black_swan_score(closes) <= 1.0)

    def test_fat_tailed_series_does_fit_and_reads_heavy(self):
        rng = np.random.default_rng(11)
        returns = rng.standard_t(2.5, 600) * 0.02
        fit = self.gauge.fit_evt(returns)
        self.assertTrue(fit.ok)
        self.assertGreater(fit.n_exceedances, self.gauge.min_exceedances)
        self.assertTrue(np.isfinite(fit.cvar))
        self.assertGreaterEqual(fit.cvar, fit.var)


class BoundednessTests(unittest.TestCase):
    """The score stays in [0,1] no matter how adversarial the input."""

    def setUp(self):
        self.gauge = TailRiskGauge()

    def test_score_in_unit_interval_on_varied_inputs(self):
        rng = np.random.default_rng(5)
        cases = [
            _calm_series(),
            _fragile_series(),
            _closes_from_returns(rng.normal(0, 0.2, 300)),      # extreme vol
            _closes_from_returns(np.full(300, -0.05)),          # relentless crash
            np.full(300, 100.0),                                # flat
            np.array([100.0, 100.0]),                           # minimal
        ]
        for closes in cases:
            score = self.gauge.black_swan_score(closes)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_crisis_posterior_is_clamped_into_the_score(self):
        calm = _calm_series()
        low = self.gauge.black_swan_score(calm, crisis_posterior=0.0)
        high = self.gauge.black_swan_score(calm, crisis_posterior=1.0)
        self.assertGreater(high, low)
        # Out-of-range posteriors are clamped, not propagated.
        for bad in (-5.0, 5.0, float("nan")):
            score = self.gauge.black_swan_score(calm, crisis_posterior=bad)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


class SubscoreAndBlendTests(unittest.TestCase):
    """The audit trail must reflect exactly what was blended."""

    def test_weights_renormalise_over_present_components(self):
        gauge = TailRiskGauge()
        reading = gauge.reading(_fragile_series())
        self.assertTrue(reading.subscores)
        self.assertAlmostEqual(sum(reading.weights.values()), 1.0, places=9)
        self.assertEqual(set(reading.weights), set(reading.subscores))

    def test_regime_component_absent_without_a_posterior(self):
        gauge = TailRiskGauge()
        reading = gauge.reading(_calm_series())
        self.assertNotIn("regime", reading.subscores)
        reading2 = gauge.reading(_calm_series(), crisis_posterior=0.8)
        self.assertIn("regime", reading2.subscores)
        self.assertAlmostEqual(reading2.subscores["regime"], 0.8, places=9)


class MandateHelperTests(unittest.TestCase):
    """Fragility may only tighten the HMM's regime_risk, and stays Decimal."""

    def test_returns_decimal(self):
        out = combine_regime_risk(Decimal("0.2"), 0.5)
        self.assertIsInstance(out, Decimal)

    def test_never_lowers_the_hmm_risk(self):
        for hmm in ("0.0", "0.3", "0.7", "1.0"):
            base = Decimal(hmm)
            for frag in (0.0, 0.3, 0.9, 1.0):
                out = combine_regime_risk(base, frag)
                self.assertGreaterEqual(out, base)
                self.assertLessEqual(out, Decimal("1"))

    def test_zero_fragility_is_identity(self):
        self.assertEqual(combine_regime_risk(Decimal("0.4"), 0.0), Decimal("0.4"))

    def test_weight_caps_the_tightening(self):
        loose = combine_regime_risk(Decimal("0.0"), 1.0, weight=0.2)
        tight = combine_regime_risk(Decimal("0.0"), 1.0, weight=1.0)
        self.assertLess(loose, tight)
        self.assertAlmostEqual(float(loose), 0.2, places=9)


class FalseAlarmEvaluationTests(unittest.TestCase):
    """The evaluation must compute an honest, non-lookahead comparison."""

    def test_forward_drawdown_is_nonpositive_and_tail_is_nan(self):
        rng = np.random.default_rng(2)
        closes = _closes_from_returns(rng.normal(0, 0.01, 200))
        fdd = forward_drawdown(closes, horizon=10)
        finite = fdd[np.isfinite(fdd)]
        self.assertTrue(np.all(finite <= 1e-9))
        self.assertTrue(np.all(np.isnan(fdd[-10:])))

    def test_report_fields_are_consistent(self):
        gauge = TailRiskGauge(window=120)
        closes = _fragile_series(seed=6, n=500)
        report = evaluate_false_alarms(
            closes, gauge, threshold=0.6, horizon=21, tail_move=0.1,
        )
        self.assertEqual(report.n_high + report.n_low, report.n)
        self.assertAlmostEqual(
            report.hit_rate_high + report.false_alarm_rate_high, 1.0, places=9,
        )
        self.assertGreaterEqual(report.alarm_rate, 0.0)
        self.assertLessEqual(report.alarm_rate, 1.0)
        self.assertIsInstance(report.summary(), str)


if __name__ == "__main__":
    unittest.main()
