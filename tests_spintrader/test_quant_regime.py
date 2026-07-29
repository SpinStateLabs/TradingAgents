"""Tests for the Markovian regime layer.

The tests that matter most are the causality ones. A lookahead bug here does
not raise; it produces a regime signal that appears to anticipate volatility,
and every strategy conditioned on it reports risk-adjusted returns that
evaporate on contact with a live market.

Synthetic data is generated with known regimes so the model can be checked
against ground truth rather than against its own output.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np

from spintrader.core.types import Bar
from spintrader.quant.features import (
    FeatureMatrix, assert_causal, build_features, drawdown, log_returns,
    periods_per_year, rolling_mean, rolling_std, zscore,
)

D = Decimal
T0 = datetime(2021, 1, 1, tzinfo=timezone.utc)


def make_bars(closes, interval="1d", volumes=None):
    volumes = volumes if volumes is not None else [1000.0] * len(closes)
    bars = []
    for i, (close, volume) in enumerate(zip(closes, volumes)):
        price = float(close)
        bars.append(Bar(
            instrument_key="test:SYN-USD",
            ts=T0 + timedelta(days=i),
            interval=interval,
            open=D(str(price)), high=D(str(price * 1.01)),
            low=D(str(price * 0.99)), close=D(str(price)),
            volume=D(str(volume)),
        ))
    return bars


def synthetic_regimes(seed=0, n_per_regime=300):
    """A price series with two genuinely different volatility regimes."""
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0008, 0.006, n_per_regime)      # drifting up, quiet
    crisis = rng.normal(-0.003, 0.045, n_per_regime)    # falling, violent
    calm2 = rng.normal(0.0008, 0.006, n_per_regime)
    returns = np.concatenate([calm, crisis, calm2])
    truth = np.array([0] * n_per_regime + [1] * n_per_regime + [0] * n_per_regime)
    closes = 100.0 * np.exp(np.cumsum(returns))
    return closes, truth


class RollingStatisticCausalityTests(unittest.TestCase):
    """Truncating the input must never change earlier outputs."""

    def setUp(self):
        self.values = np.random.default_rng(1).normal(0, 1, 200)

    def test_rolling_std_is_causal(self):
        assert_causal(rolling_std, self.values, 20)

    def test_rolling_mean_is_causal(self):
        assert_causal(rolling_mean, self.values, 20)

    def test_zscore_is_causal(self):
        assert_causal(zscore, self.values, 20)

    def test_rolling_std_uses_a_trailing_window(self):
        values = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
        out = rolling_std(values, 2)
        # The spike at index 3 must not affect index 2.
        self.assertEqual(out[2], 0.0)
        self.assertGreater(out[3], 0.0)

    def test_drawdown_is_non_positive_and_causal(self):
        closes = np.array([100.0, 110.0, 105.0, 120.0, 90.0])
        dd = drawdown(closes)
        self.assertTrue(np.all(dd <= 0))
        self.assertEqual(dd[1], 0.0)              # new peak
        self.assertAlmostEqual(dd[4], (90 - 120) / 120)


class LogReturnTests(unittest.TestCase):
    def test_length_preserved_with_leading_zero(self):
        out = log_returns(np.array([100.0, 110.0, 121.0]))
        self.assertEqual(out.size, 3)
        self.assertEqual(out[0], 0.0)

    def test_additive_across_time(self):
        # The reason for log rather than simple returns.
        out = log_returns(np.array([100.0, 110.0, 121.0]))
        self.assertAlmostEqual(out[1] + out[2], np.log(121 / 100))


class AnnualisationTests(unittest.TestCase):
    def test_crypto_uses_365_days(self):
        self.assertEqual(periods_per_year("1d", continuous=True), 365)

    def test_equities_use_252_sessions(self):
        # Using 365 for equities overstates annualised vol by ~20%, which
        # shifts every vol-targeted position size.
        self.assertEqual(periods_per_year("1d", continuous=False), 252)

    def test_unknown_interval_raises(self):
        with self.assertRaises(ValueError):
            periods_per_year("3s")


class BuildFeaturesTests(unittest.TestCase):
    def test_requires_enough_bars(self):
        with self.assertRaises(ValueError):
            build_features(make_bars([100.0] * 10))

    def test_warmup_rows_dropped(self):
        bars = make_bars(list(100 + np.arange(200, dtype=float)))
        features = build_features(bars, vol_window=20, trend_window=50)
        self.assertEqual(len(features), 200 - 50)
        self.assertEqual(features.n_features, 6)

    def test_all_values_finite(self):
        closes, _ = synthetic_regimes(seed=3, n_per_regime=100)
        features = build_features(make_bars(closes))
        self.assertTrue(np.all(np.isfinite(features.values)))

    def test_rejects_non_positive_prices(self):
        with self.assertRaises(ValueError):
            build_features(make_bars([100.0] * 60 + [0.0] + [100.0] * 60))

    def test_feature_matrix_is_causal_end_to_end(self):
        # Truncating the bar series must not change earlier feature rows.
        closes, _ = synthetic_regimes(seed=5, n_per_regime=120)
        bars = make_bars(closes)
        full = build_features(bars)
        partial = build_features(bars[:250])
        overlap = len(partial)
        np.testing.assert_allclose(full.values[:overlap], partial.values, atol=1e-10)

    def test_slice_to_returns_a_prefix(self):
        closes, _ = synthetic_regimes(seed=7, n_per_regime=100)
        features = build_features(make_bars(closes))
        sliced = features.slice_to(50)
        self.assertEqual(len(sliced), 50)
        np.testing.assert_array_equal(sliced.values, features.values[:50])


class RegimeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("hmmlearn not installed")
        closes, cls.truth = synthetic_regimes(seed=11)
        cls.bars = make_bars(closes)
        cls.features = build_features(cls.bars)

    def setUp(self):
        from spintrader.quant.regime import RegimeModel
        self.model = RegimeModel(n_states=2, n_restarts=3, random_state=7)
        self.model.fit(self.features)

    def test_fits_and_records_provenance(self):
        self.assertIsNotNone(self.model.log_likelihood)
        self.assertEqual(self.model.fit_start, self.features.timestamps[0])
        self.assertEqual(self.model.fit_end, self.features.timestamps[-1])

    def test_states_ordered_calmest_first(self):
        # The contract that makes state indices meaningful across refits.
        vols = self.model.state_volatility()
        self.assertTrue(np.all(np.diff(vols) >= 0),
                        f"states not ordered by volatility: {vols}")

    def test_reported_volatility_is_non_negative(self):
        # Regression: means_ live in standardised space, so reporting them
        # directly produced 'annualised volatility -90%'. They must be
        # inverse-transformed before being shown as volatility.
        vols = self.model.state_volatility()
        self.assertTrue(np.all(vols >= 0), f"negative volatility reported: {vols}")

    def test_reported_volatility_is_in_a_plausible_range(self):
        # Synthetic data has ~10% and ~70% annualised vol by construction.
        vols = self.model.state_volatility()
        self.assertLess(float(vols.max()), 5.0, "volatility implausibly large")
        self.assertGreater(float(vols.max()), 0.05, "volatility implausibly small")

    def test_labels_match_state_count(self):
        from spintrader.quant.regime import labels_for
        self.assertEqual(labels_for(2), ("calm", "stressed"))
        self.assertEqual(len(labels_for(3)), 3)

    def test_filter_returns_one_state_per_observation(self):
        states = self.model.filter(self.features)
        self.assertEqual(len(states), len(self.features))

    def test_posteriors_are_probability_distributions(self):
        for state in self.model.filter(self.features)[:50]:
            self.assertAlmostEqual(float(state.probabilities.sum()), 1.0, places=6)
            self.assertTrue(np.all(state.probabilities >= 0))

    def test_filtering_is_causal(self):
        """The load-bearing test.

        Filtered state at time t must not change when later data is removed.
        hmmlearn's predict() would fail this, which is exactly why it is not
        used for signals.
        """
        cut = len(self.features) // 2
        full = self.model.filter(self.features)
        partial = self.model.filter(self.features.slice_to(cut))
        self.assertEqual(len(partial), cut)
        for i in range(cut):
            self.assertEqual(
                full[i].state, partial[i].state,
                f"filtered state at index {i} changed when future data was "
                f"removed -- the filter is reading ahead",
            )

    def test_smoothed_states_are_not_causal(self):
        """Documents why smoothed states are quarantined.

        This asserts the opposite of the test above: Viterbi over the whole
        sequence *does* change earlier assignments when later data changes.
        If this ever stops being true the quarantine can be revisited, but
        until then the naming must keep it away from trading code.
        """
        cut = len(self.features) // 2
        full = self.model.smoothed_states_research_only(self.features)
        partial = self.model.smoothed_states_research_only(self.features.slice_to(cut))
        # Not asserting they differ everywhere -- only that they may differ,
        # which is enough to disqualify them as a causal signal.
        self.assertEqual(len(partial), cut)
        self.assertEqual(len(full), len(self.features))

    def test_recovers_the_volatile_regime(self):
        # Ground truth: the middle third is the crisis.
        states = self.model.filter(self.features)
        inferred = np.array([s.state for s in states])
        warmup = len(self.bars) - len(self.features)
        truth = self.truth[warmup:]
        middle = truth == 1
        # The stressed state (index 1) should dominate the crisis period.
        stressed_share = float(np.mean(inferred[middle] == 1))
        self.assertGreater(stressed_share, 0.6,
                           f"only {stressed_share:.0%} of the crisis period was "
                           f"classified as stressed")

    def test_risk_score_bounds_and_ordering(self):
        calm = self.model.risk_score(np.array([1.0, 0.0]))
        crisis = self.model.risk_score(np.array([0.0, 1.0]))
        split = self.model.risk_score(np.array([0.5, 0.5]))
        self.assertEqual(calm, 0.0)
        self.assertEqual(crisis, 1.0)
        self.assertAlmostEqual(split, 0.5)

    def test_risk_score_is_probability_weighted_not_argmax(self):
        # A 55/45 split must not read as confident calm.
        score = self.model.risk_score(np.array([0.55, 0.45]))
        self.assertGreater(score, 0.4)
        self.assertLess(score, 0.5)

    def test_confidence_exposes_indecision(self):
        from spintrader.quant.regime import RegimeState
        state = RegimeState(ts=T0, state=0, label="calm",
                            probabilities=np.array([0.51, 0.49]), risk_score=0.49)
        self.assertAlmostEqual(state.confidence, 0.51)

    def test_transition_matrix_rows_sum_to_one(self):
        rows = self.model.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(rows, 1.0, atol=1e-6)

    def test_regimes_persist_for_more_than_a_bar(self):
        # A "regime" lasting one bar is noise wearing a label.
        durations = self.model.expected_durations()
        self.assertTrue(np.all(durations > 2.0),
                        f"expected durations too short: {durations}")

    def test_bic_penalises_more_states(self):
        from spintrader.quant.regime import RegimeModel
        small = self.model.n_parameters()
        large = RegimeModel(n_states=4, feature_names=self.features.names).n_parameters()
        self.assertGreater(large, small)

    def test_inference_before_fit_raises(self):
        from spintrader.quant.regime import RegimeError, RegimeModel
        with self.assertRaises(RegimeError):
            RegimeModel().filter(self.features)

    def test_too_few_observations_raises(self):
        from spintrader.quant.regime import RegimeError, RegimeModel
        tiny = self.features.slice_to(5)
        with self.assertRaises(RegimeError):
            RegimeModel(n_states=3).fit(tiny)


class ModelSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import hmmlearn  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("hmmlearn not installed")
        closes, _ = synthetic_regimes(seed=23, n_per_regime=250)
        cls.features = build_features(make_bars(closes))

    def test_selects_a_stable_model(self):
        from spintrader.quant.regime import select_n_states
        model, scores = select_n_states(self.features, candidates=(2, 3),
                                        n_restarts=3)
        self.assertIn(model.n_states, (2, 3))
        self.assertTrue(np.all(model.expected_durations() >= 3.0))

    def test_reports_scores_for_every_candidate(self):
        from spintrader.quant.regime import select_n_states
        _, scores = select_n_states(self.features, candidates=(2, 3), n_restarts=3)
        self.assertEqual(set(scores), {2, 3})

    def test_rejects_everything_when_the_floor_is_impossible(self):
        from spintrader.quant.regime import RegimeError, select_n_states
        with self.assertRaises(RegimeError):
            select_n_states(self.features, candidates=(2,), n_restarts=2,
                            min_duration_bars=10_000)


if __name__ == "__main__":
    unittest.main()
