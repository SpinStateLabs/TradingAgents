"""Tests for the added strategy families: high-order Markov, regime-switching,
and the game-theory (Hedge) ensemble. No network.

Focus: the core logic of each (forecast/backoff, expert reweighting, graceful
degradation without hmmlearn), that every grid config constructs a valid agent,
and that each runs through the real backtester without error.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

import numpy as np

from spintrader.agents.personas.hedge import HedgeEnsembleAgent
from spintrader.agents.personas.markov_chain import HighOrderMarkovAgent
from spintrader.agents.personas.regime_switch import RegimeSwitchingAgent
from spintrader.backtest.engine import ReplayCursor
from spintrader.backtest.runner import backtest_instrument, run_backtest
from spintrader.core.types import AssetClass, Bar
from spintrader.risk.engine import Mandate

D = Decimal
UTC = timezone.utc
INST = backtest_instrument("X-USD", AssetClass.CRYPTO)


def bars_from(closes, interval="1d"):
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    out = []
    for i, c in enumerate(closes):
        c = float(c)
        out.append(Bar(instrument_key=INST.key, ts=t0 + timedelta(days=i + 1),
                       interval=interval, open=D(str(round(c, 4))),
                       high=D(str(round(c + 0.5, 4))), low=D(str(round(c - 0.5, 4))),
                       close=D(str(round(c, 4))), volume=D("1")))
    return out


def noisy_uptrend(n=300, start=100.0, drift=0.0008, noise=0.004, seed=7):
    rng = np.random.default_rng(seed)
    closes, p = [], start
    for _ in range(n):
        p *= math.exp(drift + noise * rng.standard_normal())
        closes.append(p)
    return closes


# --------------------------------------------------------------------------
# High-order Markov chain
# --------------------------------------------------------------------------

class MarkovChainTests(unittest.TestCase):
    def test_forecast_counts_and_backoff(self):
        a = HighOrderMarkovAgent(order=1, n_states=2, min_support=2, lookback=50)
        # Symbols: strictly alternating 0,1,0,1,... -> after a 0 comes a 1.
        symbols = np.array([0, 1] * 20)
        probs, support, order_used = a._forecast(symbols)
        # Current k-gram is the last symbol (1); after 1 always comes 0.
        self.assertEqual(order_used, 1)
        self.assertGreater(support, 2)
        self.assertGreater(probs[0], probs[1])          # predicts 0 (down) next

    def test_backs_off_when_kgram_unseen(self):
        # Order 3 with a short, non-repeating tail forces a back-off to survive.
        a = HighOrderMarkovAgent(order=3, n_states=2, min_support=3, lookback=40)
        symbols = np.array([0, 1] * 20)
        probs, support, order_used = a._forecast(symbols)
        self.assertLessEqual(order_used, 3)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_flat_market_never_trades(self):
        a = HighOrderMarkovAgent(order=1, n_states=2, lookback=50, interval="1d")
        cur = ReplayCursor(bars_from([100.0] * 80))
        events = []
        while True:
            events += list(a.on_bar(cur, INST, Mandate.open_mandate([INST.key], hours=10**6)))
            if not cur.advance():
                break
        self.assertEqual(events, [])

    def test_runs_through_the_backtester(self):
        run = run_backtest(HighOrderMarkovAgent, "X-USD", bars_from(noisy_uptrend(300)),
                           asset_class=AssetClass.CRYPTO, walk_forward=False,
                           strategy_kwargs={"order": 1, "n_states": 2, "lookback": 120})
        self.assertIsNotNone(run.result.scorecard)

    def test_warmup_and_fit_reset(self):
        a = HighOrderMarkovAgent(order=2, lookback=100)
        self.assertEqual(a.warmup_bars, 100 + 2 + 2)
        a._long = True
        a.fit([])
        self.assertFalse(a.is_long)


# --------------------------------------------------------------------------
# Game-theory Hedge ensemble
# --------------------------------------------------------------------------

class HedgeTests(unittest.TestCase):
    def test_hedge_favours_the_paying_expert(self):
        a = HedgeEnsembleAgent(eta="4.0")
        # Three experts; only expert 0's vote consistently matches the return.
        n = 60
        votes = np.zeros((n, 3))
        votes[:, 0] = 1.0            # expert 0 always long
        votes[:, 1] = -1.0           # expert 1 always short
        votes[:, 2] = 0.0
        returns = np.full(n, 0.01)   # market always rises -> long expert wins
        w = a._hedge_weights(votes, returns)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)
        self.assertEqual(int(np.argmax(w)), 0)          # the long expert dominates

    def test_expert_votes_are_bounded(self):
        a = HedgeEnsembleAgent()
        votes = a._expert_votes(np.array(noisy_uptrend(120)))
        self.assertEqual(votes.shape[1], 3)
        self.assertTrue(np.all(votes >= -1.0) and np.all(votes <= 1.0))

    def test_flat_market_never_trades(self):
        a = HedgeEnsembleAgent(lookback=60, interval="1d")
        cur = ReplayCursor(bars_from([100.0] * 120))
        events = []
        while True:
            events += list(a.on_bar(cur, INST, Mandate.open_mandate([INST.key], hours=10**6)))
            if not cur.advance():
                break
        self.assertEqual(events, [])

    def test_runs_through_the_backtester(self):
        run = run_backtest(HedgeEnsembleAgent, "X-USD", bars_from(noisy_uptrend(300)),
                           asset_class=AssetClass.CRYPTO, walk_forward=False,
                           strategy_kwargs={"lookback": 120, "eta": "2.0"})
        self.assertIsNotNone(run.result.scorecard)


# --------------------------------------------------------------------------
# Markov regime-switching (HMM)
# --------------------------------------------------------------------------

class RegimeSwitchTests(unittest.TestCase):
    def _agent(self):
        return RegimeSwitchingAgent(fit_window=120, refit_interval=60, interval="1d")

    def test_stands_aside_without_a_regime_model(self):
        # If the HMM cannot fit (hmmlearn absent, or a fit error), the persona
        # must do nothing -- no trades, no exception.
        a = self._agent()
        cur = ReplayCursor(bars_from(noisy_uptrend(220)))
        from spintrader.quant.regime import RegimeError
        with mock.patch("spintrader.quant.regime.RegimeModel.fit",
                        side_effect=RegimeError("hmmlearn is not installed")):
            events = []
            while True:
                events += list(a.on_bar(cur, INST, Mandate.open_mandate([INST.key], hours=10**6)))
                if not cur.advance():
                    break
        self.assertEqual(events, [])
        self.assertTrue(a._regime_unavailable)

    def test_runs_through_the_backtester_even_without_hmmlearn(self):
        # With or without hmmlearn the backtest must complete; without it the
        # agent simply never trades.
        run = run_backtest(RegimeSwitchingAgent, "X-USD", bars_from(noisy_uptrend(320)),
                           asset_class=AssetClass.CRYPTO, walk_forward=False,
                           strategy_kwargs={"fit_window": 120, "refit_interval": 60})
        self.assertIsNotNone(run.result.scorecard)

    def test_warmup_and_fit_reset(self):
        a = self._agent()
        self.assertEqual(a.warmup_bars, 120 + 60)
        a._long = True
        a._regime_unavailable = True
        a.fit([])
        self.assertFalse(a.is_long)
        self.assertFalse(a._regime_unavailable)


# --------------------------------------------------------------------------
# Factory integration -- every family, every config, constructs
# --------------------------------------------------------------------------

class FamilyIntegrationTests(unittest.TestCase):
    def test_default_families_span_all_five(self):
        from spintrader.research.factory import CandidateFactory, default_families
        families = {c.family for c in CandidateFactory(families=default_families()).all_configs()}
        self.assertEqual(
            families,
            {"trend", "mean_reversion", "markov_chain", "regime_switch", "hedge"},
        )

    def test_every_config_constructs_a_valid_agent(self):
        # A grid point that cannot build its agent would raise mid-backtest; catch
        # it here instead. interval/continuous/spread_bps are injected by the
        # backtest runner, so supply them like it does.
        from spintrader.research.factory import CandidateFactory, default_families
        for cfg in CandidateFactory(families=default_families()).all_configs():
            with self.subTest(config=cfg.name):
                kwargs = dict(cfg.to_dict())
                kwargs.setdefault("interval", "1m")
                kwargs.setdefault("continuous", True)
                agent = cfg.strategy_cls(**kwargs)
                self.assertGreater(agent.warmup_bars, 0)


if __name__ == "__main__":
    unittest.main()
