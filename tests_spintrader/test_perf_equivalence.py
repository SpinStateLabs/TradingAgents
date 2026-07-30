"""Output-equality tests for the vectorised hot paths (lessons L6 + L12).

The speedups only count if they change nothing. Each vectorised routine is
checked against a naive reference on random data: identical results, no
Python-loop O(window)-per-bar cost.
"""

from __future__ import annotations

import unittest

import numpy as np

from spintrader.agents.personas.hedge import HedgeEnsembleAgent
from spintrader.agents.personas.markov_chain import HighOrderMarkovAgent
from spintrader.quant.features import rolling_mean, rolling_std


def naive_rolling_std(values, window):
    n = values.size
    out = np.zeros(n)
    for i in range(n):
        chunk = values[max(0, i - window + 1):i + 1]
        out[i] = chunk.std(ddof=1) if chunk.size > 1 else 0.0
    return out


def naive_rolling_mean(values, window):
    n = values.size
    out = np.zeros(n)
    for i in range(n):
        out[i] = values[max(0, i - window + 1):i + 1].mean()
    return out


def naive_hedge_weights(votes, returns, eta):
    n_experts = votes.shape[1]
    log_w = np.zeros(n_experts)
    for i in range(1, votes.shape[0]):
        log_w += eta * (votes[i - 1] * returns[i])
        log_w -= log_w.max()
    w = np.exp(log_w)
    t = w.sum()
    return w / t if t > 0 else np.full(n_experts, 1.0 / n_experts)


def naive_breakout(closes, w):
    n = closes.size
    out = np.zeros(n)
    for i in range(n):
        prior = closes[max(0, i - w):i]
        if prior.size:
            if closes[i] > prior.max():
                out[i] = 1.0
            elif closes[i] < prior.min():
                out[i] = -1.0
    return out


def naive_markov_forecast(symbols, order, n_states, min_support):
    current = tuple(symbols[-order:])
    for k in range(order, -1, -1):
        counts = np.zeros(n_states)
        gram = current[order - k:] if k > 0 else ()
        for i in range(k, symbols.size):
            if k == 0 or tuple(symbols[i - k:i]) == gram:
                counts[symbols[i]] += 1
        support = int(counts.sum())
        if support >= min_support:
            return counts / support, support, k
    return np.full(n_states, 1.0 / n_states), 0, 0


class RollingEquivalenceTests(unittest.TestCase):
    def test_rolling_std_and_mean_match_naive(self):
        rng = np.random.default_rng(1)
        for n, w in [(5, 3), (20, 5), (100, 20), (100, 100), (100, 200), (1, 5)]:
            with self.subTest(n=n, w=w):
                v = rng.standard_normal(n) * 100 + 50_000     # price-scale values
                self.assertTrue(np.allclose(rolling_std(v, w), naive_rolling_std(v, w),
                                            rtol=0, atol=1e-9))
                self.assertTrue(np.allclose(rolling_mean(v, w), naive_rolling_mean(v, w),
                                            rtol=0, atol=1e-9))


class HedgeEquivalenceTests(unittest.TestCase):
    def test_weights_match_naive(self):
        rng = np.random.default_rng(2)
        a = HedgeEnsembleAgent(eta="3.0")
        for n in (2, 10, 200):
            with self.subTest(n=n):
                votes = rng.choice([-1.0, 0.0, 1.0], size=(n, 3))
                returns = rng.standard_normal(n) * 0.01
                self.assertTrue(np.allclose(
                    a._hedge_weights(votes, returns),
                    naive_hedge_weights(votes, returns, 3.0), atol=1e-9))

    def test_breakout_expert_matches_naive(self):
        rng = np.random.default_rng(3)
        a = HedgeEnsembleAgent(breakout_window=15)
        closes = np.cumsum(rng.standard_normal(300)) + 100
        got = a._expert_votes(closes)[:, 2]                  # the breakout column
        self.assertTrue(np.array_equal(got, naive_breakout(closes, 15)))


class MarkovEquivalenceTests(unittest.TestCase):
    def test_forecast_matches_naive(self):
        rng = np.random.default_rng(4)
        for order, n_states in [(1, 2), (2, 2), (2, 3), (3, 3)]:
            with self.subTest(order=order, n_states=n_states):
                a = HighOrderMarkovAgent(order=order, n_states=n_states,
                                         min_support=5, lookback=200)
                symbols = rng.integers(0, n_states, size=250)
                p1, s1, k1 = a._forecast(symbols)
                p2, s2, k2 = naive_markov_forecast(symbols, order, n_states, 5)
                self.assertEqual((s1, k1), (s2, k2))
                self.assertTrue(np.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()
