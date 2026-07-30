"""Sentiment threaded through the backtest/improve pipeline (task 17 follow-on).

The sentiment persona reads a channel bars do not carry, so it takes its series
at construction. The seam under test is the one that lets a *walk-forward* run
build that series **per fold** instead of once globally: a
:func:`~spintrader.research.factory.sentiment_family` supplies a
``context_provider`` that ``run_backtest`` calls with each fold's own window, and
:class:`~spintrader.loop.improvement.ImprovementCycle` routes it through.

Two properties carry the weight here.

**It runs.** A :class:`~spintrader.agents.personas.sentiment.SentimentAgent`,
handed an injected fake sentiment series, must trade through ``run_backtest`` and
produce a walk-forward result -- proving sentiment is now searchable like any
other family, not a bolt-on that only works in a unit test.

**It stays causal across folds.** This is the whole reason the seam is per-fold
rather than a single global map. A fold's sentiment is sliced to that fold's
window, so fold N's injected series can never contain a score aligned to fold
N+1's bars. If this regresses, the walk-forward's out-of-sample guarantee is
silently broken for the sentiment channel while the price channel still looks
honest -- the most dangerous kind of leak.

No network, no LLM, no DB: the feed is a fake keyed on the bar timestamps.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.agents.personas.sentiment import SentimentAgent
from spintrader.backtest.runner import run_backtest
from spintrader.core.types import AssetClass, Bar
from spintrader.data.sentiment import SentimentScore
from spintrader.loop.improvement import ImprovementCycle
from spintrader.loop.promotion import PromotionGate, TrialLedger
from spintrader.research.factory import CandidateFactory, sentiment_family
from spintrader.research.memory import ResearchMemory

D = Decimal
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
KEY = "paper:BTC-USD"


def make_bars(n: int = 140) -> list[Bar]:
    """A gently rising crypto daily series with alternating noise.

    The drift keeps the agent's stop from firing; the noise keeps trailing
    volatility above the near-zero floor the persona refuses to divide by, and
    well under its ceiling. Both matter -- a flat series would either never open
    (zero vol) or never be interesting.
    """
    out: list[Bar] = []
    price = 100.0
    for i in range(n):
        price *= 1.0 + 0.003 + (0.0015 if i % 2 else -0.0015)
        c = round(price, 6)
        out.append(Bar(
            instrument_key=KEY, ts=T0 + timedelta(days=i), interval="1d",
            open=D(str(round(c * 0.999, 6))), high=D(str(round(c * 1.002, 6))),
            low=D(str(round(c * 0.998, 6))), close=D(str(c)), volume=D("100"),
        ))
    return out


def make_sentiment(bars: list[Bar]) -> dict[datetime, SentimentScore]:
    """A steady, confident bullish score stamped at every bar's close-time."""
    return {
        b.ts: SentimentScore(
            ts=b.ts, symbol="BTC-USD", score=D("0.6"), mentions=5,
            volume=D("30"), source="fake", interval="1d",
        )
        for b in bars
    }


class FakeFeed:
    """A :class:`SentimentFeed` stand-in: no network, and it records its calls.

    ``mapping`` honours ``since`` the same way the real feed does (scores at or
    after it), which is what lets the provider's per-fold slicing be exercised
    for real rather than mocked away.
    """

    def __init__(self, series: dict[datetime, SentimentScore]) -> None:
        self.series = dict(series)
        self.calls: list[tuple[str, datetime | None, int]] = []

    def mapping(self, symbol, since=None, interval_minutes=1, flush=True):
        self.calls.append((symbol, since, interval_minutes))
        if since is None:
            return dict(self.series)
        return {ts: sc for ts, sc in self.series.items() if ts >= since}


class SentimentWalkForwardTests(unittest.TestCase):
    def test_agent_runs_walk_forward_with_injected_sentiment(self):
        bars = make_bars(140)
        feed = FakeFeed(make_sentiment(bars))
        provider = sentiment_family(feed, interval_minutes=1440).context_provider("BTC-USD")

        run = run_backtest(
            SentimentAgent, "BTC-USD", bars,
            asset_class=AssetClass.CRYPTO, starting_cash="100000",
            walk_forward=True, train_size=20, test_size=20, embargo=5,
            strategy_kwargs={
                "vol_lookback": 5, "entry_threshold": "0.30",
                "exit_threshold": "0.05", "min_mentions": 0,
            },
            context_provider=provider,
        )

        # The walk-forward ran with the persona, and the persona actually acted
        # on the injected sentiment (an inert run would prove nothing about the
        # seam).
        self.assertIsNotNone(run.walk_forward)
        self.assertGreaterEqual(run.walk_forward.n_folds, 2)
        self.assertGreaterEqual(run.result.n_orders, 1)
        self.assertGreaterEqual(sum(f.n_orders for f in run.walk_forward.folds), 1)

    def test_fold_sentiment_never_uses_a_later_folds_data(self):
        bars = make_bars(140)
        feed = FakeFeed(make_sentiment(bars))
        base = sentiment_family(feed, interval_minutes=1440).context_provider("BTC-USD")

        # Record what sentiment each fold's construction was actually handed.
        seen: list[tuple[datetime, datetime, tuple[datetime, ...]]] = []

        def recording(window):
            ctx = base(window)
            seen.append((window[0].ts, window[-1].ts,
                         tuple(sorted(ctx["sentiment"]))))
            return ctx

        run_backtest(
            SentimentAgent, "BTC-USD", bars,
            asset_class=AssetClass.CRYPTO, starting_cash="100000",
            walk_forward=True, train_size=20, test_size=20, embargo=5,
            strategy_kwargs={
                "vol_lookback": 5, "entry_threshold": "0.30",
                "exit_threshold": "0.05", "min_mentions": 0,
            },
            context_provider=recording,
        )

        # Drop the single full-sample call (window starts at the very first bar);
        # what remains is one call per fold.
        folds = sorted((r for r in seen if r[0] != bars[0].ts), key=lambda r: r[0])
        self.assertGreaterEqual(len(folds), 2)

        all_ts = [b.ts for b in bars]
        for start, end, keys in folds:
            # Non-empty, or the disjointness below would be vacuous.
            self.assertTrue(keys)
            # A fold's sentiment lives strictly within its own window.
            self.assertTrue(all(start <= k <= end for k in keys))

        for (a_start, a_end, a_keys), (b_start, b_end, b_keys) in zip(folds, folds[1:]):
            # The causal guarantee, stated two ways: nothing in fold N reaches
            # to or past fold N+1's window start, and the two share no timestamp.
            self.assertLess(max(a_keys), b_start)
            later_window = {t for t in all_ts if b_start <= t <= b_end}
            self.assertTrue(set(a_keys).isdisjoint(later_window))


# --------------------------------------------------------------------------
# ImprovementCycle routing
# --------------------------------------------------------------------------

@dataclass
class _Card:
    sharpe: float
    n_observations: int = 600
    skew: float = 0.0
    excess_kurtosis: float = 0.0
    periods_per_year: int = 365
    cost_drag: float = 0.1
    max_drawdown: float = -0.05
    sharpe_stderr: float = 0.1
    total_return: float = 0.2


@dataclass
class _Fold:
    scorecard: _Card


@dataclass
class _WF:
    combined: _Card | None
    n_folds: int = 4
    folds: list = field(default_factory=list)


@dataclass
class _Result:
    scorecard: _Card | None


@dataclass
class _Run:
    walk_forward: _WF | None
    result: _Result


class ImprovementRoutingTests(unittest.TestCase):
    def test_cycle_routes_the_feed_provider_per_fold(self):
        bars = make_bars(60)
        feed = FakeFeed(make_sentiment(bars))
        factory = CandidateFactory(families=[sentiment_family(feed)])

        captured: dict[str, object] = {}

        def fake_backtest_fn(strategy_cls, symbol, bars, *, asset_class,
                             aggression, starting_cash, walk_forward, n_trials,
                             strategy_kwargs, context_provider=None):
            captured["strategy_cls"] = strategy_cls
            captured["provider"] = context_provider
            card = _Card(sharpe=3.0)
            return _Run(walk_forward=_WF(combined=card, folds=[_Fold(card)] * 4),
                        result=_Result(card))

        cycle = ImprovementCycle(
            gate=PromotionGate(ledger=TrialLedger()), memory=ResearchMemory(),
            factory=factory, backtest_fn=fake_backtest_fn,
        )
        result = cycle.run_round("crypto", "BTC-USD", bars, n_candidates=1)

        # The cycle backtested the sentiment persona and threaded the family's
        # feed-backed provider through -- this is what makes it "searchable like
        # any other family".
        self.assertEqual(result.evaluated, 1)
        self.assertIs(captured["strategy_cls"], SentimentAgent)
        provider = captured["provider"]
        self.assertIsNotNone(provider)

        # And the routed provider is the real per-fold one: give it a window and
        # it slices the feed to exactly that window.
        ctx = provider(bars[10:20])
        self.assertIn("sentiment", ctx)
        self.assertTrue(ctx["sentiment"])
        self.assertTrue(all(bars[10].ts <= k <= bars[19].ts for k in ctx["sentiment"]))


if __name__ == "__main__":
    unittest.main()
