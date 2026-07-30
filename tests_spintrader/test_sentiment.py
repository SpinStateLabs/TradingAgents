"""Tests for the sentiment feed and the sentiment persona. No network.

Two things carry the weight here, and both mirror guarantees the price feeds
already make. First, the feed aggregates raw mentions into a per-interval score
stamped at the bucket CLOSE, holding the forming bucket until a later mention
proves it done -- the same lookahead-safe rule the trades feed applies, so a
strategy can look sentiment up by the current bar's timestamp. Second, the
persona is causal and long-only: it buys positive, rising sentiment, exits on
decay or a stop, evaluates exits before any entry guard, and resets on fit.

Every test injects a FAKE fetcher or builds mentions by hand, so nothing here
touches a socket.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.agents.personas.sentiment import SentimentAgent
from spintrader.backtest.engine import ReplayCursor
from spintrader.backtest.runner import backtest_instrument
from spintrader.core.types import AssetClass, Bar, Side
from spintrader.data.sentiment import (
    SentimentBucketAggregator, SentimentFeed, SentimentMention, SentimentScore,
    fetch_sentiment, parse_reddit_posts, parse_stocktwits_messages, score_text,
)
from spintrader.risk.engine import Mandate

D = Decimal
UTC = timezone.utc
INST = backtest_instrument("X-USD", AssetClass.CRYPTO)


def m(minute, second=0, score="1", weight="1", *, hour=12, day=29, source="stocktwits"):
    return SentimentMention(
        ts=datetime(2026, 7, day, hour, minute, second, tzinfo=UTC),
        symbol="X-USD", score=D(score), weight=D(weight), source=source,
    )


# --------------------------------------------------------------------------
# Lexicon and parsing (adapted from reddit / stocktwits)
# --------------------------------------------------------------------------

class LexiconTests(unittest.TestCase):
    def test_positive_negative_neutral(self):
        self.assertGreater(score_text("to the moon, huge rally"), 0)
        self.assertLess(score_text("this will crash and dump"), 0)
        self.assertEqual(score_text("the meeting is on tuesday"), D("0"))

    def test_score_is_bounded(self):
        self.assertLessEqual(score_text("buy buy buy moon rally"), D("1"))
        self.assertGreaterEqual(score_text("sell dump crash fear"), D("-1"))

    def test_strips_cashtag_and_punctuation(self):
        # "$bullish!" must still register as the bullish keyword.
        self.assertGreater(score_text("$bullish! calls"), 0)


class StockTwitsParseTests(unittest.TestCase):
    def test_user_label_is_authoritative(self):
        msgs = [
            {"created_at": "2026-07-29T12:00:00Z", "body": "eh, whatever",
             "entities": {"sentiment": {"basic": "Bullish"}}},
            {"created_at": "2026-07-29T12:00:30Z", "body": "looks great",
             "entities": {"sentiment": {"basic": "Bearish"}}},
        ]
        out = parse_stocktwits_messages("X-USD", msgs)
        self.assertEqual(out[0].score, D("1"))     # label wins over neutral body
        self.assertEqual(out[1].score, D("-1"))    # label wins over positive body
        self.assertEqual(out[0].source, "stocktwits")
        self.assertEqual(out[0].ts.tzinfo, UTC)

    def test_unlabeled_falls_back_to_lexicon(self):
        msgs = [{"created_at": "2026-07-29T12:00:00Z", "body": "buy the breakout, calls",
                 "entities": {"sentiment": None}}]
        out = parse_stocktwits_messages("X-USD", msgs)
        self.assertEqual(len(out), 1)
        self.assertGreater(out[0].score, 0)

    def test_engagement_becomes_weight(self):
        msgs = [{"created_at": "2026-07-29T12:00:00Z", "body": "x",
                 "entities": {"sentiment": {"basic": "Bullish"}},
                 "likes": {"total": 4}, "reshares": {"reshared_count": 2}}]
        out = parse_stocktwits_messages("X-USD", msgs)
        self.assertEqual(out[0].weight, D("7"))    # 1 + 4 + 2

    def test_unparseable_timestamp_is_dropped(self):
        msgs = [{"created_at": "not-a-date", "body": "moon",
                 "entities": {"sentiment": {"basic": "Bullish"}}}]
        self.assertEqual(parse_stocktwits_messages("X-USD", msgs), [])


class RedditParseTests(unittest.TestCase):
    def test_lexicon_polarity_and_engagement_weight(self):
        posts = [
            {"created_utc": 1785000000, "title": "strong rally incoming",
             "selftext": "", "score": 50, "num_comments": 10},
            {"created_utc": 1785000100, "title": "huge crash",
             "selftext": "sell now, dump it", "score": -3, "num_comments": 5},
        ]
        out = parse_reddit_posts("X-USD", "reddit:stocks", posts)
        self.assertGreater(out[0].score, 0)
        self.assertLess(out[1].score, 0)
        self.assertEqual(out[0].weight, D("61"))   # 1 + 50 + 10
        self.assertEqual(out[1].weight, D("9"))    # 1 + |−3| + 5
        self.assertEqual(out[0].source, "reddit:stocks")


# --------------------------------------------------------------------------
# fetch_sentiment (injectable fetcher, no network)
# --------------------------------------------------------------------------

def fake_fetcher(mentions):
    """A fetcher that replays a canned list and records its calls."""
    calls = []

    def fetch(symbol, since=None):
        calls.append((symbol, since))
        return list(mentions)

    fetch.calls = calls
    return fetch


class FetchSentimentTests(unittest.TestCase):
    def test_uses_injected_fetcher_and_sorts_ascending(self):
        raw = [m(2, score="0.5"), m(0, score="1"), m(1, score="-1")]
        fetch = fake_fetcher(raw)
        out = fetch_sentiment("X-USD", fetcher=fetch)
        self.assertEqual([x.ts.minute for x in out], [0, 1, 2])
        self.assertEqual(fetch.calls[0][0], "X-USD")

    def test_since_filters_older_mentions(self):
        raw = [m(0), m(5), m(10)]
        since = datetime(2026, 7, 29, 12, 5, tzinfo=UTC)
        out = fetch_sentiment("X-USD", since=since, fetcher=fake_fetcher(raw))
        self.assertEqual([x.ts.minute for x in out], [5, 10])   # 12:00 dropped


# --------------------------------------------------------------------------
# Aggregation (mirrors the trades-feed MinuteBarAggregator guarantees)
# --------------------------------------------------------------------------

class AggregatorTests(unittest.TestCase):
    def test_forming_bucket_is_not_emitted(self):
        agg = SentimentBucketAggregator("X-USD")
        self.assertEqual(agg.add([m(0, 1), m(0, 30), m(0, 59)]), [])
        self.assertTrue(agg.has_open_bucket)

    def test_bucket_completes_when_a_later_mention_arrives(self):
        agg = SentimentBucketAggregator("X-USD")
        scores = agg.add([m(0, 10), m(0, 50), m(1, 5)])
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].ts, datetime(2026, 7, 29, 12, 1, tzinfo=UTC))

    def test_close_time_convention(self):
        agg = SentimentBucketAggregator("X-USD")
        agg.add([m(3, 15)])
        s = agg.flush()
        # a mention in the 12:03 minute -> score stamped at its close, 12:04.
        self.assertEqual(s.ts, datetime(2026, 7, 29, 12, 4, tzinfo=UTC))

    def test_volume_weighted_score(self):
        agg = SentimentBucketAggregator("X-USD")
        # (1*3 + -1*1) / (3 + 1) = 2/4 = 0.5, mentions=2, volume=4.
        agg.add([m(0, 10, score="1", weight="3"), m(0, 40, score="-1", weight="1")])
        s = agg.flush()
        self.assertEqual(s.score, D("0.5"))
        self.assertEqual(s.mentions, 2)
        self.assertEqual(s.volume, D("4"))

    def test_zero_weight_falls_back_to_simple_mean(self):
        agg = SentimentBucketAggregator("X-USD")
        agg.add([m(0, 1, score="1", weight="0"), m(0, 2, score="0", weight="0")])
        s = agg.flush()
        self.assertEqual(s.score, D("0.5"))        # (1 + 0) / 2

    def test_out_of_order_mention_is_skipped(self):
        agg = SentimentBucketAggregator("X-USD")
        agg.add([m(5, 0)])                          # opens 12:05
        scores = agg.add([m(4, 0)])                 # late 12:04 must not corrupt
        self.assertEqual(scores, [])
        self.assertEqual(agg.open_bucket_start, datetime(2026, 7, 29, 12, 5, tzinfo=UTC))

    def test_five_minute_bucketing(self):
        agg = SentimentBucketAggregator("X-USD", interval_minutes=5)
        scores = agg.add([m(0), m(3), m(4, 59), m(5, 0)])
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].interval, "5m")
        self.assertEqual(scores[0].ts, datetime(2026, 7, 29, 12, 5, tzinfo=UTC))

    def test_flush_returns_and_clears(self):
        agg = SentimentBucketAggregator("X-USD")
        agg.add([m(0)])
        self.assertIsNotNone(agg.flush())
        self.assertIsNone(agg.flush())
        self.assertFalse(agg.has_open_bucket)


# --------------------------------------------------------------------------
# SentimentFeed (fetch + aggregate, offline)
# --------------------------------------------------------------------------

class FeedTests(unittest.TestCase):
    def test_series_aggregates_to_close_aligned_scores(self):
        raw = [m(0, 10, score="1"), m(0, 50, score="1"), m(1, 5, score="-1"), m(2, 0)]
        feed = SentimentFeed(fetcher=fake_fetcher(raw))
        series = feed.series("X-USD")
        self.assertEqual([s.ts.minute for s in series], [1, 2, 3])   # 3 = flushed tail
        self.assertEqual(series[0].score, D("1"))
        self.assertEqual(series[0].mentions, 2)

    def test_flush_false_drops_the_forming_tail(self):
        raw = [m(0, 10), m(1, 5)]
        feed = SentimentFeed(fetcher=fake_fetcher(raw))
        self.assertEqual(len(feed.series("X-USD", flush=True)), 2)
        self.assertEqual(len(feed.series("X-USD", flush=False)), 1)

    def test_mapping_is_keyed_by_close_time(self):
        raw = [m(0, 10), m(1, 5)]
        feed = SentimentFeed(fetcher=fake_fetcher(raw))
        mp = feed.mapping("X-USD")
        self.assertIn(datetime(2026, 7, 29, 12, 1, tzinfo=UTC), mp)
        self.assertIsInstance(next(iter(mp.values())), SentimentScore)


# --------------------------------------------------------------------------
# SentimentAgent
# --------------------------------------------------------------------------

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def ts_of(i: int) -> datetime:
    """Close-time of the i-th bar (0-based), matching ``bars`` below."""
    return T0 + timedelta(days=i + 1)


def bars(closes, interval="1d"):
    out = []
    for i, c in enumerate(closes):
        c = float(c)
        out.append(Bar(instrument_key=INST.key, ts=ts_of(i), interval=interval,
                       open=D(str(c)), high=D(str(c + 0.5)), low=D(str(c - 0.5)),
                       close=D(str(c)), volume=D("1")))
    return out


def sscore(i, score, mentions=10, volume="20"):
    """A SentimentScore stamped at the i-th bar's close-time."""
    return SentimentScore(ts=ts_of(i), symbol="X-USD", score=D(str(score)),
                          mentions=mentions, volume=D(volume), source="stocktwits",
                          interval="1d")


def agent(sentiment=None, **kw):
    params = dict(vol_lookback=10, entry_threshold="0.35", exit_threshold="0.10",
                  vol_ceiling="5.0", interval="1d", continuous=False,
                  min_annual_vol="0.001")
    params.update(kw)
    return SentimentAgent(sentiment=sentiment or {}, **params)


# A gently oscillating price path: enough return to have non-zero volatility
# (so the min-vol entry guard does not block), but calm enough to stay under the
# ceiling and clear of the stops.
WIGGLE = [100.0, 100.5] * 7          # 14 bars, indices 0..13


def run_series(a, series):
    events = []
    cursor = ReplayCursor(series)
    while True:
        for intent in a.on_bar(cursor, INST, Mandate.open_mandate([INST.key], hours=10**6)):
            events.append((cursor.index, intent.side, intent.strategy, intent))
        if not cursor.advance():
            break
    return events


class AgentSignalTests(unittest.TestCase):
    def test_warmup_is_vol_lookback_plus_two(self):
        self.assertEqual(agent(vol_lookback=10).warmup_bars, 12)

    def test_buys_on_positive_sentiment(self):
        # Sentiment appears only on the final bar; the buy must land there.
        a = agent({ts_of(13): sscore(13, "0.6")})
        events = run_series(a, bars(WIGGLE))
        buys = [e for e in events if e[1] is Side.BUY]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0][0], 13)
        self.assertEqual(buys[0][2], "sentiment_v1")

    def test_does_not_buy_on_negative_sentiment(self):
        a = agent({ts_of(13): sscore(13, "-0.5")})
        events = run_series(a, bars(WIGGLE))
        self.assertEqual([e for e in events if e[1] is Side.BUY], [])

    def test_does_not_buy_below_entry_threshold(self):
        a = agent({ts_of(13): sscore(13, "0.20")})   # positive but < 0.35
        events = run_series(a, bars(WIGGLE))
        self.assertEqual([e for e in events if e[1] is Side.BUY], [])

    def test_no_sentiment_never_trades(self):
        events = run_series(agent({}), bars(WIGGLE))
        self.assertEqual(events, [])

    def test_rising_sentiment_buys_falling_does_not(self):
        # min_mentions gates the first (high) bar so the agent stays flat and
        # records a prior score; the second bar's entry then turns on slope only.
        rising = agent(
            {ts_of(12): sscore(12, "0.40", mentions=1),      # blocked: too few mentions
             ts_of(13): sscore(13, "0.60", mentions=20)},    # slope +0.20 -> buy
            min_mentions=5,
        )
        falling = agent(
            {ts_of(12): sscore(12, "0.60", mentions=1),      # blocked: too few mentions
             ts_of(13): sscore(13, "0.40", mentions=20)},    # slope -0.20 -> no buy
            min_mentions=5,
        )
        self.assertTrue(any(e[1] is Side.BUY for e in run_series(rising, bars(WIGGLE))))
        self.assertFalse(any(e[1] is Side.BUY for e in run_series(falling, bars(WIGGLE))))

    def test_sentiment_must_align_to_the_bar_close(self):
        # A score stamped an hour off the bar close is never looked up.
        off = ts_of(13) + timedelta(hours=1)
        a = agent({off: SentimentScore(ts=off, symbol="X-USD", score=D("0.9"),
                                       mentions=10, volume=D("20"), source="stocktwits")})
        self.assertEqual([e for e in run_series(a, bars(WIGGLE)) if e[1] is Side.BUY], [])

    def test_exits_on_decay(self):
        a = agent({ts_of(12): sscore(12, "0.6"), ts_of(13): sscore(13, "0.05")})
        events = run_series(a, bars(WIGGLE))
        sides = [e[1] for e in events]
        self.assertIn(Side.BUY, sides)
        self.assertIn(Side.SELL, sides)
        sell = next(e for e in events if e[1] is Side.SELL)
        self.assertEqual(sell[2], "sentiment_v1:sentiment_decay")

    def test_exits_on_hard_stop(self):
        # Long at bar 12, then a 10% close-to-close drop with mood still positive.
        closes = [100.0, 101.0] * 6 + [100.0, 90.0]     # indices 0..13, close[13]=90
        a = agent({ts_of(12): sscore(12, "0.6"), ts_of(13): sscore(13, "0.6")})
        events = run_series(a, bars(closes))
        sell = next(e for e in events if e[1] is Side.SELL)
        self.assertEqual(sell[2], "sentiment_v1:hard_stop")

    def test_exit_is_evaluated_before_entry_guards(self):
        # Volatility spiking would block a fresh entry, but must not block the
        # exit of an open position (L1): the sell still fires as a vol_spike.
        closes = [100.0, 101.0] * 6 + [100.0, 130.0]    # big jump at bar 13
        a = agent({ts_of(12): sscore(12, "0.6"), ts_of(13): sscore(13, "0.6")},
                  vol_ceiling="0.50")                    # tight, so bar 13 breaches
        events = run_series(a, bars(closes))
        sell = next(e for e in events if e[1] is Side.SELL)
        self.assertEqual(sell[2], "sentiment_v1:vol_spike")

    def test_edge_is_a_clipped_function_of_level_and_slope(self):
        a = agent({ts_of(13): sscore(13, "0.9")})
        buy = next(e for e in run_series(a, bars(WIGGLE)) if e[1] is Side.BUY)
        intent = buy[3]
        self.assertGreaterEqual(intent.edge, D("0.005"))   # floor
        self.assertLessEqual(intent.edge, D("0.06"))       # ceiling
        self.assertGreater(intent.edge, D("0.005"))        # a strong mood clears the floor

    def test_confidence_rises_with_mention_volume(self):
        low = agent({ts_of(13): sscore(13, "0.9", volume="1")})
        high = agent({ts_of(13): sscore(13, "0.9", volume="100")})
        lo = next(e for e in run_series(low, bars(WIGGLE)) if e[1] is Side.BUY)[3]
        hi = next(e for e in run_series(high, bars(WIGGLE)) if e[1] is Side.BUY)[3]
        self.assertGreater(hi.confidence, lo.confidence)

    def test_fit_resets_state(self):
        a = agent({ts_of(13): sscore(13, "0.6")})
        run_series(a, bars(WIGGLE))
        self.assertTrue(a.is_long)
        a.fit([])
        self.assertFalse(a.is_long)
        self.assertIsNone(a.last_reading)

    def test_only_requests_the_warmup_window(self):
        class RecordingCursor(ReplayCursor):
            def __init__(self, b):
                super().__init__(b)
                self.max_lookback = 0

            def history(self, lookback=None):
                if lookback is not None:
                    self.max_lookback = max(self.max_lookback, lookback)
                return super().history(lookback)

        a = agent({ts_of(13): sscore(13, "0.6")})
        cur = RecordingCursor(bars(WIGGLE))
        while True:
            a.on_bar(cur, INST, Mandate.open_mandate([INST.key], hours=10**6))
            if not cur.advance():
                break
        self.assertLessEqual(cur.max_lookback, a.warmup_bars)


if __name__ == "__main__":
    unittest.main()
