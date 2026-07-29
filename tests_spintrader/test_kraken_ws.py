"""Tests for the Kraken WebSocket minute collector. No network.

The critical behaviour is that the forming candle is never stored. Kraken
revises the in-progress candle on every trade, so persisting it writes a
"close" for a minute that has not ended — the same lookahead trap the REST
feed guards against, and one that is easy to reintroduce because the streamed
payload looks identical to a finished candle.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest import mock

from spintrader.core.types import Bar
from spintrader.data.kraken_ws import (
    CollectorStats, KrakenWSCollector, canonical_symbol, ws_symbol,
)

D = Decimal


class FakeStore:
    def __init__(self, fail: bool = False):
        self.written: list[Bar] = []
        self.fail = fail
        self.calls = 0

    def write_bars(self, bars, source):
        self.calls += 1
        if self.fail:
            raise RuntimeError("database unavailable")
        self.written.extend(bars)
        return len(bars)


def candle(symbol="BTC/USD", begin="2026-07-01T12:00:00.000000Z",
           o="100", h="110", l="90", c="105", volume="3.5", trades=12, vwap="102"):
    return {
        "symbol": symbol, "interval_begin": begin,
        "open": o, "high": h, "low": l, "close": c,
        "volume": volume, "trades": trades, "vwap": vwap,
    }


def ohlc_message(*candles):
    return {"channel": "ohlc", "type": "update", "data": list(candles)}


def make_collector(store=None, **kw):
    return KrakenWSCollector(["BTC-USD"], store or FakeStore(), **kw)


class SymbolMappingTests(unittest.TestCase):
    def test_canonical_to_ws(self):
        self.assertEqual(ws_symbol("BTC-USD"), "BTC/USD")

    def test_ws_to_canonical(self):
        self.assertEqual(canonical_symbol("BTC/USD"), "BTC-USD")

    def test_round_trip(self):
        for symbol in ("BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"):
            with self.subTest(symbol=symbol):
                self.assertEqual(canonical_symbol(ws_symbol(symbol)), symbol)

    def test_ws_v2_uses_modern_tickers_not_rest_asset_codes(self):
        # Unlike the REST API, WS v2 says BTC rather than XBT/XXBT.
        self.assertEqual(ws_symbol("BTC-USD"), "BTC/USD")
        self.assertNotIn("XBT", ws_symbol("BTC-USD"))


class FormingCandleTests(unittest.TestCase):
    """The load-bearing behaviour: never store the candle still forming."""

    def test_first_candle_is_not_written(self):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle()))
        self.assertEqual(store.written, [])
        self.assertEqual(collector.stats.candles_closed, 0)

    def test_revisions_to_the_forming_candle_are_not_written(self):
        # Kraken streams the same interval_begin repeatedly as trades arrive.
        store = FakeStore()
        collector = make_collector(store)
        for close in ("101", "103", "107"):
            collector._handle(ohlc_message(candle(c=close)))
        self.assertEqual(store.written, [])

    def test_candle_is_written_once_the_next_one_starts(self):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:00:00.000000Z", c="105")))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z", c="106")))
        self.assertEqual(len(store.written), 1)
        self.assertEqual(store.written[0].close, D("105"))   # the closed one

    def test_final_revision_is_the_one_stored(self):
        # The value stored must be the last revision before the candle closed,
        # not the first one seen.
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle(c="101")))
        collector._handle(ohlc_message(candle(c="109")))     # revised
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        self.assertEqual(store.written[0].close, D("109"))

    def test_out_of_order_candle_does_not_close_a_newer_one(self):
        # A late message for an earlier minute must not be treated as progress.
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:05:00.000000Z")))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:04:00.000000Z")))
        self.assertEqual(store.written, [])


class TimestampTests(unittest.TestCase):
    def test_interval_begin_converted_to_close_time(self):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:00:00.000000Z")))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        # Kraken said 12:00 (open); we store 12:01 (close).
        self.assertEqual(store.written[0].ts,
                         datetime(2026, 7, 1, 12, 1, tzinfo=timezone.utc))

    def test_five_minute_interval_shifts_by_five(self):
        store = FakeStore()
        collector = make_collector(store, interval_minutes=5)
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:00:00.000000Z")))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:05:00.000000Z")))
        self.assertEqual(store.written[0].ts,
                         datetime(2026, 7, 1, 12, 5, tzinfo=timezone.utc))
        self.assertEqual(store.written[0].interval, "5m")

    def test_timestamps_are_utc_aware(self):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle()))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        self.assertEqual(store.written[0].ts.tzinfo, timezone.utc)


class ParsingTests(unittest.TestCase):
    def _closed_bar(self, **kw):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle(**kw)))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        return store.written[0]

    def test_fields_are_decimal(self):
        bar = self._closed_bar(o="100.5", h="110.25", l="90.1", c="105.75",
                               volume="3.25", trades=42)
        self.assertEqual(bar.open, D("100.5"))
        self.assertEqual(bar.high, D("110.25"))
        self.assertEqual(bar.close, D("105.75"))
        self.assertEqual(bar.volume, D("3.25"))
        self.assertEqual(bar.trades, 42)

    def test_instrument_key_is_venue_qualified(self):
        self.assertEqual(self._closed_bar().instrument_key, "kraken:BTC-USD")

    def test_malformed_candle_is_discarded_not_fatal(self):
        store = FakeStore()
        collector = make_collector(store)
        broken = candle()
        del broken["high"]
        collector._handle(ohlc_message(broken))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        self.assertEqual(store.written, [])
        self.assertEqual(collector.stats.errors, 1)

    def test_inverted_ohlc_is_rejected(self):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle(h="50", l="200")))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        self.assertEqual(store.written, [])
        self.assertEqual(collector.stats.errors, 1)


class MultiSymbolTests(unittest.TestCase):
    def test_symbols_tracked_independently(self):
        store = FakeStore()
        collector = KrakenWSCollector(["BTC-USD", "ETH-USD"], store)
        collector._handle(ohlc_message(candle(symbol="BTC/USD"),
                                       candle(symbol="ETH/USD")))
        self.assertEqual(store.written, [])
        # Only BTC advances; ETH's candle must stay open.
        collector._handle(ohlc_message(
            candle(symbol="BTC/USD", begin="2026-07-01T12:01:00.000000Z")))
        self.assertEqual(len(store.written), 1)
        self.assertEqual(store.written[0].instrument_key, "kraken:BTC-USD")


class NonOhlcMessageTests(unittest.TestCase):
    def test_heartbeat_ignored(self):
        collector = make_collector()
        collector._handle({"channel": "heartbeat"})
        self.assertEqual(collector.stats.errors, 0)

    def test_subscription_ack_ignored(self):
        collector = make_collector()
        collector._handle({"method": "subscribe", "success": True})
        self.assertEqual(collector.stats.errors, 0)

    def test_error_message_counted(self):
        collector = make_collector()
        collector._handle({"error": "Subscription failed"})
        self.assertEqual(collector.stats.errors, 1)


class FlushTests(unittest.TestCase):
    def test_batching_defers_writes(self):
        store = FakeStore()
        collector = make_collector(store, flush_size=3)
        for minute in range(4):
            collector._handle(ohlc_message(
                candle(begin=f"2026-07-01T12:0{minute}:00.000000Z")))
        # Three candles closed after the fourth arrives -> one flush.
        self.assertEqual(store.calls, 1)
        self.assertEqual(len(store.written), 3)

    def test_write_failure_does_not_stop_collection(self):
        # An unbounded buffer would OOM; a visible gap in the store is the
        # better failure.
        store = FakeStore(fail=True)
        collector = make_collector(store)
        collector._handle(ohlc_message(candle()))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        self.assertEqual(collector.stats.errors, 1)
        self.assertEqual(collector._pending, [])
        self.assertEqual(collector.stats.candles_written, 0)

    def test_stats_report_what_happened(self):
        store = FakeStore()
        collector = make_collector(store)
        collector._handle(ohlc_message(candle()))
        collector._handle(ohlc_message(candle(begin="2026-07-01T12:01:00.000000Z")))
        stats = collector.stats.as_dict()
        self.assertEqual(stats["candles_closed"], 1)
        self.assertEqual(stats["candles_written"], 1)
        # `messages` counts raw socket frames and is incremented in the consume
        # loop, not in _handle, so it stays 0 when handlers are driven directly.
        self.assertEqual(stats["messages"], 0)


if __name__ == "__main__":
    unittest.main()
