"""Tests for the Kraken /Trades 1-minute backfill. No network.

The load-bearing behaviour, and the reason this feed is not just "aggregate each
page": a minute split across a page boundary must produce ONE correct bar, and
the minute still forming must never be written. Both are the same failure the
OHLC and WebSocket feeds guard against -- a "close" written for a period that is
not over -- reached from a different direction.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.core.types import AssetClass, Bar, Instrument, VenueId
from spintrader.data.kraken_feed import FeedError
from spintrader.data.kraken_trades import (
    MinuteBarAggregator, TradeTick, backfill_1m, fetch_trades,
)

D = Decimal
UTC = timezone.utc


def t(minute: int, second: int = 0, price="100", volume="1", *, hour=12, day=1):
    return TradeTick(
        ts=datetime(2026, 7, day, hour, minute, second, tzinfo=UTC),
        price=D(price), volume=D(volume), side="b", order_type="m", trade_id=0,
    )


INSTRUMENT = Instrument(
    symbol="BTC-USD", asset_class=AssetClass.CRYPTO, venue=VenueId.KRAKEN,
    venue_symbol="XXBTZUSD",
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeStore:
    def __init__(self, last_ts: datetime | None = None):
        self.written: list[Bar] = []
        self.instruments: list[Instrument] = []
        self._last_ts = last_ts

    def upsert_instrument(self, instrument):
        self.instruments.append(instrument)

    def write_bars(self, bars, source):
        self.source = source
        self.written.extend(bars)
        return len(bars)

    def last_bar_ts(self, instrument_key, interval):
        return self._last_ts

    def bar_coverage(self, instrument_key, interval):
        if not self.written:
            return {"bars": 0, "first": None, "last": None}
        stamps = [b.ts for b in self.written]
        return {"bars": len(self.written), "first": min(stamps), "last": max(stamps)}


class FakeVenue:
    def __init__(self):
        self._session = None

    def resolve(self, symbol):
        return INSTRUMENT


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Records the params of each GET and replays a scripted payload."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params or {}))
        return FakeResponse(self._payloads.pop(0))


def kraken_payload(rows, last="123", error=None):
    return {"error": error or [], "result": {"XXBTZUSD": rows, "last": last}}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

class AggregatorTests(unittest.TestCase):
    def test_forming_minute_is_not_emitted(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        # All trades in one minute: nothing is complete yet.
        self.assertEqual(agg.add([t(0, 1), t(0, 30), t(0, 59)]), [])
        self.assertTrue(agg.has_open_bucket)

    def test_minute_completes_when_a_later_trade_arrives(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        bars = agg.add([t(0, 10), t(0, 50), t(1, 5)])
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].ts, datetime(2026, 7, 1, 12, 1, tzinfo=UTC))

    def test_ohlcv_and_vwap(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        agg.add([
            t(0, 1, price="100", volume="1"),
            t(0, 2, price="110", volume="2"),   # high
            t(0, 3, price="90", volume="3"),    # low
            t(0, 4, price="105", volume="4"),   # close
        ])
        bar = agg.flush()
        self.assertEqual(bar.open, D("100"))
        self.assertEqual(bar.high, D("110"))
        self.assertEqual(bar.low, D("90"))
        self.assertEqual(bar.close, D("105"))
        self.assertEqual(bar.volume, D("10"))
        self.assertEqual(bar.trades, 4)
        # vwap = (100*1 + 110*2 + 90*3 + 105*4) / 10 = (100+220+270+420)/10 = 101.0
        self.assertEqual(bar.vwap, D("101"))

    def test_close_time_convention(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        agg.add([t(3, 15)])
        bar = agg.flush()
        # trade in the 12:03 minute -> bar stamped at its close, 12:04.
        self.assertEqual(bar.ts, datetime(2026, 7, 1, 12, 4, tzinfo=UTC))

    def test_trade_on_minute_boundary_opens_the_new_bucket(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        # 12:01:00 belongs to [12:01, 12:02), so it closes the 12:00 bucket.
        bars = agg.add([t(0, 30), t(1, 0)])
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].ts, datetime(2026, 7, 1, 12, 1, tzinfo=UTC))

    def test_carryover_across_pages_merges_one_minute(self):
        # A minute split across two add() calls (a page boundary) must produce a
        # single bar carrying trades from both pages, not two partial bars.
        agg = MinuteBarAggregator("kraken:BTC-USD")
        page1 = agg.add([t(0, 10, price="100", volume="1"),
                         t(1, 10, price="200", volume="1")])   # 12:01 opens, held
        self.assertEqual(len(page1), 1)                        # only 12:00 done
        page2 = agg.add([t(1, 40, price="250", volume="2"),    # same 12:01 minute
                         t(2, 5, price="300", volume="1")])    # 12:02 -> 12:01 done
        self.assertEqual(len(page2), 1)
        merged = page2[0]
        self.assertEqual(merged.ts, datetime(2026, 7, 1, 12, 2, tzinfo=UTC))
        self.assertEqual(merged.open, D("200"))     # first trade of the minute
        self.assertEqual(merged.close, D("250"))    # last trade of the minute
        self.assertEqual(merged.high, D("250"))
        self.assertEqual(merged.volume, D("3"))     # 1 + 2, both pages
        self.assertEqual(merged.trades, 2)

    def test_out_of_order_trade_is_skipped(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        agg.add([t(5, 0)])            # opens 12:05
        # A late trade for 12:04 must not corrupt or reopen an earlier bucket.
        bars = agg.add([t(4, 0)])
        self.assertEqual(bars, [])
        self.assertEqual(agg.open_bucket_start, datetime(2026, 7, 1, 12, 5, tzinfo=UTC))

    def test_five_minute_bucketing(self):
        agg = MinuteBarAggregator("kraken:BTC-USD", interval_minutes=5)
        # 12:00..12:04 in one bucket; 12:05 opens the next.
        bars = agg.add([t(0), t(3), t(4, 59), t(5, 0)])
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].interval, "5m")
        self.assertEqual(bars[0].ts, datetime(2026, 7, 1, 12, 5, tzinfo=UTC))

    def test_flush_returns_and_clears(self):
        agg = MinuteBarAggregator("kraken:BTC-USD")
        agg.add([t(0)])
        self.assertIsNotNone(agg.flush())
        self.assertIsNone(agg.flush())          # nothing left
        self.assertFalse(agg.has_open_bucket)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

class FetchTests(unittest.TestCase):
    def test_parses_rows_and_cursor(self):
        rows = [
            ["100.5", "0.25", 1785000000.5, "b", "l", "", 1],
            ["101.0", "0.10", 1785000005.0, "s", "m", "", 2],
        ]
        session = FakeSession([kraken_payload(rows, last="999")])
        ticks, cursor = fetch_trades(INSTRUMENT, since=0, session=session)
        self.assertEqual(cursor, "999")
        self.assertEqual(len(ticks), 2)
        self.assertEqual(ticks[0].price, D("100.5"))
        self.assertEqual(ticks[0].volume, D("0.25"))
        self.assertEqual(ticks[0].ts.tzinfo, UTC)
        self.assertEqual(ticks[1].side, "s")

    def test_error_array_raises(self):
        session = FakeSession([kraken_payload([], error=["EGeneral:Invalid"])])
        with self.assertRaises(FeedError):
            fetch_trades(INSTRUMENT, since=0, session=session)

    def test_since_datetime_becomes_seconds(self):
        session = FakeSession([kraken_payload([])])
        fetch_trades(INSTRUMENT, since=datetime(2023, 1, 1, tzinfo=UTC), session=session)
        self.assertEqual(session.calls[0]["since"],
                         str(int(datetime(2023, 1, 1, tzinfo=UTC).timestamp())))

    def test_since_cursor_passes_through(self):
        session = FakeSession([kraken_payload([])])
        fetch_trades(INSTRUMENT, since="1672535479745656494", session=session)
        self.assertEqual(session.calls[0]["since"], "1672535479745656494")

    def test_pair_param_is_venue_symbol(self):
        session = FakeSession([kraken_payload([])])
        fetch_trades(INSTRUMENT, session=session)
        self.assertEqual(session.calls[0]["pair"], "XXBTZUSD")


# --------------------------------------------------------------------------
# Backfill loop
# --------------------------------------------------------------------------

def scripted_fetcher(pages):
    """Return a fetcher that yields (ticks, cursor) pages, recording `since`."""
    state = {"i": 0, "since": []}

    def fetch(instrument, since):
        state["since"].append(since)
        i = state["i"]
        state["i"] += 1
        if i < len(pages):
            return pages[i]
        return ([], None)          # exhausted -> live edge

    fetch.state = state
    return fetch


class BackfillTests(unittest.TestCase):
    def setUp(self):
        # A reference "now" so the live-edge check is deterministic.
        self.now = datetime(2026, 7, 1, 13, 0, tzinfo=UTC)

    def test_writes_complete_bars_and_drops_forming_minute(self):
        store = FakeStore()
        # Page 1: minute 12:00 completes (12:01 seen); 12:01 held.
        # Page 2: 12:01 continues, 12:02 seen -> 12:01 completes; 12:02 held.
        pages = [
            ([t(0, 10, price="100", volume="1"),
              t(1, 10, price="110", volume="1")], "c1"),
            ([t(1, 40, price="120", volume="2"),
              t(2, 5, price="130", volume="1")], "c2"),
        ]
        result = backfill_1m(
            store, FakeVenue(), "BTC-USD",
            start=0, fetcher=scripted_fetcher(pages), sleep_s=0, now=self.now,
        )
        stamps = [b.ts for b in store.written]
        self.assertEqual(stamps, [
            datetime(2026, 7, 1, 12, 1, tzinfo=UTC),    # the 12:00 minute
            datetime(2026, 7, 1, 12, 2, tzinfo=UTC),    # the 12:01 minute
        ])
        # The 12:02 minute was still forming and must not appear.
        self.assertNotIn(datetime(2026, 7, 1, 12, 3, tzinfo=UTC), stamps)
        # The 12:01 bar merged both pages.
        merged = store.written[1]
        self.assertEqual(merged.volume, D("3"))
        self.assertEqual(merged.close, D("120"))
        self.assertTrue(result["reached_live_edge"])
        self.assertEqual(store.source, "kraken_trades")

    def test_resume_uses_last_stored_bar(self):
        last = datetime(2026, 7, 1, 12, 30, tzinfo=UTC)
        store = FakeStore(last_ts=last)
        fetch = scripted_fetcher([([t(31, 0)], "c1")])
        backfill_1m(store, FakeVenue(), "BTC-USD",
                    fetcher=fetch, sleep_s=0, now=self.now)
        # First call resumes from the last stored bar's close time.
        self.assertEqual(fetch.state["since"][0], last)

    def test_start_overrides_resume(self):
        store = FakeStore(last_ts=datetime(2026, 7, 1, 12, 30, tzinfo=UTC))
        fetch = scripted_fetcher([([t(0)], "c1")])
        backfill_1m(store, FakeVenue(), "BTC-USD",
                    start=0, fetcher=fetch, sleep_s=0, now=self.now)
        self.assertEqual(fetch.state["since"][0], 0)

    def test_advances_cursor_between_pages(self):
        store = FakeStore()
        pages = [([t(0), t(1)], "cursor-A"), ([t(2), t(3)], "cursor-B")]
        fetch = scripted_fetcher(pages)
        backfill_1m(store, FakeVenue(), "BTC-USD",
                    start=0, fetcher=fetch, sleep_s=0, now=self.now)
        # Page 2 must be requested with page 1's returned cursor.
        self.assertEqual(fetch.state["since"][1], "cursor-A")

    def test_stops_when_cursor_stalls(self):
        store = FakeStore()
        # Same cursor twice: no forward progress, must terminate.
        pages = [([t(0), t(1)], "stuck"), ([t(1), t(2)], "stuck")]
        result = backfill_1m(store, FakeVenue(), "BTC-USD",
                             start=0, fetcher=scripted_fetcher(pages),
                             sleep_s=0, now=self.now)
        self.assertTrue(result["reached_live_edge"])
        self.assertLessEqual(result["pages"], 2)

    def test_max_pages_caps_the_walk(self):
        store = FakeStore()
        pages = [([t(m, 0), t(m, 30)], f"c{m}") for m in range(10)]
        result = backfill_1m(store, FakeVenue(), "BTC-USD",
                             start=0, fetcher=scripted_fetcher(pages),
                             sleep_s=0, max_pages=3, now=self.now)
        self.assertEqual(result["pages"], 3)
        self.assertFalse(result["reached_live_edge"])

    def test_live_edge_stops_the_walk(self):
        store = FakeStore()
        # A trade inside the last minute before `now` means we have caught up.
        recent = self.now - timedelta(seconds=30)
        tick = TradeTick(ts=recent, price=D("100"), volume=D("1"),
                         side="b", order_type="m", trade_id=0)
        pages = [([t(0, 0), tick], "c1")]
        result = backfill_1m(store, FakeVenue(), "BTC-USD",
                             start=0, fetcher=scripted_fetcher(pages),
                             sleep_s=0, now=self.now)
        self.assertTrue(result["reached_live_edge"])

    def test_end_bound_stops_and_excludes_later_bars(self):
        store = FakeStore()
        end = datetime(2026, 7, 1, 12, 2, tzinfo=UTC)
        pages = [([t(0, 10), t(1, 10), t(2, 10), t(3, 10)], "c1")]
        result = backfill_1m(store, FakeVenue(), "BTC-USD",
                             start=0, end=end, fetcher=scripted_fetcher(pages),
                             sleep_s=0, now=self.now)
        self.assertTrue(result["hit_end"])
        # No bar with a close after the end bound.
        self.assertTrue(all(b.ts <= end for b in store.written))

    def test_instrument_is_upserted(self):
        store = FakeStore()
        backfill_1m(store, FakeVenue(), "BTC-USD",
                    start=0, fetcher=scripted_fetcher([]), sleep_s=0, now=self.now)
        self.assertEqual(store.instruments, [INSTRUMENT])


if __name__ == "__main__":
    unittest.main()
