"""Tests for Kraken OHLCV ingestion. No network.

The load-bearing tests are the ones about the in-progress bar and the
timestamp convention. Both are silent-corruption bugs: nothing errors, the
data merely becomes subtly predictive of itself, and every backtest built on
it reports alpha that does not exist.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

import requests

from spintrader.core.types import AssetClass, Instrument, VenueId
from spintrader.data.kraken_feed import (
    FeedError, INTERVAL_MINUTES, KRAKEN_MAX_BARS, available_history,
    fetch_ohlc, interval_delta,
)

D = Decimal
BTC = Instrument("BTC-USD", AssetClass.CRYPTO, VenueId.KRAKEN, "XXBTZUSD")

# 2026-07-01 12:00 UTC
BASE_TS = int(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc).timestamp())
HOUR = 3600


def ohlc_row(ts: int, o="100", h="110", l="90", c="105", vwap="102", vol="12", n=7):
    return [ts, o, h, l, c, vwap, vol, n]


def feed_session(rows, error=None):
    session = mock.Mock(spec=requests.Session)
    body = {"error": error or [], "result": {"XXBTZUSD": rows, "last": rows[-1][0] if rows else 0}}
    response = mock.Mock()
    response.json.return_value = body
    response.raise_for_status.return_value = None
    session.get.return_value = response
    return session


class IntervalTests(unittest.TestCase):
    def test_known_intervals_map_to_minutes(self):
        self.assertEqual(INTERVAL_MINUTES["1h"], 60)
        self.assertEqual(INTERVAL_MINUTES["1d"], 1440)

    def test_unsupported_interval_lists_alternatives(self):
        with self.assertRaises(FeedError) as ctx:
            interval_delta("3m")
        self.assertIn("available", str(ctx.exception))

    def test_delta_matches_the_interval(self):
        self.assertEqual(interval_delta("4h"), timedelta(hours=4))


class TimestampConventionTests(unittest.TestCase):
    """Kraken timestamps bars by OPEN time; this system uses CLOSE time.

    Storing open-time is the most common source of off-by-one-bar lookahead:
    a bar labelled 12:00 that actually covers 12:00-13:00 looks, to anything
    reading it at 12:30, like knowledge of the future.
    """

    def test_open_time_converted_to_close_time(self):
        session = feed_session([ohlc_row(BASE_TS)])
        now = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)
        bars = fetch_ohlc(BTC, "1h", session=session, now=now)
        self.assertEqual(len(bars), 1)
        # Kraken said 12:00 (open); we store 13:00 (close).
        self.assertEqual(bars[0].ts, datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc))

    def test_daily_bars_shift_by_a_day(self):
        session = feed_session([ohlc_row(BASE_TS)])
        now = datetime(2026, 7, 5, tzinfo=timezone.utc)
        bars = fetch_ohlc(BTC, "1d", session=session, now=now)
        self.assertEqual(bars[0].ts, datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc))

    def test_stored_timestamps_are_utc(self):
        session = feed_session([ohlc_row(BASE_TS)])
        now = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)
        self.assertEqual(fetch_ohlc(BTC, "1h", session=session, now=now)[0].ts.tzinfo,
                         timezone.utc)


class IncompleteBarTests(unittest.TestCase):
    """Kraken returns the in-progress bar with the closed ones."""

    def test_in_progress_bar_is_dropped(self):
        # Two bars: 12:00-13:00 (closed) and 13:00-14:00 (still forming).
        session = feed_session([ohlc_row(BASE_TS), ohlc_row(BASE_TS + HOUR)])
        now = datetime(2026, 7, 1, 13, 30, tzinfo=timezone.utc)   # mid-bar
        bars = fetch_ohlc(BTC, "1h", session=session, now=now)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].ts, datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc))

    def test_bar_closing_exactly_now_is_kept(self):
        session = feed_session([ohlc_row(BASE_TS)])
        now = datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc)    # exactly the close
        self.assertEqual(len(fetch_ohlc(BTC, "1h", session=session, now=now)), 1)

    def test_all_bars_dropped_when_all_are_in_progress(self):
        session = feed_session([ohlc_row(BASE_TS)])
        now = datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc)
        self.assertEqual(fetch_ohlc(BTC, "1h", session=session, now=now), [])

    def test_incomplete_bar_can_be_kept_explicitly(self):
        # Live marking legitimately wants the forming bar; it must be an
        # explicit choice, never the default.
        session = feed_session([ohlc_row(BASE_TS)])
        now = datetime(2026, 7, 1, 12, 30, tzinfo=timezone.utc)
        bars = fetch_ohlc(BTC, "1h", session=session, now=now, drop_incomplete=False)
        self.assertEqual(len(bars), 1)


class ParsingTests(unittest.TestCase):
    def test_fields_parsed_as_decimal(self):
        session = feed_session([ohlc_row(BASE_TS, o="100.5", h="110.25", l="90.1",
                                         c="105.75", vol="12.5", n=42)])
        now = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)
        bar = fetch_ohlc(BTC, "1h", session=session, now=now)[0]
        self.assertEqual(bar.open, D("100.5"))
        self.assertEqual(bar.high, D("110.25"))
        self.assertEqual(bar.low, D("90.1"))
        self.assertEqual(bar.close, D("105.75"))
        self.assertEqual(bar.volume, D("12.5"))
        self.assertEqual(bar.trades, 42)
        self.assertIsInstance(bar.open, Decimal)

    def test_zero_vwap_becomes_none(self):
        session = feed_session([ohlc_row(BASE_TS, vwap="0")])
        now = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)
        self.assertIsNone(fetch_ohlc(BTC, "1h", session=session, now=now)[0].vwap)

    def test_ordering_preserved(self):
        rows = [ohlc_row(BASE_TS + i * HOUR) for i in range(5)]
        session = feed_session(rows)
        now = datetime(2026, 7, 2, tzinfo=timezone.utc)
        bars = fetch_ohlc(BTC, "1h", session=session, now=now)
        self.assertEqual([b.ts for b in bars], sorted(b.ts for b in bars))

    def test_invalid_ohlc_relationship_rejected_at_construction(self):
        # Bar.__post_init__ guards this; a high below the low is corrupt data
        # and must not reach the store.
        session = feed_session([ohlc_row(BASE_TS, h="50", l="200")])
        now = datetime(2026, 7, 1, 20, 0, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            fetch_ohlc(BTC, "1h", session=session, now=now)


class ErrorTests(unittest.TestCase):
    def test_kraken_error_array_raises(self):
        session = feed_session([ohlc_row(BASE_TS)], error=["EQuery:Unknown asset pair"])
        with self.assertRaises(FeedError) as ctx:
            fetch_ohlc(BTC, "1h", session=session)
        self.assertIn("Unknown asset pair", str(ctx.exception))

    def test_unsupported_interval_raises_before_the_request(self):
        session = feed_session([])
        with self.assertRaises(FeedError):
            fetch_ohlc(BTC, "7m", session=session)
        session.get.assert_not_called()

    def test_since_forwarded_as_epoch_seconds(self):
        session = feed_session([ohlc_row(BASE_TS)])
        since = datetime(2026, 6, 1, tzinfo=timezone.utc)
        fetch_ohlc(BTC, "1h", since=since, session=session,
                   now=datetime(2026, 7, 2, tzinfo=timezone.utc))
        self.assertEqual(session.get.call_args.kwargs["params"]["since"],
                         int(since.timestamp()))


if __name__ == "__main__":
    unittest.main()


class HistoryLimitTests(unittest.TestCase):
    """Kraken serves only the most recent 720 bars, whatever `since` says.

    Verified against the live endpoint 2026-07-29: 1h bars requested from
    2024-01-01 came back covering only the last 30 days. Any code that pages
    backwards is burning rate limit for nothing, so the limit is named and
    asserted rather than discovered again later.
    """

    def test_hourly_reaches_about_a_month(self):
        self.assertEqual(available_history("1h"), timedelta(hours=KRAKEN_MAX_BARS))
        self.assertLess(available_history("1h"), timedelta(days=31))

    def test_daily_reaches_about_two_years(self):
        self.assertGreater(available_history("1d"), timedelta(days=700))

    def test_limit_scales_with_interval(self):
        self.assertEqual(available_history("4h"), available_history("1h") * 4)
