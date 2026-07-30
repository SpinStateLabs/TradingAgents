"""Tests for bar resampling (the tradeable-horizon lever). No network."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.core.types import Bar
from spintrader.data.resample import resample_bars

D = Decimal
UTC = timezone.utc
KEY = "kraken:BTC-USD"


def m1(i, o, h, l, c, v="1", vwap=None, trades=1):
    # 1m bar closing at 12:00 + (i+1) minutes.
    t0 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
    return Bar(instrument_key=KEY, ts=t0 + timedelta(minutes=i + 1), interval="1m",
               open=D(str(o)), high=D(str(h)), low=D(str(l)), close=D(str(c)),
               volume=D(str(v)), vwap=D(str(vwap)) if vwap is not None else None,
               trades=trades)


class ResampleTests(unittest.TestCase):
    def test_hourly_aggregation(self):
        # Two clean hours of 1m bars (120), then resample to 1h.
        bars = [m1(i, 100 + i, 100 + i + 2, 100 + i - 2, 100 + i + 1, v="1")
                for i in range(120)]
        hourly = resample_bars(bars, "1h")
        self.assertEqual(len(hourly), 2)
        first = hourly[0]
        self.assertEqual(first.interval, "1h")
        # bucket [12:00,13:00) closes at 13:00
        self.assertEqual(first.ts, datetime(2026, 7, 1, 13, 0, tzinfo=UTC))
        self.assertEqual(first.open, bars[0].open)          # first sub-bar's open
        self.assertEqual(first.close, bars[59].close)       # last sub-bar's close
        self.assertEqual(first.high, max(b.high for b in bars[:60]))
        self.assertEqual(first.low, min(b.low for b in bars[:60]))
        self.assertEqual(first.volume, D("60"))             # 60 * 1
        self.assertEqual(first.trades, 60)

    def test_volume_weighted_vwap(self):
        bars = [m1(0, 100, 100, 100, 100, v="1", vwap="100"),
                m1(1, 110, 110, 110, 110, v="3", vwap="110")]
        # both in the 12:00 hour bucket
        hourly = resample_bars(bars, "1h")
        self.assertEqual(len(hourly), 1)
        # vwap = (100*1 + 110*3) / 4 = 107.5
        self.assertEqual(hourly[0].vwap, D("107.5"))

    def test_close_time_bucketing_is_by_open(self):
        # A bar closing exactly at 13:00 (opening 12:59) belongs to the 12:00 hour.
        bars = [m1(58, 100, 101, 99, 100), m1(59, 100, 101, 99, 100)]  # close 12:59, 13:00
        hourly = resample_bars(bars, "1h")
        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly[0].ts, datetime(2026, 7, 1, 13, 0, tzinfo=UTC))

    def test_partial_trailing_bucket_is_kept(self):
        bars = [m1(i, 100, 101, 99, 100) for i in range(75)]   # 1h15m -> 2 buckets
        hourly = resample_bars(bars, "1h")
        self.assertEqual(len(hourly), 2)
        self.assertEqual(hourly[1].volume, D("15"))            # the partial hour

    def test_rejects_non_dividing_or_finer_target(self):
        bars = [m1(0, 100, 100, 100, 100)]
        with self.assertRaises(ValueError):
            resample_bars(bars, "1m")          # not coarser
        with self.assertRaises(ValueError):
            resample_bars(bars, "nonsense")

    def test_empty(self):
        self.assertEqual(resample_bars([], "1h"), [])


if __name__ == "__main__":
    unittest.main()
