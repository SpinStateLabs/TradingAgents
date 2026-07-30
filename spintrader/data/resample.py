"""Resample fine bars into coarser ones -- the tradeable-horizon lever.

The measured constraint on this book is blunt: a taker round trip is ~0.52%, and
1-minute BTC sigma is ~0.063%, an ~8-sigma hurdle no signal clears. The hurdle
scales with the square root of the holding period, so at an hourly hold it is
~1.4 sigma and feasible. The system decides every minute but should *hold for
hours*; searching for an edge at 1m is searching where costs structurally win.

This resamples the deep 1-minute history into coarser bars (5m, 15m, 1h, ...) so
the backtester and improvement cycle can look for an edge at a horizon where an
edge can actually survive its own costs.

Aggregation is clock-aligned and preserves the conventions the rest of the system
relies on:

* **Bars are grouped by their OPEN time**, floored to the target interval, so a
  1m bar closing at 10:01 (opening 10:00) lands in the 10:00 hourly bucket. The
  coarse bar's ``ts`` is that bucket's CLOSE time -- the same close-time
  convention enforced everywhere else.
* **OHLCV aggregates the obvious way**: open of the first sub-bar, high/low the
  extremes, close of the last, volume and trade-count summed, vwap
  volume-weighted. All ``Decimal``.
* A trailing bucket with fewer than a full interval of sub-bars is still a real
  (shorter) bar and is kept; the backtester treats it as one observation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.core.types import Bar
from spintrader.data.kraken_feed import INTERVAL_MINUTES

ZERO = Decimal("0")


def _bucket_open(open_time: datetime, minutes: int) -> datetime:
    """Floor an open time to the start of its target-interval bucket."""
    floored = open_time.replace(second=0, microsecond=0)
    discard = (floored.hour * 60 + floored.minute) % minutes
    return floored - timedelta(minutes=discard)


def resample_bars(bars: list[Bar], target_interval: str) -> list[Bar]:
    """Aggregate ``bars`` into ``target_interval`` bars, chronological.

    ``bars`` must be a single instrument's series at one (finer) interval, in
    any order; the result is sorted by close time. The source interval must
    divide the target evenly (e.g. 1m -> 1h, 5m -> 15m); otherwise a ValueError
    is raised rather than silently mis-bucketing.
    """
    if not bars:
        return []

    target_min = INTERVAL_MINUTES.get(target_interval)
    if target_min is None:
        raise ValueError(
            f"unknown target interval {target_interval!r}; "
            f"known: {sorted(INTERVAL_MINUTES)}"
        )

    ordered = sorted(bars, key=lambda b: b.ts)
    source_min = INTERVAL_MINUTES.get(ordered[0].interval)
    if source_min is None:
        raise ValueError(f"unknown source interval {ordered[0].interval!r}")
    if source_min >= target_min:
        raise ValueError(
            f"target {target_interval} ({target_min}m) must be coarser than the "
            f"source {ordered[0].interval} ({source_min}m)"
        )
    if target_min % source_min != 0:
        raise ValueError(
            f"{ordered[0].interval} does not divide {target_interval} evenly"
        )

    delta = timedelta(minutes=target_min)
    buckets: dict[datetime, list[Bar]] = {}
    for bar in ordered:
        open_time = bar.ts - timedelta(minutes=source_min)   # close -> open
        start = _bucket_open(open_time, target_min)
        buckets.setdefault(start, []).append(bar)

    out: list[Bar] = []
    for start in sorted(buckets):
        group = buckets[start]
        volume = sum((b.volume for b in group), ZERO)
        notional = sum((b.vwap * b.volume for b in group if b.vwap is not None), ZERO)
        vol_with_vwap = sum((b.volume for b in group if b.vwap is not None), ZERO)
        trades = sum((b.trades or 0) for b in group) or None
        out.append(Bar(
            instrument_key=group[0].instrument_key,
            ts=start + delta,                       # close-time convention
            interval=target_interval,
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=volume,
            trades=trades,
            vwap=(notional / vol_with_vwap) if vol_with_vwap > 0 else None,
        ))
    return out


__all__ = ["resample_bars"]
