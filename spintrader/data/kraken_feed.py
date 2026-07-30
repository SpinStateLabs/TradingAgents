"""Kraken OHLCV ingestion.

The subtle correctness issue here is the **partially-formed final bar**.

Kraken's OHLC endpoint returns the bar currently in progress alongside closed
ones. Its ``close`` is the last trade so far, not the bar's actual close. Store
it and two things go wrong: the bar is written with a value that will change,
and -- far worse -- any strategy reading the latest bar sees a "close" for a
period that has not finished, which is lookahead. A backtest built on such data
appears to predict the very move it is reading.

:func:`fetch_ohlc` therefore drops the in-progress bar by default, and
timestamps are converted from Kraken's bar-OPEN convention to the close-time
convention the rest of the system uses.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping

import requests

from spintrader.core.types import Bar, Instrument, to_decimal
from spintrader.data.store import Store
from spintrader.venues.kraken import KrakenVenue

log = logging.getLogger(__name__)

API_BASE = "https://api.kraken.com"

# Kraken's OHLC interval parameter is in minutes. Only these are accepted.
INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
}


class FeedError(RuntimeError):
    """Ingestion failed."""


def interval_delta(interval: str) -> timedelta:
    try:
        return timedelta(minutes=INTERVAL_MINUTES[interval])
    except KeyError:
        raise FeedError(
            f"kraken does not offer a {interval!r} interval; "
            f"available: {sorted(INTERVAL_MINUTES)}"
        ) from None


def fetch_ohlc(
    instrument: Instrument,
    interval: str = "1h",
    since: datetime | None = None,
    session: requests.Session | None = None,
    drop_incomplete: bool = True,
    now: datetime | None = None,
) -> list[Bar]:
    """Fetch bars for ``instrument``.

    ``since`` is inclusive of the bar containing it. Kraken caps the response
    at 720 bars regardless, so deep history needs repeated calls walking
    forward -- see :func:`backfill`.
    """
    minutes = INTERVAL_MINUTES.get(interval)
    if minutes is None:
        raise FeedError(f"unsupported interval {interval!r}")

    params: dict[str, Any] = {"pair": instrument.venue_symbol, "interval": minutes}
    if since is not None:
        params["since"] = int(since.timestamp())

    http = session or requests
    response = http.get(f"{API_BASE}/0/public/OHLC", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    errors = payload.get("error") or []
    if errors:
        raise FeedError(f"kraken OHLC {instrument.symbol}: {', '.join(errors)}")

    result = payload.get("result", {})
    rows: list[list[Any]] = []
    for key, value in result.items():
        if key != "last" and isinstance(value, list):
            rows = value
            break

    delta = timedelta(minutes=minutes)
    reference = now or datetime.now(timezone.utc)
    bars: list[Bar] = []

    for row in rows:
        # [time, open, high, low, close, vwap, volume, count]
        open_time = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
        close_time = open_time + delta

        # The bar in progress has not closed yet. Its 'close' is merely the
        # last trade so far and will change -- storing it is lookahead.
        if drop_incomplete and close_time > reference:
            continue

        bars.append(Bar(
            instrument_key=instrument.key,
            ts=close_time,                 # close-time convention
            interval=interval,
            open=to_decimal(row[1]),
            high=to_decimal(row[2]),
            low=to_decimal(row[3]),
            close=to_decimal(row[4]),
            vwap=to_decimal(row[5]) if row[5] not in (None, "0") else None,
            volume=to_decimal(row[6]),
            trades=int(row[7]) if len(row) > 7 else None,
        ))

    return bars


# Kraken's OHLC endpoint hard-caps the response at 720 of the MOST RECENT
# bars. `since` filters within that window; it cannot reach further back.
# Verified 2026-07-29: requesting 1h bars since 2024-01-01 returned 721 bars
# covering only the last 30 days.
#
# Practical consequence, and the reason this constant is named rather than
# buried: an interval's total available history is 720 * interval. Hourly
# gives 30 days, which is thin for fitting a regime model; daily gives about
# two years. Deeper history needs a different source -- see
# spintrader.data.yfinance_feed.
KRAKEN_MAX_BARS = 720


def available_history(interval: str) -> timedelta:
    """How far back Kraken's OHLC endpoint can reach for this interval."""
    return interval_delta(interval) * KRAKEN_MAX_BARS


def backfill(
    store: Store,
    venue: KrakenVenue,
    symbol: str,
    interval: str = "1h",
    start: datetime | None = None,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Fetch and store the window Kraken makes available, then stop.

    There is deliberately no paging loop. Kraken serves only the most recent
    720 bars regardless of ``since``, so walking backwards cannot work and a
    loop that appears to try is worse than none -- it burns rate limit and
    implies a completeness it never achieves.

    The returned ``reached_api_limit`` says whether the result is bounded by
    Kraken rather than by available market history, so a caller can tell
    "this asset only has 30 days of data" from "this is all the endpoint
    will give".
    """
    instrument = venue.resolve(symbol)
    store.upsert_instrument(instrument)

    http = session or requests.Session()
    since = start or (datetime.now(timezone.utc) - available_history(interval))
    batch = fetch_ohlc(instrument, interval, since=since, session=http)
    written = store.write_bars(batch, source="kraken")

    coverage = store.bar_coverage(instrument.key, interval)
    reached_limit = len(batch) >= KRAKEN_MAX_BARS
    if reached_limit:
        log.info(
            "%s %s: hit Kraken's %d-bar ceiling; history before %s is not "
            "available from this endpoint",
            symbol, interval, KRAKEN_MAX_BARS, coverage["first"],
        )

    return {
        "symbol": symbol,
        "interval": interval,
        "written": written,
        "fetched": len(batch),
        "reached_api_limit": reached_limit,
        **coverage,
    }


__all__ = [
    "FeedError", "INTERVAL_MINUTES", "KRAKEN_MAX_BARS", "available_history",
    "backfill", "fetch_ohlc", "interval_delta",
]
