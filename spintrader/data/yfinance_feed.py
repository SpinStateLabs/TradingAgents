"""Deep daily history via yfinance, for both crypto and equities.

Why this exists alongside the Kraken feed
-----------------------------------------
Kraken serves only the most recent 720 bars, which is 30 days of hourly data --
far too little to fit a regime model whose whole purpose is to distinguish
market states that persist for months. yfinance provides years of daily
history for free, for crypto and equities alike.

The two sources have different roles and must not be conflated:

* **yfinance** -- deep daily history, for model fitting and backtesting.
  Unofficial, occasionally revises or gaps, and its prices are consolidated
  rather than venue-specific.
* **Kraken / IBKR** -- recent and intraday data, and the prices we will
  actually trade against.

Bars are tagged with their source in the store precisely so a model fitted on
yfinance daily data is never silently mixed with Kraken hourly bars at a
different price basis.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from spintrader.core.types import Bar, Instrument, to_decimal
from spintrader.data.store import Store

log = logging.getLogger(__name__)

# Canonical symbol -> yfinance ticker. Crypto needs the -USD suffix form,
# which happens to match ours; equities are bare.
_YF_OVERRIDES = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
    "BNB-USD": "BNB-USD",
}

_YF_INTERVALS = {"1d": "1d", "1h": "1h", "1wk": "1wk"}


class YFinanceError(RuntimeError):
    """yfinance fetch failed."""


def yf_ticker(symbol: str) -> str:
    return _YF_OVERRIDES.get(symbol.upper(), symbol.upper())


def fetch_daily(
    instrument: Instrument,
    start: datetime | None = None,
    end: datetime | None = None,
    interval: str = "1d",
    now: datetime | None = None,
) -> list[Bar]:
    """Fetch daily bars.

    yfinance labels daily rows by session DATE, not by close instant. They are
    converted to a close timestamp so they share the store's convention with
    every other source; without that, a daily bar would sort before intraday
    bars from the same session and read as though it were known earlier.
    """
    try:
        import yfinance
    except ImportError as exc:
        raise YFinanceError("yfinance is not installed") from exc

    if interval not in _YF_INTERVALS:
        raise YFinanceError(f"unsupported interval {interval!r}")

    ticker = yf_ticker(instrument.symbol)
    reference = now or datetime.now(timezone.utc)
    start = start or (reference - timedelta(days=365 * 5))
    end = end or reference

    frame = yfinance.download(
        ticker, start=start.date(), end=(end + timedelta(days=1)).date(),
        interval=_YF_INTERVALS[interval], progress=False, auto_adjust=False,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise YFinanceError(f"yfinance returned no data for {ticker}")

    bars: list[Bar] = []
    for index, row in frame.iterrows():
        ts = index.to_pydatetime()
        ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
        if interval == "1d":
            # Session date -> close instant. 21:00 UTC approximates the US
            # equity close; crypto trades continuously so any consistent
            # convention works, provided it is the same one everywhere.
            ts = ts.replace(hour=21, minute=0, second=0, microsecond=0)

        # Skip the session in progress for the same reason the Kraken feed
        # drops its forming bar: its close is not yet the close.
        if ts > reference:
            continue

        try:
            bar = Bar(
                instrument_key=instrument.key,
                ts=ts,
                interval=interval,
                open=to_decimal(float(row["Open"])),
                high=to_decimal(float(row["High"])),
                low=to_decimal(float(row["Low"])),
                close=to_decimal(float(row["Close"])),
                volume=to_decimal(float(row["Volume"])),
            )
        except (ValueError, KeyError, TypeError) as exc:
            # yfinance emits NaN rows around holidays and delistings.
            log.debug("skipping malformed yfinance row at %s: %s", ts, exc)
            continue
        bars.append(bar)

    return bars


def backfill(
    store: Store,
    instrument: Instrument,
    years: int = 5,
    interval: str = "1d",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fetch and store deep daily history."""
    store.upsert_instrument(instrument)
    reference = now or datetime.now(timezone.utc)
    bars = fetch_daily(
        instrument, start=reference - timedelta(days=365 * years),
        interval=interval, now=reference,
    )
    written = store.write_bars(bars, source="yfinance")
    coverage = store.bar_coverage(instrument.key, interval)
    return {
        "symbol": instrument.symbol,
        "interval": interval,
        "written": written,
        "fetched": len(bars),
        **coverage,
    }


__all__ = ["YFinanceError", "backfill", "fetch_daily", "yf_ticker"]
