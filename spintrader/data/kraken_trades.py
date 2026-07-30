"""Deep 1-minute history from Kraken's public ``/Trades`` endpoint.

Why this exists (task 13)
-------------------------
Two existing feeds cannot recover deep minute history, by construction:

* :mod:`spintrader.data.kraken_feed` reads the OHLC endpoint, which serves only
  the most recent 720 bars -- twelve hours at 1-minute resolution -- and cannot
  reach further back however ``since`` is set.
* :mod:`spintrader.data.kraken_ws` streams live candles, so it can only
  accumulate minute bars *going forward* from the moment it starts.

The ``/0/public/Trades`` endpoint can reach back to an asset's first trade. It
returns individual executions in ascending time order, paged forward by a
nanosecond cursor, up to 1000 per call. Aggregating those trades into
1-minute OHLCV bars gives years of minute history the other two feeds cannot.

Correctness rules (identical in spirit to the OHLC and WS feeds)
----------------------------------------------------------------
* **Bar ``ts`` is the CLOSE time.** A trade at 10:03:45 belongs to the minute
  bucket ``[10:03:00, 10:04:00)``, whose close -- the first instant it is fully
  observable -- is 10:04:00. Storing open-time is the single most common source
  of off-by-one-bar lookahead, so it is converted here.
* **The forming minute is never written.** A minute is treated as complete only
  once a trade in a *later* minute has been observed -- the same rule
  :class:`~spintrader.data.kraken_ws.KrakenWSCollector` applies to streamed
  candles. This is what makes paging safe: a 1000-trade page routinely ends
  mid-minute, and finalising that minute before the next page's trades arrive
  would write a bar missing its tail, which the upsert would then make
  permanent. The held bucket therefore survives across pages and merges with
  the next page's trades for the same minute.
* **Decimal from the boundary, UTC everywhere.** Prices and volumes are
  converted to ``Decimal`` as they arrive; trade times are timezone-aware UTC.
* **Writes are upserts**, so overlapping OHLC / WS / trades ingestion converges
  on one series rather than duplicating it.

Cost of running this
--------------------
1000 trades is roughly an hour of BTC/USD history in a quiet period and far
less in a busy one, so backfilling years is tens of thousands of forward calls.
It is a long-running background job, meant for the GB10, and it is resumable:
re-running continues from the last stored 1-minute bar.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Iterable, Sequence

import requests

from spintrader.core.types import Bar, Instrument, ensure_utc, to_decimal
from spintrader.data.kraken_feed import FeedError
from spintrader.data.store import Store
from spintrader.venues.kraken import KrakenVenue

log = logging.getLogger(__name__)

API_BASE = "https://api.kraken.com"

# Kraken returns at most this many trades per call, regardless of the window
# requested. The cursor in the ``last`` field pages forward from there.
TRADES_PER_PAGE = 1000

# Public endpoints share a decaying rate-limit counter. One second between
# calls keeps a multi-hour backfill comfortably under it; the endpoint is not
# the bottleneck anyway -- aggregation is trivial next to the network round
# trip.
DEFAULT_SLEEP_S = 1.0


@dataclass(frozen=True, slots=True)
class TradeTick:
    """A single execution from the Trades endpoint.

    ``ts`` is the execution instant. The row layout Kraken returns is
    ``[price, volume, time, side, order_type, misc, trade_id]``.
    """
    ts: datetime
    price: Decimal
    volume: Decimal
    side: str            # 'b' (buy) | 's' (sell)
    order_type: str      # 'l' (limit) | 'm' (market)
    trade_id: int


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def _since_param(since: datetime | int | str) -> str:
    """Coerce a start value into Kraken's ``since`` form.

    Kraken accepts either a unix-seconds integer or a nanosecond cursor (the
    19-digit ``last`` from a previous page). A ``datetime`` becomes seconds; an
    ``int``/``str`` is passed through, so a cursor round-trips exactly.
    """
    if isinstance(since, datetime):
        return str(int(ensure_utc(since).timestamp()))
    return str(since)


def fetch_trades(
    instrument: Instrument,
    since: datetime | int | str | None = None,
    session: requests.Session | None = None,
    count: int | None = None,
) -> tuple[list[TradeTick], str | None]:
    """Fetch one page of trades for ``instrument``, ascending by time.

    Returns the trades and the ``last`` cursor. Passing that cursor back as
    ``since`` returns the trades *after* the last one here, so paging forward
    never duplicates and never skips.
    """
    params: dict[str, Any] = {"pair": instrument.venue_symbol}
    if since is not None:
        params["since"] = _since_param(since)
    if count is not None:
        params["count"] = count

    http = session or requests
    response = http.get(f"{API_BASE}/0/public/Trades", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    errors = payload.get("error") or []
    if errors:
        # Kraken answers HTTP 200 with an error array, so a naive client reads
        # a failure as an empty success.
        raise FeedError(f"kraken Trades {instrument.symbol}: {', '.join(errors)}")

    result = payload.get("result", {})
    last = result.get("last")
    rows: list[list[Any]] = []
    for key, value in result.items():
        if key != "last" and isinstance(value, list):
            rows = value
            break

    ticks: list[TradeTick] = []
    for row in rows:
        # [price, volume, time, buy/sell, market/limit, misc, trade_id]
        ticks.append(TradeTick(
            ts=datetime.fromtimestamp(float(row[2]), tz=timezone.utc),
            price=to_decimal(row[0]),
            volume=to_decimal(row[1]),
            side=str(row[3]),
            order_type=str(row[4]),
            trade_id=int(row[6]) if len(row) > 6 else 0,
        ))
    return ticks, (str(last) if last is not None else None)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

class MinuteBarAggregator:
    """Buckets ascending trades into OHLCV bars, one minute (or N) at a time.

    The single subtle property: the most recent minute is *held*, never
    emitted, until a trade in a later minute proves it complete. That lets
    trades stream in across many pages -- with any minute split arbitrarily
    across a page boundary -- and still produce exactly one correct bar per
    minute. See the module docstring for why this matters.
    """

    def __init__(self, instrument_key: str, interval_minutes: int = 1) -> None:
        if interval_minutes < 1:
            raise ValueError("interval_minutes must be at least 1")
        self.instrument_key = instrument_key
        self.interval_minutes = interval_minutes
        self.interval = f"{interval_minutes}m"
        self._delta = timedelta(minutes=interval_minutes)
        # State for the bucket currently forming.
        self._start: datetime | None = None
        self._open: Decimal = Decimal("0")
        self._high: Decimal = Decimal("0")
        self._low: Decimal = Decimal("0")
        self._close: Decimal = Decimal("0")
        self._volume: Decimal = Decimal("0")
        self._notional: Decimal = Decimal("0")   # sum(price * volume), for vwap
        self._count: int = 0
        self._last_ts: datetime | None = None

    def bucket_start(self, ts: datetime) -> datetime:
        """Floor ``ts`` to the start of its interval bucket.

        Computed off the wall-clock datetime rather than a float epoch so the
        boundary is exact: a trade at exactly 10:04:00.000 opens the 10:04
        bucket, it does not close the 10:03 one.
        """
        ts = ensure_utc(ts)
        floored = ts.replace(second=0, microsecond=0)
        discard = floored.minute % self.interval_minutes
        return floored - timedelta(minutes=discard)

    def add(self, ticks: Iterable[TradeTick]) -> list[Bar]:
        """Fold trades in and return every bar they *complete*.

        The forming (latest) minute is not returned -- only minutes proven done
        by the arrival of a later trade. Trades must be ascending; a regression
        signals a paging error and is skipped rather than silently corrupting a
        bar that was already emitted.
        """
        completed: list[Bar] = []
        for tick in ticks:
            start = self.bucket_start(tick.ts)

            if self._start is None:
                self._begin(start, tick)
            elif start == self._start:
                self._accumulate(tick)
            elif start > self._start:
                completed.append(self._finalise())
                self._begin(start, tick)
            else:
                # start < current bucket: out-of-order trade. The endpoint
                # returns ascending trades, so this only happens on a paging
                # bug; dropping it is safer than mutating an already-emitted bar.
                log.warning(
                    "%s: out-of-order trade at %s (< current bucket %s); skipping",
                    self.instrument_key, tick.ts, self._start,
                )
        return completed

    def flush(self) -> Bar | None:
        """Finalise and return the held bucket, if any, then clear it.

        The caller takes responsibility for completeness: in a forward backfill
        the held minute is the one still forming, so ``backfill_1m`` does *not*
        call this -- it drops the partial minute and re-acquires it on the next
        run. It exists for tests and for a caller that can prove, from an
        external bound, that the last minute is closed.
        """
        if self._start is None:
            return None
        bar = self._finalise()
        return bar

    @property
    def has_open_bucket(self) -> bool:
        return self._start is not None

    @property
    def open_bucket_start(self) -> datetime | None:
        return self._start

    # -- internals ---------------------------------------------------------

    def _begin(self, start: datetime, tick: TradeTick) -> None:
        self._start = start
        self._open = tick.price
        self._high = tick.price
        self._low = tick.price
        self._close = tick.price
        self._volume = tick.volume
        self._notional = tick.price * tick.volume
        self._count = 1
        self._last_ts = tick.ts

    def _accumulate(self, tick: TradeTick) -> None:
        if tick.price > self._high:
            self._high = tick.price
        if tick.price < self._low:
            self._low = tick.price
        self._close = tick.price
        self._volume += tick.volume
        self._notional += tick.price * tick.volume
        self._count += 1
        self._last_ts = tick.ts

    def _finalise(self) -> Bar:
        assert self._start is not None
        vwap = (self._notional / self._volume) if self._volume > 0 else None
        bar = Bar(
            instrument_key=self.instrument_key,
            ts=self._start + self._delta,        # close-time convention
            interval=self.interval,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            trades=self._count,
            vwap=vwap,
        )
        self._start = None
        return bar


# --------------------------------------------------------------------------
# Backfill
# --------------------------------------------------------------------------

# A fetcher matches fetch_trades' signature, so tests can inject a fake page
# source without a live endpoint or a mock HTTP session.
Fetcher = Callable[..., tuple[list["TradeTick"], "str | None"]]


def _resolve_start(
    store: Store,
    instrument_key: str,
    interval: str,
    start: datetime | int | str | None,
    resume: bool,
) -> datetime | int | str:
    """Decide where to begin paging.

    Priority: an explicit ``start``; then, if resuming, the close of the last
    stored bar (which is the open of the next minute, so nothing is lost and
    the previously-dropped forming minute is re-acquired now that it is
    complete); then the unix epoch, i.e. the whole available history.
    """
    if start is not None:
        return start
    if resume:
        last_ts = store.last_bar_ts(instrument_key, interval)
        if last_ts is not None:
            return ensure_utc(last_ts)
    return 0


def backfill_1m(
    store: Store,
    venue: KrakenVenue,
    symbol: str,
    start: datetime | int | str | None = None,
    end: datetime | None = None,
    interval_minutes: int = 1,
    session: requests.Session | None = None,
    fetcher: Fetcher | None = None,
    max_pages: int | None = None,
    sleep_s: float = DEFAULT_SLEEP_S,
    resume: bool = True,
    now: datetime | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Page the Trades endpoint forward and store aggregated 1-minute bars.

    Walks from ``start`` (or the resume point, or genesis) toward the present,
    aggregating trades into complete minute bars and upserting them as it goes.
    Stops at ``end`` if given, at ``max_pages`` if given, when the cursor stops
    advancing, or when it catches up to the live edge -- whichever comes first.

    The forming final minute is deliberately never written; it is re-acquired
    on the next run. Returns a summary including whether it reached the live
    edge, so a caller can tell "this is all of history" from "stopped early".
    """
    instrument = venue.resolve(symbol)
    store.upsert_instrument(instrument)

    http = session or requests.Session()
    fetch = fetcher or (lambda inst, since: fetch_trades(inst, since=since, session=http))

    interval = f"{interval_minutes}m"
    since = _resolve_start(store, instrument.key, interval, start, resume)
    end_ts = ensure_utc(end) if end is not None else None
    reference = ensure_utc(now) if now is not None else datetime.now(timezone.utc)
    live_edge = reference - timedelta(minutes=interval_minutes)

    agg = MinuteBarAggregator(instrument.key, interval_minutes)
    pages = 0
    written = 0
    trades_seen = 0
    reached_live = False
    hit_end = False
    prev_cursor: str | None = None

    while True:
        if max_pages is not None and pages >= max_pages:
            break

        ticks, cursor = fetch(instrument, since)
        pages += 1

        if not ticks:
            # No trades at or after the cursor: we are at the live edge.
            reached_live = True
            break

        trades_seen += len(ticks)

        # Respect an explicit end bound: a trade at or beyond ``end`` also acts
        # as the "later minute" that finalises the last in-range minute, so keep
        # such a trade for the aggregator's transition logic but never store a
        # bar whose close exceeds the bound.
        newest = ticks[-1].ts
        bars = agg.add(ticks)
        if end_ts is not None:
            keep, dropped = [], False
            for bar in bars:
                if bar.ts <= end_ts:
                    keep.append(bar)
                else:
                    dropped = True
            bars = keep
            if newest >= end_ts or dropped:
                hit_end = True

        written += store.write_bars(bars, source="kraken_trades")

        if progress is not None:
            progress({
                "symbol": symbol, "pages": pages, "written": written,
                "trades_seen": trades_seen, "cursor_ts": newest, "bars_in_page": len(bars),
            })

        if hit_end:
            break
        if cursor is None or cursor == prev_cursor:
            # The cursor did not advance: there is nothing newer to fetch.
            reached_live = True
            break
        if newest >= live_edge:
            # Caught up to the current minute; the WS collector owns the edge
            # from here, so a historical backfill stops rather than polling.
            reached_live = True
            break

        prev_cursor = cursor
        since = cursor
        if sleep_s:
            time.sleep(sleep_s)

    coverage = store.bar_coverage(instrument.key, interval)
    return {
        "symbol": symbol,
        "interval": interval,
        "pages": pages,
        "trades_seen": trades_seen,
        "written": written,
        "reached_live_edge": reached_live,
        "hit_end": hit_end,
        **coverage,
    }


__all__ = [
    "API_BASE", "DEFAULT_SLEEP_S", "Fetcher", "MinuteBarAggregator",
    "TRADES_PER_PAGE", "TradeTick", "backfill_1m", "fetch_trades",
]
