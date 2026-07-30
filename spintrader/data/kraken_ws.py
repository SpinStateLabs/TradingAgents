"""Kraken WebSocket v2 collector for live 1-minute bars.

Why this exists
---------------
Kraken's REST OHLC endpoint returns at most 720 bars, which at 1-minute
resolution is **twelve hours** of history. Minute-level strategies need far
more, and the only free way to get it is to accumulate it going forward. Every
day this collector is not running is a day of minute history that cannot be
recovered from REST later.

Correctness rules
-----------------
* **Only closed candles are stored.** Kraken streams the forming candle and
  revises it on every trade. Persisting it would write a "close" for a minute
  that has not ended -- the same lookahead trap the REST feed guards against.
  A candle is treated as final only once a candle with a later
  ``interval_begin`` arrives, which is Kraken telling us the previous one is
  done.
* **Timestamps are converted to close time**, matching every other source.
  Kraken's ``interval_begin`` is the bar's open.
* **Writes are upserts**, so overlapping REST backfill and live collection
  converge rather than duplicate.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Iterable

from spintrader.core.types import Bar, to_decimal, utcnow
from spintrader.data.store import Store

log = logging.getLogger(__name__)

WS_URL = "wss://ws.kraken.com/v2"

# Kraken WS v2 uses 'BTC/USD'; our canonical form is 'BTC-USD'. XBT does not
# appear here -- the v2 API uses modern tickers, unlike the REST asset codes.
def ws_symbol(symbol: str) -> str:
    return symbol.upper().replace("-", "/")


def canonical_symbol(ws: str) -> str:
    return ws.upper().replace("/", "-")


@dataclass(slots=True)
class CollectorStats:
    connects: int = 0
    reconnects: int = 0
    candles_closed: int = 0
    candles_written: int = 0
    messages: int = 0
    errors: int = 0
    last_message_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "connects": self.connects,
            "reconnects": self.reconnects,
            "candles_closed": self.candles_closed,
            "candles_written": self.candles_written,
            "messages": self.messages,
            "errors": self.errors,
            "last_message_at": self.last_message_at,
        }


class KrakenWSCollector:
    """Accumulates closed 1-minute bars from Kraken's OHLC stream."""

    def __init__(
        self,
        symbols: Iterable[str],
        store: Store,
        interval_minutes: int = 1,
        flush_size: int = 1,
        url: str = WS_URL,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.store = store
        self.interval_minutes = interval_minutes
        self.interval = f"{interval_minutes}m"
        self.flush_size = flush_size
        self.url = url
        self.stats = CollectorStats()
        self._running = False
        # symbol -> (interval_begin, candle payload) for the bar still forming.
        self._open_candle: dict[str, tuple[datetime, dict[str, Any]]] = {}
        self._pending: list[Bar] = []

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    async def run(self, max_reconnects: int | None = None) -> None:
        """Connect and collect until stopped.

        Reconnects with exponential backoff. A dropped socket is routine --
        Kraken cycles connections -- so this is a normal path, not an error
        path, and the open candle is discarded on reconnect because we cannot
        know whether we missed trades within it.
        """
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets is not installed") from exc

        self._running = True
        attempt = 0

        while self._running:
            if max_reconnects is not None and self.stats.reconnects > max_reconnects:
                log.info("reached the reconnect limit, stopping")
                break
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, ping_timeout=20, close_timeout=5,
                ) as socket:
                    self.stats.connects += 1
                    attempt = 0
                    await self._subscribe(socket)
                    log.info("kraken ws: subscribed to %d symbols at %s",
                             len(self.symbols), self.interval)
                    await self._consume(socket)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                    # noqa: BLE001 - reconnect
                if not self._running:
                    break
                self.stats.errors += 1
                self.stats.reconnects += 1
                attempt += 1
                delay = min(2 ** attempt, 60)
                log.warning("kraken ws disconnected (%s); reconnecting in %ds",
                            exc, delay)
                # Discard partial state: we cannot know what we missed.
                self._open_candle.clear()
                await asyncio.sleep(delay)

        self._flush()

    async def _subscribe(self, socket: Any) -> None:
        await socket.send(json.dumps({
            "method": "subscribe",
            "params": {
                "channel": "ohlc",
                "symbol": [ws_symbol(s) for s in self.symbols],
                "interval": self.interval_minutes,
                "snapshot": True,
            },
        }))

    async def _consume(self, socket: Any) -> None:
        async for raw in socket:
            if not self._running:
                return
            self.stats.messages += 1
            self.stats.last_message_at = utcnow()
            try:
                self._handle(json.loads(raw))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                self.stats.errors += 1
                log.debug("skipping malformed ws message: %s", exc)

    # -- message handling --------------------------------------------------

    def _handle(self, message: dict[str, Any]) -> None:
        if message.get("channel") != "ohlc":
            # Heartbeats, subscription acks and status messages.
            if message.get("error"):
                self.stats.errors += 1
                log.warning("kraken ws error: %s", message["error"])
            return

        for payload in message.get("data") or []:
            self._handle_candle(payload)

    def _handle_candle(self, payload: dict[str, Any]) -> None:
        ws_sym = payload.get("symbol")
        begin_raw = payload.get("interval_begin")
        if not ws_sym or not begin_raw:
            return

        symbol = canonical_symbol(ws_sym)
        begin = _parse_ts(begin_raw)
        previous = self._open_candle.get(symbol)

        if previous is not None and begin > previous[0]:
            # A newer candle has started, so the previous one is final.
            self._close_candle(symbol, previous[0], previous[1])

        self._open_candle[symbol] = (begin, payload)

    def _close_candle(self, symbol: str, begin: datetime, payload: dict[str, Any]) -> None:
        """Convert a finished candle to a Bar and queue it for storage."""
        instrument_key = f"kraken:{symbol}"
        close_time = begin + timedelta(minutes=self.interval_minutes)
        try:
            bar = Bar(
                instrument_key=instrument_key,
                ts=close_time,                      # close-time convention
                interval=self.interval,
                open=to_decimal(payload["open"]),
                high=to_decimal(payload["high"]),
                low=to_decimal(payload["low"]),
                close=to_decimal(payload["close"]),
                volume=to_decimal(payload.get("volume", 0)),
                trades=int(payload["trades"]) if payload.get("trades") is not None else None,
                vwap=to_decimal(payload["vwap"]) if payload.get("vwap") else None,
            )
        except (KeyError, ValueError) as exc:
            self.stats.errors += 1
            log.warning("discarding malformed candle for %s at %s: %s",
                        symbol, begin, exc)
            return

        self.stats.candles_closed += 1
        self._pending.append(bar)
        if len(self._pending) >= self.flush_size:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        try:
            written = self.store.write_bars(self._pending, source="kraken_ws")
            self.stats.candles_written += written
            log.debug("wrote %d 1m bars", written)
        except Exception as exc:                        # noqa: BLE001 - keep collecting
            self.stats.errors += 1
            # Dropping the buffer beats unbounded growth: the gap is visible in
            # the store, whereas an OOM kills collection entirely.
            log.error("failed to write %d bars, dropping them: %s",
                      len(self._pending), exc)
        finally:
            self._pending.clear()


def _parse_ts(value: str) -> datetime:
    """Parse Kraken's RFC3339 timestamps, which use a trailing Z."""
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


async def collect(
    symbols: Iterable[str],
    store: Store,
    interval_minutes: int = 1,
    max_reconnects: int | None = None,
) -> CollectorStats:
    """Run a collector until interrupted, then report what it did."""
    collector = KrakenWSCollector(symbols, store, interval_minutes=interval_minutes)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, ValueError):
            loop.add_signal_handler(sig, collector.stop)

    await collector.run(max_reconnects=max_reconnects)
    return collector.stats


__all__ = [
    "CollectorStats", "KrakenWSCollector", "WS_URL", "canonical_symbol",
    "collect", "ws_symbol",
]
