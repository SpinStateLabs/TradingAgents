"""TimescaleDB store: schema management and idempotent persistence.

Every write is an upsert keyed on natural identity, because ingestion is
re-run constantly -- on restart, on backfill, after a gap. A store that
duplicates on re-ingest silently doubles volume figures and corrupts every
indicator computed from them.

Reads return the exact columns the quant layer needs and nothing more; wide
``SELECT *`` queries over a hypertable are how a research loop ends up pulling
gigabytes to compute a 20-period average.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from spintrader.core.config import StorageConfig, get_settings
from spintrader.core.types import (
    Bar, Decision, Fill, Instrument, Order, Quote, ensure_utc, to_decimal,
)

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class StoreError(RuntimeError):
    """Database access failed."""


class Store:
    """Connection pool plus typed persistence helpers."""

    def __init__(self, config: StorageConfig | None = None) -> None:
        self.config = config or get_settings().storage
        self._pool: Any | None = None

    # -- connection --------------------------------------------------------

    def connect(self) -> None:
        if self._pool is not None:
            return
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as exc:
            raise StoreError(
                "psycopg is not installed (uv pip install 'psycopg[binary,pool]')"
            ) from exc

        try:
            self._pool = ConnectionPool(
                self.config.dsn,
                min_size=self.config.pool_min,
                max_size=self.config.pool_max,
                open=True,
                timeout=30,
            )
        except Exception as exc:                        # noqa: BLE001 - re-raised
            raise StoreError(
                f"cannot connect to {self.config.redacted_dsn()}: {exc}"
            ) from exc

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        if self._pool is None:
            self.connect()
        with self._pool.connection() as conn:           # type: ignore[union-attr]
            with conn.cursor() as cur:
                yield cur

    def __enter__(self) -> "Store":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- schema ------------------------------------------------------------

    def migrate(self) -> None:
        """Apply the schema. Idempotent -- safe to run on every start."""
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.cursor() as cur:
            cur.execute(sql)
        log.info("schema applied to %s", self.config.redacted_dsn())

    def health(self) -> dict[str, Any]:
        with self.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            cur.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            row = cur.fetchone()
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name
            """)
            tables = [r[0] for r in cur.fetchall()]
        return {
            "postgres": version.split(" on ")[0],
            "timescaledb": row[0] if row else None,
            "tables": tables,
        }

    # -- instruments -------------------------------------------------------

    def upsert_instrument(self, instrument: Instrument) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO instruments (
                    instrument_key, venue, symbol, venue_symbol, asset_class,
                    base_currency, quote_currency, price_increment, qty_increment,
                    min_qty, min_notional, maker_fee, taker_fee, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (instrument_key) DO UPDATE SET
                    venue_symbol    = EXCLUDED.venue_symbol,
                    price_increment = EXCLUDED.price_increment,
                    qty_increment   = EXCLUDED.qty_increment,
                    min_qty         = EXCLUDED.min_qty,
                    min_notional    = EXCLUDED.min_notional,
                    maker_fee       = EXCLUDED.maker_fee,
                    taker_fee       = EXCLUDED.taker_fee,
                    updated_at      = now()
            """, (
                instrument.key, instrument.venue.value, instrument.symbol,
                instrument.venue_symbol, instrument.asset_class.value,
                instrument.base_currency, instrument.quote_currency,
                instrument.price_increment, instrument.qty_increment,
                instrument.min_qty, instrument.min_notional,
                instrument.maker_fee, instrument.taker_fee,
            ))

    def known_instruments(self) -> list[str]:
        with self.cursor() as cur:
            cur.execute("SELECT instrument_key FROM instruments ORDER BY instrument_key")
            return [r[0] for r in cur.fetchall()]

    # -- bars --------------------------------------------------------------

    def write_bars(self, bars: Sequence[Bar], source: str) -> int:
        """Upsert bars. Returns the number of rows written.

        Later data for an existing (instrument, interval, ts) overwrites
        earlier: exchanges revise recent bars, and the newer version is
        authoritative.
        """
        if not bars:
            return 0
        rows = [
            (b.instrument_key, b.interval, b.ts, b.open, b.high, b.low,
             b.close, b.volume, b.trades, b.vwap, source)
            for b in bars
        ]
        with self.cursor() as cur:
            cur.executemany("""
                INSERT INTO bars (instrument_key, interval, ts, open, high, low,
                                  close, volume, trades, vwap, source, ingested_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                ON CONFLICT (instrument_key, interval, ts) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low  = EXCLUDED.low,  close = EXCLUDED.close,
                    volume = EXCLUDED.volume, trades = EXCLUDED.trades,
                    vwap = EXCLUDED.vwap, source = EXCLUDED.source,
                    ingested_at = now()
            """, rows)
        return len(rows)

    def read_bars(
        self,
        instrument_key: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Bar]:
        clauses = ["instrument_key = %s", "interval = %s"]
        params: list[Any] = [instrument_key, interval]
        if start is not None:
            clauses.append("ts >= %s")
            params.append(ensure_utc(start))
        if end is not None:
            clauses.append("ts <= %s")
            params.append(ensure_utc(end))

        sql = f"""
            SELECT instrument_key, ts, interval, open, high, low, close,
                   volume, trades, vwap
            FROM bars WHERE {' AND '.join(clauses)}
            ORDER BY ts DESC
        """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        # Query descending so LIMIT takes the most recent, then flip to
        # chronological, which is what every indicator expects.
        return [
            Bar(instrument_key=r[0], ts=r[1], interval=r[2], open=r[3], high=r[4],
                low=r[5], close=r[6], volume=r[7], trades=r[8], vwap=r[9])
            for r in reversed(rows)
        ]

    def last_bar_ts(self, instrument_key: str, interval: str) -> datetime | None:
        """Latest stored bar close, used to resume incremental sync."""
        with self.cursor() as cur:
            cur.execute(
                "SELECT max(ts) FROM bars WHERE instrument_key = %s AND interval = %s",
                (instrument_key, interval),
            )
            return cur.fetchone()[0]

    def bar_coverage(self, instrument_key: str, interval: str) -> dict[str, Any]:
        with self.cursor() as cur:
            cur.execute("""
                SELECT count(*), min(ts), max(ts)
                FROM bars WHERE instrument_key = %s AND interval = %s
            """, (instrument_key, interval))
            count, first, last = cur.fetchone()
        return {"bars": count, "first": first, "last": last}

    # -- orders, fills, decisions -----------------------------------------

    def write_order(self, order: Order) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO orders (
                    order_id, venue_order_id, client_order_id, decision_id, strategy,
                    instrument_key, side, order_type, qty, limit_price, stop_price,
                    time_in_force, mode, status, filled_qty, avg_fill_price,
                    fees_paid, reject_reason, created_at, updated_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (order_id) DO UPDATE SET
                    venue_order_id = EXCLUDED.venue_order_id,
                    status         = EXCLUDED.status,
                    filled_qty     = EXCLUDED.filled_qty,
                    avg_fill_price = EXCLUDED.avg_fill_price,
                    fees_paid      = EXCLUDED.fees_paid,
                    reject_reason  = EXCLUDED.reject_reason,
                    updated_at     = EXCLUDED.updated_at
            """, (
                order.order_id, order.venue_order_id, order.client_order_id,
                order.decision_id, order.strategy, order.instrument.key,
                order.side.value, order.order_type.value, order.qty,
                order.limit_price, order.stop_price, order.time_in_force.value,
                order.mode.value, order.status.value, order.filled_qty,
                order.avg_fill_price, order.fees_paid, order.reject_reason,
                order.created_at, order.updated_at,
            ))

    def write_fill(self, fill: Fill) -> None:
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO fills (
                    fill_id, order_id, venue_fill_id, instrument_key, side, qty,
                    price, fee, fee_currency, liquidity, mode, ts
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (fill_id) DO NOTHING
            """, (
                fill.fill_id, fill.order_id, fill.venue_fill_id, fill.instrument_key,
                fill.side.value, fill.qty, fill.price, fill.fee, fill.fee_currency,
                fill.liquidity, fill.mode.value, fill.ts,
            ))

    def write_decision(self, decision: Decision, mode: str) -> None:
        import json
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO decisions (
                    decision_id, instrument_key, ts, action, confidence,
                    target_weight, horizon, regime, rationale, contributions,
                    metadata, mode
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (decision_id) DO NOTHING
            """, (
                decision.decision_id, decision.instrument_key, decision.ts,
                decision.action.value, decision.confidence, decision.target_weight,
                decision.horizon, decision.regime, decision.rationale,
                json.dumps(decision.contributions, default=str),
                json.dumps(dict(decision.metadata), default=str), mode,
            ))

    # -- research cache ----------------------------------------------------

    def cache_get(self, cache_key: str) -> dict[str, Any] | None:
        """Return a cached payload if present and unexpired, counting the hit."""
        with self.cursor() as cur:
            cur.execute("""
                UPDATE research_cache SET hit_count = hit_count + 1
                WHERE cache_key = %s
                  AND (expires_at IS NULL OR expires_at > now())
                RETURNING payload
            """, (cache_key,))
            row = cur.fetchone()
        return row[0] if row else None

    def cache_put(
        self,
        cache_key: str,
        source: str,
        payload: dict[str, Any],
        url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        import hashlib
        import json
        blob = json.dumps(payload, sort_keys=True, default=str)
        content_hash = hashlib.sha256(blob.encode()).hexdigest()
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO research_cache (
                    cache_key, source, url, content_hash, payload,
                    fetched_at, expires_at
                ) VALUES (%s,%s,%s,%s,%s, now(),
                          CASE WHEN %s IS NULL THEN NULL
                               ELSE now() + (%s || ' seconds')::interval END)
                ON CONFLICT (cache_key) DO UPDATE SET
                    content_hash = EXCLUDED.content_hash,
                    payload      = EXCLUDED.payload,
                    fetched_at   = now(),
                    expires_at   = EXCLUDED.expires_at
            """, (cache_key, source, url, content_hash, blob,
                  ttl_seconds, ttl_seconds))


__all__ = ["Store", "StoreError", "SCHEMA_PATH"]
