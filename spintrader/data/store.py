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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from spintrader.core.config import StorageConfig, get_settings
from spintrader.core.types import (
    Bar, Decision, Fill, Instrument, Order, Position, Quote, ensure_utc,
    to_decimal,
)

log = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

ZERO = Decimal("0")


class StoreError(RuntimeError):
    """Database access failed."""


# --------------------------------------------------------------------------
# Read-side value types
# --------------------------------------------------------------------------
#
# Reads return typed rows rather than raw tuples, for the same reason
# ``read_bars`` returns :class:`Bar`s: the column order lives in exactly one
# place, and a caller cannot silently transpose two fields. Monetary columns come
# back as :class:`~decimal.Decimal` (the NUMERIC adapter already does this; the
# constructors re-assert it) and every timestamp is aware UTC.

@dataclass(frozen=True, slots=True)
class EquityRow:
    """One persisted mark of the book, straight from ``equity_curve``.

    Mirrors :class:`~spintrader.portfolio.ledger.EquitySnapshot` in the fields the
    schema keeps; ``positions`` is the decoded JSONB (instrument key -> a small
    ``{qty, avg_cost, last_price}`` dict), enough to rebuild the open book without
    the full ledger.
    """
    ts: datetime
    mode: str
    run_id: str
    equity: Decimal
    cash: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    fees_paid: Decimal
    gross_exposure: Decimal
    positions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One persisted decision, straight from ``decisions``."""
    decision_id: str
    instrument_key: str
    ts: datetime
    action: str
    confidence: Decimal
    target_weight: Decimal | None
    horizon: str
    regime: str | None
    rationale: str
    contributions: dict[str, Any]
    metadata: dict[str, Any]
    mode: str


def _positions_payload(positions: Mapping[str, Position] | None) -> dict[str, Any]:
    """Reduce a ledger position map to the compact JSON the schema stores.

    Only non-flat positions are kept, and only the three fields a reader needs to
    revalue them -- qty, basis, last mark. Decimals are stringified so the JSON
    round-trips without float drift, matching how ``write_decision`` serialises.
    """
    out: dict[str, Any] = {}
    for key, pos in (positions or {}).items():
        if getattr(pos, "is_flat", pos.qty == ZERO):
            continue
        last = pos.last_price
        out[key] = {
            "qty": str(pos.qty),
            "avg_cost": str(pos.avg_cost),
            "last_price": None if last is None else str(last),
        }
    return out


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

    def read_decisions(
        self,
        mode: str,
        instrument_key: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = 100,
    ) -> list[DecisionRow]:
        """Recent decisions for ``mode``, most recent first.

        Returns them descending because "the last thing the loop decided" is what
        a dashboard leads with; a caller that wants chronological order can
        reverse. ``limit`` defaults to 100 so an unbounded scan of a busy minute
        loop is never the accidental default.
        """
        clauses = ["mode = %s"]
        params: list[Any] = [mode]
        if instrument_key is not None:
            clauses.append("instrument_key = %s")
            params.append(instrument_key)
        if start is not None:
            clauses.append("ts >= %s")
            params.append(ensure_utc(start))
        if end is not None:
            clauses.append("ts <= %s")
            params.append(ensure_utc(end))

        sql = f"""
            SELECT decision_id, instrument_key, ts, action, confidence,
                   target_weight, horizon, regime, rationale, contributions,
                   metadata, mode
            FROM decisions WHERE {' AND '.join(clauses)}
            ORDER BY ts DESC
        """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            DecisionRow(
                decision_id=r[0], instrument_key=r[1], ts=ensure_utc(r[2]),
                action=r[3], confidence=to_decimal(r[4]),
                target_weight=None if r[5] is None else to_decimal(r[5]),
                horizon=r[6], regime=r[7], rationale=r[8] or "",
                contributions=dict(r[9] or {}), metadata=dict(r[10] or {}), mode=r[11],
            )
            for r in rows
        ]

    # -- equity curve ------------------------------------------------------

    def write_equity_point(
        self,
        snapshot: Any,
        mode: str,
        run_id: str = "live",
        positions: Mapping[str, Position] | None = None,
    ) -> None:
        """Upsert one equity mark, keyed on (mode, run_id, ts).

        Idempotent like every other write: the loop marks on a fixed cadence and
        restarts replay the same instants, so a re-persisted mark must overwrite
        rather than duplicate -- a doubled row would put a phantom step in the
        curve. The primary-key columns are never in the SET clause, matching the
        store's read-only-key discipline (a mark cannot migrate to another mode or
        instant on update). ``snapshot`` is any
        :class:`~spintrader.portfolio.ledger.EquitySnapshot`-shaped object.
        """
        import json
        row = (
            ensure_utc(snapshot.ts), mode, run_id,
            to_decimal(snapshot.equity), to_decimal(snapshot.cash),
            to_decimal(snapshot.unrealized_pnl), to_decimal(snapshot.realized_pnl),
            to_decimal(snapshot.fees_paid), to_decimal(snapshot.gross_exposure),
            json.dumps(_positions_payload(positions), default=str),
        )
        with self.cursor() as cur:
            cur.execute("""
                INSERT INTO equity_curve (
                    ts, mode, run_id, equity, cash, unrealized_pnl, realized_pnl,
                    fees_paid, gross_exposure, positions
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (mode, run_id, ts) DO UPDATE SET
                    equity         = EXCLUDED.equity,
                    cash           = EXCLUDED.cash,
                    unrealized_pnl = EXCLUDED.unrealized_pnl,
                    realized_pnl   = EXCLUDED.realized_pnl,
                    fees_paid      = EXCLUDED.fees_paid,
                    gross_exposure = EXCLUDED.gross_exposure,
                    positions      = EXCLUDED.positions
            """, row)

    def read_equity_curve(
        self,
        mode: str,
        run_id: str = "live",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[EquityRow]:
        """The equity curve for one (mode, run_id), in chronological order.

        Queried descending so ``limit`` keeps the most *recent* marks, then
        flipped to chronological -- the order a curve is drawn and the order a
        return is compounded, exactly as ``read_bars`` does for bars.
        """
        clauses = ["mode = %s", "run_id = %s"]
        params: list[Any] = [mode, run_id]
        if start is not None:
            clauses.append("ts >= %s")
            params.append(ensure_utc(start))
        if end is not None:
            clauses.append("ts <= %s")
            params.append(ensure_utc(end))

        sql = f"""
            SELECT ts, mode, run_id, equity, cash, unrealized_pnl, realized_pnl,
                   fees_paid, gross_exposure, positions
            FROM equity_curve WHERE {' AND '.join(clauses)}
            ORDER BY ts DESC
        """
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            EquityRow(
                ts=ensure_utc(r[0]), mode=r[1], run_id=r[2],
                equity=to_decimal(r[3]), cash=to_decimal(r[4]),
                unrealized_pnl=to_decimal(r[5]), realized_pnl=to_decimal(r[6]),
                fees_paid=to_decimal(r[7]), gross_exposure=to_decimal(r[8]),
                positions=dict(r[9] or {}),
            )
            for r in reversed(rows)
        ]

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


__all__ = ["DecisionRow", "EquityRow", "Store", "StoreError", "SCHEMA_PATH"]
