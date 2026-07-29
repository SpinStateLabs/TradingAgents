"""Core domain types shared by every layer of SpinTrader.

Two rules hold everywhere below and are not negotiable:

1. Money and quantities are ``Decimal``. Float arithmetic silently corrupts a
   ledger -- ``0.1 + 0.2 != 0.3`` becomes a reconciliation break against the
   venue three weeks later, and you cannot tell whether the bug is yours or
   theirs. Prices arriving as floats from an API are converted at the boundary
   via :func:`to_decimal`, never mid-calculation.
2. Timestamps are timezone-aware UTC. A naive datetime anywhere in a backtest
   is a lookahead bug waiting for a DST transition to surface it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping


# --------------------------------------------------------------------------
# Conversion helpers
# --------------------------------------------------------------------------

def to_decimal(value: Any) -> Decimal:
    """Convert an API-supplied number to ``Decimal`` without float artifacts.

    Floats are routed through ``repr`` so that ``0.1`` becomes ``Decimal("0.1")``
    rather than the full binary expansion ``0.1000000000000000055511...``.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"cannot convert {value!r} to Decimal") from exc


def utcnow() -> datetime:
    """Timezone-aware current time. Use this, never ``datetime.utcnow()``."""
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalise aware ones to UTC.

    Naive datetimes are rejected rather than assumed-UTC on purpose: silently
    assuming is how a local-time bar timestamp ends up shifted by hours in a
    backtest and produces spurious alpha.
    """
    if dt.tzinfo is None:
        raise ValueError(f"naive datetime not allowed: {dt!r} (attach a tzinfo)")
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    FX = "fx"
    OPTION = "option"
    FUTURE = "future"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        """+1 for buy, -1 for sell. Used to signed-ify quantities."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    DAY = "day"


class OrderStatus(str, Enum):
    # Local-only states, before the venue has seen the order.
    DRAFT = "draft"              # constructed, not yet risk-checked
    REJECTED_BY_RISK = "rejected_by_risk"
    # Venue states.
    PENDING = "pending"          # sent, no ack yet
    OPEN = "open"                # acked, working
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"        # rejected by the venue
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES = frozenset({
    OrderStatus.REJECTED_BY_RISK,
    OrderStatus.FILLED,
    OrderStatus.CANCELED,
    OrderStatus.REJECTED,
    OrderStatus.EXPIRED,
})


class VenueId(str, Enum):
    KRAKEN = "kraken"
    IBKR = "ibkr"
    PAPER = "paper"


class TradingMode(str, Enum):
    """How an order is allowed to be routed.

    ``BACKTEST`` and ``PAPER`` both use simulated fills; they are distinct so
    that logs, ledgers and scorecards never commingle a replayed historical
    run with a forward paper run.
    """
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

    @property
    def is_simulated(self) -> bool:
        return self is not TradingMode.LIVE


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradeable symbol on a specific venue.

    ``symbol`` is SpinTrader's canonical form (``BTC-USD``, ``AAPL``);
    ``venue_symbol`` is whatever that venue insists on (Kraken says ``XBTUSD``).
    Keeping both means the strategy layer never learns venue quirks.
    """
    symbol: str
    asset_class: AssetClass
    venue: VenueId
    venue_symbol: str
    quote_currency: str = "USD"
    base_currency: str | None = None
    # Venue trading rules. Orders are snapped to these before transmission.
    price_increment: Decimal = Decimal("0.01")
    qty_increment: Decimal = Decimal("0.00000001")
    min_qty: Decimal = Decimal("0")
    min_notional: Decimal = Decimal("0")
    supports_fractional: bool = True
    # Venue fee schedule, as a fraction (0.0026 == 26 bps).
    maker_fee: Decimal = Decimal("0")
    taker_fee: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.base_currency is None and "-" in self.symbol:
            object.__setattr__(self, "base_currency", self.symbol.split("-", 1)[0])

    @property
    def key(self) -> str:
        """Globally unique key, e.g. ``kraken:BTC-USD``."""
        return f"{self.venue.value}:{self.symbol}"


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bar:
    """An OHLCV bar.

    ``ts`` is the bar's **close** time, i.e. the first instant at which this bar
    is fully observable. Storing open-time is the single most common source of
    off-by-one-bar lookahead, so the convention is enforced here and asserted
    in the ingestion layer.
    """
    instrument_key: str
    ts: datetime
    interval: str          # "1m", "5m", "1h", "1d"
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades: int | None = None
    vwap: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        if self.high < self.low:
            raise ValueError(f"bar {self.instrument_key}@{self.ts}: high {self.high} < low {self.low}")


@dataclass(frozen=True, slots=True)
class Quote:
    """Top-of-book snapshot."""
    instrument_key: str
    ts: datetime
    bid: Decimal
    ask: Decimal
    bid_size: Decimal = Decimal("0")
    ask_size: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / Decimal(2)

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> Decimal:
        mid = self.mid
        if mid <= 0:
            return Decimal("0")
        return (self.spread / mid) * Decimal(10_000)


# --------------------------------------------------------------------------
# Orders and fills
# --------------------------------------------------------------------------

def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Order:
    """An order request and its lifecycle state.

    Every order carries the ``decision_id`` that produced it. That link is what
    makes post-hoc attribution possible -- without it the self-improvement loop
    cannot tell which agent's reasoning earned or lost the money.
    """
    instrument: Instrument
    side: Side
    qty: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    mode: TradingMode = TradingMode.PAPER

    order_id: str = field(default_factory=_new_id)
    venue_order_id: str | None = None
    client_order_id: str | None = None
    decision_id: str | None = None
    strategy: str | None = None

    status: OrderStatus = OrderStatus.DRAFT
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    fees_paid: Decimal = Decimal("0")
    reject_reason: str | None = None

    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.qty = to_decimal(self.qty)
        if self.qty <= 0:
            raise ValueError(f"order qty must be positive, got {self.qty}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} order requires a limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires a stop_price")
        if self.client_order_id is None:
            self.client_order_id = f"st-{self.order_id[:16]}"

    @property
    def remaining_qty(self) -> Decimal:
        return self.qty - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def notional(self) -> Decimal | None:
        """Best-effort notional for risk checks, using the order's own price.

        Returns ``None`` for market orders, which have no intrinsic price --
        the risk engine must supply a reference quote for those.
        """
        price = self.limit_price or self.stop_price
        return None if price is None else self.qty * to_decimal(price)

    def apply_fill(self, fill: "Fill") -> None:
        """Fold a fill into this order's aggregate state."""
        if fill.order_id != self.order_id:
            raise ValueError(f"fill {fill.fill_id} belongs to order {fill.order_id}, not {self.order_id}")
        new_filled = self.filled_qty + fill.qty
        if new_filled > self.qty:
            raise ValueError(
                f"overfill on {self.order_id}: {new_filled} filled against qty {self.qty}"
            )
        # Volume-weight the average price across fills.
        prior_notional = (self.avg_fill_price or Decimal("0")) * self.filled_qty
        self.avg_fill_price = (prior_notional + fill.price * fill.qty) / new_filled
        self.filled_qty = new_filled
        self.fees_paid += fill.fee
        self.status = OrderStatus.FILLED if new_filled == self.qty else OrderStatus.PARTIALLY_FILLED
        self.updated_at = utcnow()


@dataclass(frozen=True, slots=True)
class Fill:
    """A single execution against an order. Immutable once recorded."""
    order_id: str
    instrument_key: str
    side: Side
    qty: Decimal
    price: Decimal
    ts: datetime
    fee: Decimal = Decimal("0")
    fee_currency: str = "USD"
    fill_id: str = field(default_factory=_new_id)
    venue_fill_id: str | None = None
    liquidity: str | None = None   # "maker" | "taker"
    mode: TradingMode = TradingMode.PAPER

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", ensure_utc(self.ts))
        if self.qty <= 0:
            raise ValueError(f"fill qty must be positive, got {self.qty}")

    @property
    def notional(self) -> Decimal:
        return self.qty * self.price

    @property
    def signed_qty(self) -> Decimal:
        return self.qty * self.side.sign

    @property
    def cash_delta(self) -> Decimal:
        """Change in cash. Buying spends (negative), selling receives; fees always cost."""
        return -(self.signed_qty * self.price) - self.fee


# --------------------------------------------------------------------------
# Positions and balances
# --------------------------------------------------------------------------

@dataclass
class Position:
    """A net position with average-cost basis.

    Average cost (not FIFO lots) is used for P&L attribution because the
    strategies here are net-directional and the tax lot detail is not
    actionable for this account. If lot-level reporting is ever needed, the
    fill ledger retains everything required to reconstruct it.
    """
    instrument_key: str
    qty: Decimal = Decimal("0")          # signed: negative == short
    avg_cost: Decimal = Decimal("0")     # per unit, in quote currency
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    last_price: Decimal | None = None
    updated_at: datetime = field(default_factory=utcnow)

    @property
    def is_flat(self) -> bool:
        return self.qty == 0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def cost_basis(self) -> Decimal:
        return abs(self.qty) * self.avg_cost

    def market_value(self, price: Decimal | None = None) -> Decimal:
        px = price if price is not None else self.last_price
        return Decimal("0") if px is None else self.qty * px

    def unrealized_pnl(self, price: Decimal | None = None) -> Decimal:
        px = price if price is not None else self.last_price
        return Decimal("0") if px is None else (px - self.avg_cost) * self.qty

    def apply_fill(self, fill: Fill) -> Decimal:
        """Apply a fill, returning the realized P&L it generated.

        Handles the four cases explicitly: opening, adding to a position,
        reducing it, and flipping through zero. The flip case is the one that
        naive implementations get wrong -- it must realize the full P&L of the
        closed portion and then re-base the average cost on the residual.
        """
        signed = fill.signed_qty
        realized = Decimal("0")
        self.fees_paid += fill.fee

        if self.qty == 0:
            # Opening a new position.
            self.qty = signed
            self.avg_cost = fill.price
        elif (self.qty > 0) == (signed > 0):
            # Adding in the same direction: weighted-average the cost.
            total = self.qty + signed
            self.avg_cost = (self.avg_cost * self.qty + fill.price * signed) / total
            self.qty = total
        else:
            # Reducing, closing, or flipping.
            closing_qty = min(abs(signed), abs(self.qty))
            # Sign of the P&L follows the direction of the position being closed.
            direction = Decimal(1) if self.qty > 0 else Decimal(-1)
            realized = (fill.price - self.avg_cost) * closing_qty * direction
            self.realized_pnl += realized
            residual = self.qty + signed
            if residual == 0:
                self.qty = Decimal("0")
                self.avg_cost = Decimal("0")
            elif (residual > 0) == (self.qty > 0):
                # Still on the same side, cost basis unchanged.
                self.qty = residual
            else:
                # Flipped through zero: the residual is a fresh position at fill price.
                self.qty = residual
                self.avg_cost = fill.price

        self.last_price = fill.price
        self.updated_at = fill.ts
        return realized


@dataclass(frozen=True, slots=True)
class Balance:
    """A currency balance as reported by a venue."""
    currency: str
    total: Decimal
    available: Decimal
    venue: VenueId
    ts: datetime = field(default_factory=utcnow)

    @property
    def held(self) -> Decimal:
        """Amount encumbered by working orders or unsettled trades."""
        return self.total - self.available


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------

class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Decision:
    """The output of the agent graph: what to do and, critically, why.

    ``rationale`` and ``contributions`` exist so that the self-improvement loop
    can attribute realized P&L back to individual agents and signals. A
    decision without attribution data teaches the system nothing.
    """
    instrument_key: str
    action: Action
    confidence: Decimal              # 0..1
    target_weight: Decimal | None = None   # fraction of portfolio equity
    decision_id: str = field(default_factory=_new_id)
    ts: datetime = field(default_factory=utcnow)
    horizon: str = "1d"
    rationale: str = ""
    regime: str | None = None
    contributions: dict[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = to_decimal(self.confidence)
        if not (Decimal("0") <= self.confidence <= Decimal("1")):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")


__all__ = [
    "Action", "AssetClass", "Balance", "Bar", "Decision", "Fill", "Instrument",
    "Order", "OrderStatus", "OrderType", "Position", "Quote", "Side",
    "TimeInForce", "TradingMode", "VenueId",
    "ensure_utc", "to_decimal", "utcnow",
]
