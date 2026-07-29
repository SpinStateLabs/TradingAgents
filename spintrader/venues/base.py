"""The venue abstraction: one interface over Kraken, IBKR and simulated fills.

Design rules
------------
1. **The strategy layer never learns venue quirks.** Symbol mapping, tick and
   lot rounding, fee schedules and settlement rules live behind this interface.
   A strategy asks to buy 0.01 BTC; whether that becomes ``XXBTZUSD`` on Kraken
   or a fractional ``BTC`` contract at IBKR is not its problem.

2. **Every order passes through one submission path.** :meth:`Venue.submit`
   is concrete and final on the base class: it validates, snaps to venue
   increments, checks the live gate, then delegates the actual transmission to
   ``_transmit``. Subclasses cannot accidentally bypass the gate, because they
   never implement the public entry point.

3. **Simulated and live share this code.** The backtester drives the same
   :class:`PaperVenue` the paper trader does, through the same interface the
   live venues implement. A backtest that exercises different code from live
   trading is measuring the wrong program.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Iterable, Mapping, Sequence

from spintrader.core.config import LiveGate, Settings, get_settings
from spintrader.core.types import (
    Balance, Fill, Instrument, Order, OrderStatus, OrderType, Position, Quote,
    Side, TradingMode, VenueId, to_decimal, utcnow,
)

log = logging.getLogger(__name__)


class VenueError(RuntimeError):
    """Base for venue failures."""


class OrderRejected(VenueError):
    """The venue (or our pre-flight validation) refused the order."""


class InsufficientFunds(OrderRejected):
    """Not enough settled cash or position to support the order."""


class NotConnected(VenueError):
    """An operation was attempted before ``connect()`` succeeded."""


# --------------------------------------------------------------------------
# Rounding helpers
# --------------------------------------------------------------------------

def snap_to_increment(
    value: Decimal, increment: Decimal, rounding: str = ROUND_DOWN
) -> Decimal:
    """Round ``value`` to a multiple of ``increment``.

    Quantities round **down** by default: rounding a size up can overspend
    available cash, whereas rounding down merely trades slightly less. Prices
    use ROUND_HALF_UP at the call site, where being off by one tick is
    immaterial but being unrepresentable is a rejection.
    """
    if increment <= 0:
        return value
    return (value / increment).quantize(Decimal("1"), rounding=rounding) * increment


# --------------------------------------------------------------------------
# Account snapshot
# --------------------------------------------------------------------------

@dataclass(slots=True)
class AccountSnapshot:
    """Point-in-time view of a venue account, used for reconciliation."""
    venue: VenueId
    balances: dict[str, Balance] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    equity: Decimal = Decimal("0")
    ts: object = field(default_factory=utcnow)

    def cash(self, currency: str = "USD") -> Decimal:
        bal = self.balances.get(currency)
        return bal.total if bal else Decimal("0")

    def available_cash(self, currency: str = "USD") -> Decimal:
        """Cash free to spend now.

        Distinct from :meth:`cash` because of unsettled proceeds: in a cash
        account, selling today does not free the money until T+1. Sizing
        against total rather than available is how a cash account walks into a
        good-faith violation.
        """
        bal = self.balances.get(currency)
        return bal.available if bal else Decimal("0")


# --------------------------------------------------------------------------
# Venue interface
# --------------------------------------------------------------------------

class Venue(ABC):
    """Common interface to a trading venue.

    Subclasses implement the ``_``-prefixed hooks. The public methods are
    concrete so that validation and the live gate cannot be bypassed.
    """

    venue_id: VenueId

    def __init__(
        self,
        settings: Settings | None = None,
        live_gate: LiveGate | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.live_gate = live_gate or self.settings.live
        self._connected = False
        self._orders: dict[str, Order] = {}

    # -- lifecycle ---------------------------------------------------------

    @abstractmethod
    def _connect(self) -> None:
        """Establish the venue session. Raise on failure."""

    @abstractmethod
    def _disconnect(self) -> None:
        ...

    def connect(self) -> None:
        if self._connected:
            return
        self._connect()
        self._connected = True
        log.info("connected to %s (mode=%s)", self.venue_id.value, self.settings.mode.value)

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._disconnect()
        self._connected = False

    def _require_connection(self) -> None:
        if not self._connected:
            raise NotConnected(f"{self.venue_id.value}: call connect() first")

    def __enter__(self) -> "Venue":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.disconnect()

    # -- market data -------------------------------------------------------

    @abstractmethod
    def get_quote(self, instrument: Instrument) -> Quote:
        """Current top of book."""

    @abstractmethod
    def resolve(self, symbol: str) -> Instrument:
        """Map a canonical symbol to this venue's instrument definition.

        Raises :class:`VenueError` for symbols this venue does not list, which
        is how the router discovers that e.g. BNB/CAD does not exist.
        """

    # -- account -----------------------------------------------------------

    @abstractmethod
    def snapshot(self) -> AccountSnapshot:
        """Balances, positions and equity as the venue reports them."""

    # -- orders ------------------------------------------------------------

    @abstractmethod
    def _transmit(self, order: Order) -> Order:
        """Send a validated order. Implementations must not re-check the gate."""

    @abstractmethod
    def cancel(self, order: Order) -> Order:
        ...

    def submit(self, order: Order) -> Order:
        """Validate, snap, gate-check and transmit an order.

        This is the single path to the market. It is intentionally not
        overridable in spirit: subclasses implement ``_transmit`` instead, so
        no venue can be written that skips the live gate.
        """
        self._require_connection()

        if order.instrument.venue is not self.venue_id:
            raise OrderRejected(
                f"order for {order.instrument.venue.value} submitted to {self.venue_id.value}"
            )
        if order.status is not OrderStatus.DRAFT:
            raise OrderRejected(
                f"order {order.order_id} already has status {order.status.value}; "
                f"submit accepts DRAFT orders only"
            )

        order.mode = self.settings.mode
        self._normalise(order)
        self._check_tradability(order)

        # Live gate last, on the final normalised size, so the cap applies to
        # what actually goes out rather than to what was requested.
        notional = self._reference_notional(order)
        self.live_gate.check(self.venue_id, order.mode, notional)

        if self.settings.dry_run:
            order.status = OrderStatus.REJECTED_BY_RISK
            order.reject_reason = "dry_run enabled; order not transmitted"
            log.info("DRY RUN: would have sent %s", self.describe_order(order))
            return order

        order.status = OrderStatus.PENDING
        transmitted = self._transmit(order)
        self._orders[transmitted.order_id] = transmitted
        return transmitted

    # -- validation helpers ------------------------------------------------

    def _normalise(self, order: Order) -> None:
        """Snap quantity and price to the venue's increments, in place."""
        inst = order.instrument

        qty = snap_to_increment(order.qty, inst.qty_increment, ROUND_DOWN)
        if qty <= 0:
            raise OrderRejected(
                f"quantity {order.qty} rounds to zero at increment {inst.qty_increment}"
            )
        order.qty = qty

        for attr in ("limit_price", "stop_price"):
            price = getattr(order, attr)
            if price is not None:
                setattr(order, attr,
                        snap_to_increment(to_decimal(price), inst.price_increment, ROUND_HALF_UP))

    def _check_tradability(self, order: Order) -> None:
        inst = order.instrument
        if order.qty < inst.min_qty:
            raise OrderRejected(
                f"quantity {order.qty} below {inst.symbol} minimum {inst.min_qty}"
            )
        # No separate fractional check: ``qty_increment`` is the venue's actual
        # rule and _normalise has already snapped to it, so an instrument that
        # trades in whole units (increment 1) cannot reach here with a
        # fractional quantity. A second check on the same condition would be
        # dead code, and rounding down is the consistent behaviour anyway --
        # 1.5 shares of a non-fractional symbol means "buy 1", exactly as
        # 0.123456 BTC means "buy 0.1234".
        if not inst.supports_fractional and inst.qty_increment < 1:
            raise OrderRejected(
                f"{inst.symbol} is marked non-fractional but has a sub-unit "
                f"qty_increment of {inst.qty_increment}; the instrument definition "
                f"is inconsistent"
            )

        notional = self._reference_notional(order)
        if notional is not None and inst.min_notional > 0 and notional < inst.min_notional:
            raise OrderRejected(
                f"notional {notional} below {inst.symbol} minimum {inst.min_notional}"
            )

        if order.side is Side.SELL and not self.settings.risk.allow_shorts:
            # Only a problem if this would *open* a short; reducing a long is
            # always allowed. The portfolio layer supplies the current position.
            position = self.snapshot().positions.get(inst.key)
            held = position.qty if position else Decimal("0")
            if held < order.qty:
                raise OrderRejected(
                    f"selling {order.qty} against a position of {held} would open a short, "
                    f"which the {self.settings.aggression.value} profile forbids"
                )

    def _reference_notional(self, order: Order) -> Decimal | None:
        """Notional for cap checks, falling back to a live quote for market orders."""
        explicit = order.notional
        if explicit is not None:
            return explicit
        try:
            quote = self.get_quote(order.instrument)
        except VenueError:
            # Without a price we cannot bound the order; refuse rather than
            # let a market order slip past the notional cap unmeasured.
            return None
        reference = quote.ask if order.side is Side.BUY else quote.bid
        return order.qty * reference

    # -- reporting ---------------------------------------------------------

    def describe_order(self, order: Order) -> str:
        bits = [
            order.side.value.upper(), str(order.qty), order.instrument.symbol,
            order.order_type.value,
        ]
        if order.limit_price is not None:
            bits.append(f"@ {order.limit_price}")
        bits.append(f"[{self.venue_id.value}/{order.mode.value}]")
        return " ".join(bits)

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def open_orders(self) -> list[Order]:
        return [o for o in self._orders.values() if not o.is_terminal]

    # -- reconciliation ----------------------------------------------------

    def reconcile(self, expected: Mapping[str, Position], tolerance: Decimal = Decimal("0")) -> list[str]:
        """Compare our ledger against the venue's own view.

        Returns a list of human-readable divergences. An empty list means the
        books agree. Silent divergence is the failure mode that matters here:
        trading against a position you do not actually hold produces losses
        that look like strategy failure.
        """
        actual = self.snapshot().positions
        divergences: list[str] = []

        for key in sorted(set(expected) | set(actual)):
            ours = expected[key].qty if key in expected else Decimal("0")
            theirs = actual[key].qty if key in actual else Decimal("0")
            if abs(ours - theirs) > tolerance:
                divergences.append(
                    f"{key}: ledger says {ours}, {self.venue_id.value} says {theirs} "
                    f"(diff {ours - theirs})"
                )
        return divergences


__all__ = [
    "AccountSnapshot", "InsufficientFunds", "NotConnected", "OrderRejected",
    "Venue", "VenueError", "snap_to_increment",
]
