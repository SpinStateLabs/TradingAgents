"""Simulated venue used by both paper trading and the backtester.

Fill realism is the whole point. A paper broker that fills every order at the
mid price instantly and for free will make almost any strategy look profitable,
because the costs it omits -- spread, slippage, fees, queue position, unsettled
cash -- are precisely the costs that decide whether a small account makes money.
This implementation therefore models:

* **Spread**, not mid. Buys cross to the ask, sells to the bid.
* **Size-dependent slippage**, so that a strategy cannot pretend it can move
  size through a thin book at the touch.
* **Real fee schedules**, taken from the instrument definition.
* **Limit orders that rest** and only fill when the market actually trades
  through them -- no optimistic same-bar fills.
* **T+1 cash settlement** when ``enforce_cash_account`` is set, mirroring the
  real IBKR cash account where sale proceeds are unavailable until the next
  session. Ignoring this is how a backtest "earns" returns by round-tripping
  the same dollar five times a day, which the real account cannot do.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable

from spintrader.core.config import Settings
from spintrader.core.types import (
    Balance, Fill, Instrument, Order, OrderStatus, OrderType, Position, Quote,
    Side, TradingMode, VenueId, to_decimal, utcnow,
)
from spintrader.venues.base import (
    AccountSnapshot, InsufficientFunds, OrderRejected, Venue, VenueError,
)

log = logging.getLogger(__name__)

QuoteSource = Callable[[Instrument], Quote]


@dataclass(slots=True)
class SlippageModel:
    """Size- and volatility-aware slippage on top of the spread.

    ``impact_coefficient`` applies the square-root market-impact rule: cost
    scales with the square root of participation rate, which is the standard
    empirical finding and is far kinder than linear at small sizes -- correct
    here, since a $25 order is a rounding error against BTC/USD depth.
    """
    base_bps: Decimal = Decimal("1")           # unconditional cost, in bps
    impact_coefficient: Decimal = Decimal("10")
    reference_size: Decimal = Decimal("100000")   # notional at which impact = coefficient

    def slippage_bps(self, notional: Decimal) -> Decimal:
        if notional <= 0 or self.reference_size <= 0:
            return self.base_bps
        ratio = notional / self.reference_size
        # Decimal has no sqrt on all versions; go via float, which is fine for
        # a cost *model* (it never touches the ledger).
        impact = self.impact_coefficient * to_decimal(float(ratio) ** 0.5)
        return self.base_bps + impact

    def apply(self, price: Decimal, side: Side, notional: Decimal) -> Decimal:
        """Worsen ``price`` against the trader by the modelled slippage."""
        bps = self.slippage_bps(notional)
        adjustment = price * bps / Decimal(10_000)
        return price + adjustment if side is Side.BUY else price - adjustment


@dataclass(slots=True)
class _SettlementBucket:
    """Cash that becomes available at a future time (T+1 settlement)."""
    amount: Decimal
    available_at: datetime


class PaperVenue(Venue):
    """Simulated fills against real quotes.

    ``quote_source`` supplies the market. In paper trading that is a live feed;
    in a backtest it is the replay cursor. Because both go through the same
    object, a backtest exercises the identical order-handling code that paper
    and live trading do.
    """

    venue_id = VenueId.PAPER

    def __init__(
        self,
        quote_source: QuoteSource,
        starting_cash: Decimal = Decimal("100"),
        currency: str = "USD",
        settings: Settings | None = None,
        slippage: SlippageModel | None = None,
        settlement_days: int = 1,
    ) -> None:
        super().__init__(settings=settings)
        self._quote_source = quote_source
        self._currency = currency
        self._cash = to_decimal(starting_cash)
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []
        self._resting: list[Order] = []
        self._pending_settlement: list[_SettlementBucket] = []
        self._slippage = slippage or SlippageModel()
        self._settlement_days = settlement_days
        self._now: datetime = utcnow()
        self._instruments: dict[str, Instrument] = {}

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> None:
        return

    def _disconnect(self) -> None:
        return

    # -- clock (the backtester drives this) --------------------------------

    def set_time(self, ts: datetime) -> None:
        """Advance the simulated clock, releasing settled cash and filling rests."""
        self._now = ts
        self._release_settled_cash()
        self._check_resting_orders()

    @property
    def now(self) -> datetime:
        return self._now

    # -- market data -------------------------------------------------------

    def register(self, instrument: Instrument) -> None:
        self._instruments[instrument.symbol] = instrument

    def resolve(self, symbol: str) -> Instrument:
        try:
            return self._instruments[symbol]
        except KeyError:
            raise VenueError(f"paper venue has no instrument registered for {symbol!r}") from None

    def get_quote(self, instrument: Instrument) -> Quote:
        return self._quote_source(instrument)

    # -- account -----------------------------------------------------------

    def _settled_cash(self) -> Decimal:
        """Cash available right now, excluding unsettled proceeds."""
        return self._cash

    def _unsettled_cash(self) -> Decimal:
        return sum((b.amount for b in self._pending_settlement), Decimal("0"))

    def _release_settled_cash(self) -> None:
        still_pending: list[_SettlementBucket] = []
        for bucket in self._pending_settlement:
            if bucket.available_at <= self._now:
                self._cash += bucket.amount
            else:
                still_pending.append(bucket)
        self._pending_settlement = still_pending

    def snapshot(self) -> AccountSnapshot:
        equity = self._cash + self._unsettled_cash()
        for key, position in self._positions.items():
            if position.is_flat:
                continue
            try:
                quote = self.get_quote(self.resolve(key.split(":", 1)[-1]))
                position.last_price = quote.mid
            except VenueError:
                pass    # keep the last known mark
            equity += position.market_value()

        return AccountSnapshot(
            venue=self.venue_id,
            balances={
                self._currency: Balance(
                    currency=self._currency,
                    total=self._cash + self._unsettled_cash(),
                    available=self._cash,
                    venue=self.venue_id,
                    ts=self._now,
                )
            },
            positions=dict(self._positions),
            equity=equity,
            ts=self._now,
        )

    # -- order handling ----------------------------------------------------

    def _transmit(self, order: Order) -> Order:
        quote = self.get_quote(order.instrument)

        if order.order_type is OrderType.MARKET:
            return self._fill_at_market(order, quote)

        if order.order_type is OrderType.LIMIT:
            # Marketable limits fill immediately; the rest rest. Filling a
            # non-marketable limit in the same bar it was placed is the single
            # most common way a backtest invents returns that do not exist.
            marketable = (
                (order.side is Side.BUY and quote.ask <= order.limit_price)
                or (order.side is Side.SELL and quote.bid >= order.limit_price)
            )
            if marketable:
                return self._fill_at_market(order, quote, cap_price=order.limit_price)
            order.status = OrderStatus.OPEN
            self._resting.append(order)
            return order

        # Stops rest until triggered.
        order.status = OrderStatus.OPEN
        self._resting.append(order)
        return order

    def _fill_at_market(
        self, order: Order, quote: Quote, cap_price: Decimal | None = None
    ) -> Order:
        # Cross the spread, then apply slippage on top.
        touch = quote.ask if order.side is Side.BUY else quote.bid
        notional_estimate = order.qty * touch
        price = self._slippage.apply(touch, order.side, notional_estimate)

        # A limit order can never fill worse than its limit, however bad the
        # modelled slippage -- that is what a limit price means.
        if cap_price is not None:
            price = min(price, cap_price) if order.side is Side.BUY else max(price, cap_price)

        gross = order.qty * price
        fee = gross * order.instrument.taker_fee

        self._assert_affordable(order, gross, fee)

        fill = Fill(
            order_id=order.order_id,
            instrument_key=order.instrument.key,
            side=order.side,
            qty=order.qty,
            price=price,
            ts=self._now,
            fee=fee,
            fee_currency=self._currency,
            liquidity="taker",
            mode=order.mode,
        )
        self._apply_fill(order, fill)
        return order

    def _assert_affordable(self, order: Order, gross: Decimal, fee: Decimal) -> None:
        if order.side is Side.BUY:
            required = gross + fee
            if required > self._cash:
                raise InsufficientFunds(
                    f"buy needs {required:.2f} {self._currency} but only "
                    f"{self._cash:.2f} is settled"
                    + (f" ({self._unsettled_cash():.2f} unsettled)"
                       if self._unsettled_cash() else "")
                )
        else:
            position = self._positions.get(order.instrument.key)
            held = position.qty if position else Decimal("0")
            if held < order.qty and self.settings.enforce_cash_account:
                raise OrderRejected(
                    f"cash account cannot short: selling {order.qty} against a "
                    f"position of {held}"
                )

    def _apply_fill(self, order: Order, fill: Fill) -> None:
        order.apply_fill(fill)
        self._fills.append(fill)

        position = self._positions.setdefault(
            fill.instrument_key, Position(fill.instrument_key)
        )
        position.apply_fill(fill)

        if fill.side is Side.BUY:
            # Purchases settle instantly against settled cash.
            self._cash += fill.cash_delta
        else:
            # Sale proceeds are unavailable until T+N in a cash account.
            proceeds = fill.cash_delta
            if self.settings.enforce_cash_account and self._settlement_days > 0:
                self._pending_settlement.append(
                    _SettlementBucket(
                        amount=proceeds,
                        available_at=self._now + timedelta(days=self._settlement_days),
                    )
                )
            else:
                self._cash += proceeds

        log.debug("paper fill: %s %s @ %s (fee %s)",
                  fill.side.value, fill.qty, fill.price, fill.fee)

    def _check_resting_orders(self) -> None:
        """Fill resting orders whose trigger the market has reached."""
        still_resting: list[Order] = []
        for order in self._resting:
            if order.is_terminal:
                continue
            try:
                quote = self.get_quote(order.instrument)
            except VenueError:
                still_resting.append(order)
                continue

            triggered = False
            if order.order_type is OrderType.LIMIT:
                triggered = (
                    (order.side is Side.BUY and quote.ask <= order.limit_price)
                    or (order.side is Side.SELL and quote.bid >= order.limit_price)
                )
            elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                triggered = (
                    (order.side is Side.BUY and quote.ask >= order.stop_price)
                    or (order.side is Side.SELL and quote.bid <= order.stop_price)
                )

            if not triggered:
                still_resting.append(order)
                continue

            try:
                cap = order.limit_price if order.order_type in (
                    OrderType.LIMIT, OrderType.STOP_LIMIT) else None
                self._fill_at_market(order, quote, cap_price=cap)
            except (InsufficientFunds, OrderRejected) as exc:
                order.status = OrderStatus.REJECTED
                order.reject_reason = str(exc)
                log.info("resting order %s rejected on trigger: %s", order.order_id, exc)

        self._resting = still_resting

    def cancel(self, order: Order) -> Order:
        if order.is_terminal:
            return order
        self._resting = [o for o in self._resting if o.order_id != order.order_id]
        order.status = OrderStatus.CANCELED
        order.updated_at = self._now
        return order

    # -- introspection -----------------------------------------------------

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def realized_pnl(self) -> Decimal:
        return sum((p.realized_pnl for p in self._positions.values()), Decimal("0"))

    @property
    def total_fees(self) -> Decimal:
        return sum((f.fee for f in self._fills), Decimal("0"))


__all__ = ["PaperVenue", "SlippageModel"]
