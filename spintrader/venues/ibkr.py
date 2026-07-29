"""IBKR venue client, via ib_async against IB Gateway.

Two IBKR-specific realities shape this module:

**Delayed data is the default here.** The account has no market-data
subscriptions (a deliberate choice -- they cost more per month than the
account's expected returns), so ``reqMarketDataType(3)`` requests delayed
quotes. Delayed data is fine for daily-horizon decisions and useless for
anything intraday, and the distinction must be visible rather than implicit:
:class:`IBKRVenue` marks every quote it produces with its data type so the
strategy layer can refuse to act on stale prices.

**It is a cash account.** No margin, no shorting, and sale proceeds are
unavailable until T+1. IBKR reports buying power that assumes otherwise when
the paper account is a margin account, so the numbers it returns cannot be
trusted as constraints -- ``AccountSnapshot.available_cash`` reports settled
funds and the risk layer sizes against that.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from spintrader.core.config import Settings, env_int, env_str
from spintrader.core.types import (
    AssetClass, Balance, Instrument, Order, OrderStatus, OrderType, Position,
    Quote, Side, TimeInForce, TradingMode, VenueId, to_decimal, utcnow,
)
from spintrader.venues.base import (
    AccountSnapshot, OrderRejected, Venue, VenueError,
)

log = logging.getLogger(__name__)

# IBKR market data types. 1 = live (needs a subscription), 2 = frozen,
# 3 = delayed, 4 = delayed-frozen.
MARKET_DATA_LIVE, MARKET_DATA_FROZEN, MARKET_DATA_DELAYED = 1, 2, 3

# Paper account identifiers start with D; live with U.
PAPER_PREFIXES = ("DU", "DF", "DUQ")


class IBKRVenue(Venue):
    """US equities and ETFs through IB Gateway."""

    venue_id = VenueId.IBKR

    def __init__(
        self,
        settings: Settings | None = None,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        market_data_type: int = MARKET_DATA_DELAYED,
        ib: Any | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._host = host or env_str("IBKR_HOST", "127.0.0.1")
        self._port = port or env_int("IBKR_PORT", 4002)
        self._client_id = client_id or env_int("IBKR_CLIENT_ID", 17)
        self._market_data_type = market_data_type
        self._ib = ib
        self._instruments: dict[str, Instrument] = {}
        self._account: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> None:
        if self._ib is None:
            try:
                from ib_async import IB
            except ImportError as exc:
                raise VenueError(
                    "ib_async is not installed (uv pip install ib_async)"
                ) from exc
            self._ib = IB()

        try:
            self._ib.connect(
                self._host, self._port, clientId=self._client_id,
                timeout=30, readonly=False,
            )
        except Exception as exc:                        # noqa: BLE001 - re-raised
            raise VenueError(
                f"cannot reach IB Gateway at {self._host}:{self._port} -- {exc}. "
                f"Is the container running? docker ps | grep spintrader-ibgw"
            ) from exc

        self._ib.reqMarketDataType(self._market_data_type)

        accounts = self._ib.managedAccounts()
        if not accounts:
            raise VenueError("IB Gateway returned no managed accounts")
        wanted = env_str("IBKR_ACCOUNT", "")
        self._account = wanted or accounts[0]
        if wanted and wanted not in accounts:
            raise VenueError(
                f"IBKR_ACCOUNT={wanted!r} is not among the available accounts {accounts}"
            )

        # A live account reached through a paper-mode config is the one
        # misconfiguration that costs real money, so it is fatal, not a warning.
        is_paper = self._account.startswith(PAPER_PREFIXES)
        if not is_paper and self.settings.mode is not TradingMode.LIVE:
            raise VenueError(
                f"connected to LIVE account {self._account} while running in "
                f"{self.settings.mode.value} mode; refusing to continue"
            )
        log.info("ibkr: connected to %s (%s), data type %d",
                 self._account, "paper" if is_paper else "LIVE", self._market_data_type)

    def _disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    # -- instruments -------------------------------------------------------

    def resolve(self, symbol: str) -> Instrument:
        symbol = symbol.upper()
        if symbol in self._instruments:
            return self._instruments[symbol]

        from ib_async import Stock

        contract = Stock(symbol, "SMART", self.settings.base_currency)
        details = self._ib.reqContractDetails(contract)
        if not details:
            raise VenueError(f"ibkr does not recognise {symbol!r}")
        detail = details[0]

        tick = to_decimal(detail.minTick or "0.01")
        instrument = Instrument(
            symbol=symbol,
            asset_class=AssetClass.ETF if detail.stockType == "ETF" else AssetClass.EQUITY,
            venue=self.venue_id,
            venue_symbol=symbol,
            quote_currency=detail.contract.currency,
            price_increment=tick,
            # IBKR fractional shares go to 4 decimals, minimum USD 1.00 notional.
            qty_increment=Decimal("0.0001"),
            min_qty=Decimal("0.0001"),
            min_notional=Decimal("1"),
            supports_fractional=True,
            # Tiered/fixed US equity commission is ~0.35 USD minimum per order,
            # which on a $10 order is 3.5% -- modelled at the portfolio layer
            # rather than as a percentage rate here.
            taker_fee=Decimal("0"),
            maker_fee=Decimal("0"),
        )
        instrument = _attach_contract(instrument, detail.contract)
        self._instruments[symbol] = instrument
        return instrument

    # -- market data -------------------------------------------------------

    def get_quote(self, instrument: Instrument) -> Quote:
        contract = _contract_of(instrument)
        tickers = self._ib.reqTickers(contract)
        if not tickers:
            raise VenueError(f"ibkr returned no ticker for {instrument.symbol}")
        ticker = tickers[0]

        bid, ask = ticker.bid, ticker.ask
        # Outside regular hours IBKR returns -1 or nan for the touch; fall back
        # to the last trade so the caller gets a usable mark rather than a
        # nonsense negative spread.
        if not _valid(bid) or not _valid(ask):
            last = ticker.last if _valid(ticker.last) else ticker.close
            if not _valid(last):
                raise VenueError(
                    f"ibkr has no usable price for {instrument.symbol} "
                    f"(market closed and no last trade)"
                )
            bid = ask = last

        return Quote(
            instrument_key=instrument.key,
            ts=utcnow(),
            bid=to_decimal(bid),
            ask=to_decimal(ask),
            bid_size=to_decimal(ticker.bidSize or 0),
            ask_size=to_decimal(ticker.askSize or 0),
        )

    # -- account -----------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        self._require_connection()
        base = self.settings.base_currency

        balances: dict[str, Balance] = {}
        equity = Decimal("0")
        settled = Decimal("0")

        for row in self._ib.accountSummary(self._account):
            if row.tag == "NetLiquidation" and row.currency:
                equity = to_decimal(row.value)
            elif row.tag == "TotalCashValue" and row.currency:
                total = to_decimal(row.value)
                balances[row.currency] = Balance(
                    currency=row.currency, total=total, available=total,
                    venue=self.venue_id, ts=utcnow(),
                )
            elif row.tag == "SettledCash" and row.currency:
                settled = to_decimal(row.value)

        # SettledCash is the real constraint in a cash account: TotalCashValue
        # includes unsettled sale proceeds that cannot be spent yet.
        for currency, balance in list(balances.items()):
            if settled and currency == base:
                balances[currency] = Balance(
                    currency=currency, total=balance.total, available=settled,
                    venue=self.venue_id, ts=balance.ts,
                )

        positions: dict[str, Position] = {}
        for item in self._ib.positions(self._account):
            symbol = item.contract.symbol
            key = f"{self.venue_id.value}:{symbol}"
            positions[key] = Position(
                instrument_key=key,
                qty=to_decimal(item.position),
                avg_cost=to_decimal(item.avgCost),
            )

        return AccountSnapshot(
            venue=self.venue_id, balances=balances, positions=positions,
            equity=equity, ts=utcnow(),
        )

    # -- orders ------------------------------------------------------------

    def _transmit(self, order: Order) -> Order:
        from ib_async import LimitOrder, MarketOrder, StopOrder

        action = "BUY" if order.side is Side.BUY else "SELL"
        qty = float(order.qty)

        if order.order_type is OrderType.MARKET:
            ib_order = MarketOrder(action, qty)
        elif order.order_type is OrderType.LIMIT:
            ib_order = LimitOrder(action, qty, float(order.limit_price))
        elif order.order_type is OrderType.STOP:
            ib_order = StopOrder(action, qty, float(order.stop_price))
        else:
            raise OrderRejected(f"ibkr venue does not implement {order.order_type.value}")

        ib_order.account = self._account
        ib_order.tif = _tif(order.time_in_force)
        # Fractional-share orders must be routed as such or IBKR rejects them.
        if order.qty != order.qty.to_integral_value():
            ib_order.cashQty = 0

        trade = self._ib.placeOrder(_contract_of(order.instrument), ib_order)
        self._ib.sleep(1)      # let the first status callback arrive

        status = (trade.orderStatus.status or "").lower()
        if status in ("inactive", "cancelled", "apicancelled"):
            order.status = OrderStatus.REJECTED
            order.reject_reason = _reject_reason(trade)
            raise OrderRejected(f"ibkr rejected the order: {order.reject_reason}")

        order.venue_order_id = str(trade.order.orderId)
        order.status = OrderStatus.OPEN
        order.updated_at = utcnow()
        log.info("ibkr: submitted %s -> %s", self.describe_order(order), order.venue_order_id)
        return order

    def cancel(self, order: Order) -> Order:
        if order.venue_order_id is None:
            raise VenueError(f"order {order.order_id} has no venue id")
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order.venue_order_id:
                self._ib.cancelOrder(trade.order)
                order.status = OrderStatus.CANCELED
                order.updated_at = utcnow()
                return order
        raise VenueError(f"ibkr has no open order {order.venue_order_id}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _valid(value: Any) -> bool:
    """IBKR uses -1 and nan for 'no data'."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number > 0      # nan != nan


def _tif(tif: TimeInForce) -> str:
    return {
        TimeInForce.GTC: "GTC",
        TimeInForce.DAY: "DAY",
        TimeInForce.IOC: "IOC",
        TimeInForce.FOK: "FOK",
    }.get(tif, "DAY")


def _reject_reason(trade: Any) -> str:
    for entry in reversed(getattr(trade, "log", []) or []):
        message = getattr(entry, "message", "")
        if message:
            return message
    return getattr(trade.orderStatus, "status", "unknown")


# Instruments are frozen dataclasses, so the IB contract is kept alongside
# rather than stored on them.
_CONTRACTS: dict[str, Any] = {}


def _attach_contract(instrument: Instrument, contract: Any) -> Instrument:
    _CONTRACTS[instrument.key] = contract
    return instrument


def _contract_of(instrument: Instrument) -> Any:
    contract = _CONTRACTS.get(instrument.key)
    if contract is None:
        raise VenueError(
            f"no IB contract cached for {instrument.symbol}; call resolve() first"
        )
    return contract


__all__ = ["IBKRVenue", "MARKET_DATA_DELAYED", "MARKET_DATA_LIVE", "PAPER_PREFIXES"]
