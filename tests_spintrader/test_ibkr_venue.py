"""Tests for the IBKR venue client. ib_async is stubbed throughout.

The safety-critical test here is
:meth:`ConnectionTests.test_live_account_in_paper_mode_is_fatal`. Connecting to
a ``U...`` live account while the system believes it is paper trading is the
one misconfiguration that spends real money, and it must be an exception
rather than a log line.
"""

from __future__ import annotations

import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest import mock

from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    Order, OrderStatus, OrderType, Side, TradingMode, VenueId,
)
from spintrader.venues.base import OrderRejected, VenueError
from spintrader.venues.ibkr import IBKRVenue, _valid

D = Decimal


def settings(**kw) -> Settings:
    base = dict(mode=TradingMode.PAPER, aggression=Aggression.BALANCED,
                live=LiveGate(enabled=False), base_currency="USD")
    base.update(kw)
    return Settings(**base)


class FakeIB:
    """Minimal ib_async.IB stand-in."""

    def __init__(self, accounts=("DUQ980038",), summary=None, positions=(),
                 ticker=None, contract_details=True):
        self._accounts = list(accounts)
        self._summary = summary if summary is not None else [
            SimpleNamespace(tag="NetLiquidation", value="100.00", currency="USD"),
            SimpleNamespace(tag="TotalCashValue", value="100.00", currency="USD"),
            SimpleNamespace(tag="SettledCash", value="60.00", currency="USD"),
        ]
        self._positions = list(positions)
        self._ticker = ticker
        self._contract_details = contract_details
        self.connected = False
        self.market_data_type = None
        self.placed = []

    def connect(self, *_a, **_kw):
        self.connected = True

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    def reqMarketDataType(self, value):
        self.market_data_type = value

    def managedAccounts(self):
        return self._accounts

    def accountSummary(self, _account=None):
        return self._summary

    def positions(self, _account=None):
        return self._positions

    def reqContractDetails(self, contract):
        if not self._contract_details:
            return []
        return [SimpleNamespace(
            minTick=0.01, stockType="COMMON",
            contract=SimpleNamespace(symbol=contract.symbol, currency="USD"),
        )]

    def reqTickers(self, _contract):
        return [self._ticker] if self._ticker else []

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        trade = SimpleNamespace(
            order=SimpleNamespace(orderId=42),
            orderStatus=SimpleNamespace(status="Submitted"),
            log=[],
        )
        return trade

    def sleep(self, _seconds):
        return

    def openTrades(self):
        return []


def ticker(bid=100.0, ask=100.5, last=100.2, close=100.1, bid_size=5, ask_size=5):
    return SimpleNamespace(bid=bid, ask=ask, last=last, close=close,
                           bidSize=bid_size, askSize=ask_size)


def make_venue(ib=None, **kw) -> tuple[IBKRVenue, FakeIB]:
    ib = ib or FakeIB(ticker=ticker())
    venue = IBKRVenue(settings=kw.pop("settings", settings()), ib=ib, **kw)
    with mock.patch.dict("os.environ", {}, clear=True):
        venue.connect()
    return venue, ib


class ValidHelperTests(unittest.TestCase):
    def test_rejects_ibkr_no_data_sentinels(self):
        # IBKR signals "no data" with -1 and nan, not with an error.
        self.assertFalse(_valid(-1))
        self.assertFalse(_valid(float("nan")))
        self.assertFalse(_valid(0))
        self.assertFalse(_valid(None))

    def test_accepts_real_prices(self):
        self.assertTrue(_valid(100.5))
        self.assertTrue(_valid("42"))


class ConnectionTests(unittest.TestCase):
    def test_delayed_data_requested_by_default(self):
        # The account has no market-data subscriptions on purpose.
        _, ib = make_venue()
        self.assertEqual(ib.market_data_type, 3)

    def test_paper_account_accepted(self):
        venue, _ = make_venue()
        self.assertEqual(venue._account, "DUQ980038")

    def test_live_account_in_paper_mode_is_fatal(self):
        # The misconfiguration that spends real money. Must raise, not warn.
        ib = FakeIB(accounts=("U23271203",))
        venue = IBKRVenue(settings=settings(mode=TradingMode.PAPER), ib=ib)
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(VenueError) as ctx:
                venue.connect()
        self.assertIn("LIVE account", str(ctx.exception))

    def test_live_account_allowed_in_live_mode(self):
        ib = FakeIB(accounts=("U23271203",))
        venue = IBKRVenue(settings=settings(mode=TradingMode.LIVE), ib=ib)
        with mock.patch.dict("os.environ", {}, clear=True):
            venue.connect()
        self.assertEqual(venue._account, "U23271203")

    def test_no_accounts_raises(self):
        venue = IBKRVenue(settings=settings(), ib=FakeIB(accounts=()))
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(VenueError):
                venue.connect()

    def test_explicit_account_must_exist(self):
        venue = IBKRVenue(settings=settings(), ib=FakeIB(accounts=("DUQ980038",)))
        with mock.patch.dict("os.environ", {"IBKR_ACCOUNT": "DU999"}, clear=True):
            with self.assertRaises(VenueError) as ctx:
                venue.connect()
        self.assertIn("not among", str(ctx.exception))


class ResolveTests(unittest.TestCase):
    def test_resolves_a_stock(self):
        venue, _ = make_venue()
        inst = venue.resolve("AAPL")
        self.assertEqual(inst.symbol, "AAPL")
        self.assertEqual(inst.venue, VenueId.IBKR)
        self.assertEqual(inst.price_increment, D("0.01"))

    def test_fractional_minimums_reflect_ibkr_rules(self):
        venue, _ = make_venue()
        inst = venue.resolve("AAPL")
        self.assertTrue(inst.supports_fractional)
        self.assertEqual(inst.min_notional, D("1"))     # IBKR's USD 1.00 minimum

    def test_unknown_symbol_raises(self):
        venue, _ = make_venue(ib=FakeIB(ticker=ticker(), contract_details=False))
        with self.assertRaises(VenueError):
            venue.resolve("NOTATICKER")

    def test_resolution_is_cached(self):
        venue, _ = make_venue()
        self.assertIs(venue.resolve("AAPL"), venue.resolve("AAPL"))


class QuoteTests(unittest.TestCase):
    def test_uses_bid_and_ask(self):
        venue, _ = make_venue()
        quote = venue.get_quote(venue.resolve("AAPL"))
        self.assertEqual(quote.bid, D("100"))
        self.assertEqual(quote.ask, D("100.5"))

    def test_falls_back_to_last_when_book_is_empty(self):
        # Outside RTH IBKR returns -1 for the touch; without a fallback the
        # Quote would carry a negative spread and poison every downstream calc.
        venue, _ = make_venue(ib=FakeIB(ticker=ticker(bid=-1, ask=-1, last=99.0)))
        quote = venue.get_quote(venue.resolve("AAPL"))
        self.assertEqual(quote.bid, D("99"))
        self.assertEqual(quote.ask, D("99"))
        self.assertEqual(quote.spread, D("0"))

    def test_falls_back_to_close_when_last_is_absent(self):
        venue, _ = make_venue(
            ib=FakeIB(ticker=ticker(bid=-1, ask=-1, last=float("nan"), close=98.0))
        )
        self.assertEqual(venue.get_quote(venue.resolve("AAPL")).mid, D("98"))

    def test_raises_when_no_price_is_available_at_all(self):
        venue, _ = make_venue(
            ib=FakeIB(ticker=ticker(bid=-1, ask=-1, last=-1, close=-1))
        )
        with self.assertRaises(VenueError) as ctx:
            venue.get_quote(venue.resolve("AAPL"))
        self.assertIn("no usable price", str(ctx.exception))


class SnapshotTests(unittest.TestCase):
    def test_settled_cash_becomes_available_not_total(self):
        # TotalCashValue includes unsettled proceeds a cash account cannot
        # spend; sizing against it invites a good-faith violation.
        venue, _ = make_venue()
        balance = venue.snapshot().balances["USD"]
        self.assertEqual(balance.total, D("100"))
        self.assertEqual(balance.available, D("60"))
        self.assertEqual(balance.held, D("40"))

    def test_equity_from_net_liquidation(self):
        venue, _ = make_venue()
        self.assertEqual(venue.snapshot().equity, D("100"))

    def test_positions_carry_cost_basis(self):
        ib = FakeIB(ticker=ticker(), positions=[
            SimpleNamespace(contract=SimpleNamespace(symbol="AAPL"),
                            position=2.0, avgCost=150.0),
        ])
        venue, _ = make_venue(ib=ib)
        position = venue.snapshot().positions["ibkr:AAPL"]
        self.assertEqual(position.qty, D("2"))
        self.assertEqual(position.avg_cost, D("150"))


class OrderTests(unittest.TestCase):
    def test_limit_order_fields(self):
        venue, ib = make_venue()
        inst = venue.resolve("AAPL")
        venue._transmit(Order(inst, Side.BUY, D("1"),
                              order_type=OrderType.LIMIT, limit_price=D("99")))
        _, ib_order = ib.placed[0]
        self.assertEqual(ib_order.action, "BUY")
        self.assertEqual(ib_order.totalQuantity, 1)
        self.assertEqual(ib_order.lmtPrice, 99)
        self.assertEqual(ib_order.account, "DUQ980038")

    def test_order_id_recorded(self):
        venue, _ = make_venue()
        order = venue._transmit(Order(venue.resolve("AAPL"), Side.BUY, D("1")))
        self.assertEqual(order.venue_order_id, "42")
        self.assertEqual(order.status, OrderStatus.OPEN)

    def test_rejection_raises_and_marks_the_order(self):
        class RejectingIB(FakeIB):
            def placeOrder(self, contract, order):
                return SimpleNamespace(
                    order=SimpleNamespace(orderId=7),
                    orderStatus=SimpleNamespace(status="Inactive"),
                    log=[SimpleNamespace(message="Order size below minimum")],
                )

        venue, _ = make_venue(ib=RejectingIB(ticker=ticker()))
        order = Order(venue.resolve("AAPL"), Side.BUY, D("1"))
        with self.assertRaises(OrderRejected):
            venue._transmit(order)
        self.assertEqual(order.status, OrderStatus.REJECTED)
        self.assertIn("below minimum", order.reject_reason)

    def test_unsupported_order_type_rejected(self):
        venue, _ = make_venue()
        order = Order(venue.resolve("AAPL"), Side.BUY, D("1"),
                      order_type=OrderType.STOP_LIMIT,
                      stop_price=D("90"), limit_price=D("89"))
        with self.assertRaises(OrderRejected):
            venue._transmit(order)

    def test_cancel_without_venue_id_raises(self):
        venue, _ = make_venue()
        with self.assertRaises(VenueError):
            venue.cancel(Order(venue.resolve("AAPL"), Side.BUY, D("1")))


if __name__ == "__main__":
    unittest.main()
