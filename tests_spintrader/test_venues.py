"""Tests for the venue abstraction and the paper broker.

Two themes carry most of the weight:

* **The live gate cannot be bypassed.** Order submission has exactly one path,
  and these tests pin that a venue implementation cannot route around it.
* **Fills are pessimistic.** Every test here that could plausibly be written
  two ways is written the way that costs the strategy money, because the
  opposite bias is what makes a backtest lie.
"""

from __future__ import annotations

import os
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import mock

from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    AssetClass, Instrument, Order, OrderStatus, OrderType, Position, Quote,
    Side, TradingMode, VenueId,
)
from spintrader.venues.base import (
    InsufficientFunds, NotConnected, OrderRejected, snap_to_increment,
)
from spintrader.venues.paper import PaperVenue, SlippageModel

D = Decimal
T0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)

BTC = Instrument(
    symbol="BTC-USD", asset_class=AssetClass.CRYPTO, venue=VenueId.PAPER,
    venue_symbol="BTC-USD", price_increment=D("0.1"), qty_increment=D("0.0001"),
    min_qty=D("0.0001"), min_notional=D("5"), taker_fee=D("0.0026"),
)
SHARE = Instrument(
    symbol="XYZ", asset_class=AssetClass.EQUITY, venue=VenueId.PAPER,
    venue_symbol="XYZ", price_increment=D("0.01"), qty_increment=D("1"),
    min_qty=D("1"), supports_fractional=False,
)


def settings(**kw) -> Settings:
    base = dict(mode=TradingMode.PAPER, aggression=Aggression.BALANCED,
                live=LiveGate(enabled=False), enforce_cash_account=False)
    base.update(kw)
    return Settings(**base)


class Book:
    """Mutable quote source, so tests can move the market."""

    def __init__(self, bid: str = "99", ask: str = "101"):
        self.bid, self.ask = D(bid), D(ask)

    def __call__(self, instrument: Instrument) -> Quote:
        return Quote(instrument.key, T0, bid=self.bid, ask=self.ask)

    def move(self, bid: str, ask: str) -> None:
        self.bid, self.ask = D(bid), D(ask)


def make_venue(cash="10000", book: Book | None = None, **kw) -> tuple[PaperVenue, Book]:
    book = book or Book()
    # No slippage by default so price assertions isolate one effect at a time.
    kw.setdefault("slippage", SlippageModel(base_bps=D("0"), impact_coefficient=D("0")))
    venue = PaperVenue(book, starting_cash=D(cash), settings=kw.pop("settings", settings()), **kw)
    venue.register(BTC)
    venue.register(SHARE)
    venue.connect()
    venue.set_time(T0)
    return venue, book


class SnapToIncrementTests(unittest.TestCase):
    def test_quantities_round_down(self):
        # Rounding a size up can overspend available cash.
        self.assertEqual(snap_to_increment(D("0.123456"), D("0.0001")), D("0.1234"))

    def test_exact_multiples_unchanged(self):
        self.assertEqual(snap_to_increment(D("0.5"), D("0.1")), D("0.5"))

    def test_zero_increment_is_a_noop(self):
        self.assertEqual(snap_to_increment(D("1.23456"), D("0")), D("1.23456"))


class ConnectionTests(unittest.TestCase):
    def test_submit_requires_connection(self):
        venue = PaperVenue(Book(), settings=settings())
        venue.register(BTC)
        with self.assertRaises(NotConnected):
            venue.submit(Order(BTC, Side.BUY, D("0.1")))

    def test_context_manager_connects(self):
        with PaperVenue(Book(), settings=settings()) as venue:
            self.assertTrue(venue._connected)


class ValidationTests(unittest.TestCase):
    def test_quantity_snapped_to_increment(self):
        venue, _ = make_venue()
        order = venue.submit(Order(BTC, Side.BUY, D("0.123456")))
        self.assertEqual(order.qty, D("0.1234"))

    def test_quantity_rounding_to_zero_rejected(self):
        venue, _ = make_venue()
        with self.assertRaises(OrderRejected) as ctx:
            venue.submit(Order(BTC, Side.BUY, D("0.00001")))
        self.assertIn("rounds to zero", str(ctx.exception))

    def test_below_minimum_notional_rejected(self):
        venue, _ = make_venue()
        with self.assertRaises(OrderRejected) as ctx:
            venue.submit(Order(BTC, Side.BUY, D("0.0001")))   # ~$0.01
        self.assertIn("below", str(ctx.exception))

    def test_whole_unit_instrument_rounds_down_rather_than_rejecting(self):
        # qty_increment=1 is the venue's real rule; 1.5 shares means "buy 1",
        # consistent with 0.123456 BTC meaning "buy 0.1234".
        venue, _ = make_venue()
        order = venue.submit(Order(SHARE, Side.BUY, D("1.5")))
        self.assertEqual(order.qty, D("1"))

    def test_inconsistent_instrument_definition_rejected(self):
        # Non-fractional but with a sub-unit increment is a config error, and
        # would otherwise silently permit fractional quantities.
        venue, _ = make_venue()
        broken = Instrument(
            "BAD", AssetClass.EQUITY, VenueId.PAPER, "BAD",
            qty_increment=D("0.01"), supports_fractional=False,
        )
        venue.register(broken)
        with self.assertRaises(OrderRejected) as ctx:
            venue.submit(Order(broken, Side.BUY, D("1.5")))
        self.assertIn("inconsistent", str(ctx.exception))

    def test_wrong_venue_rejected(self):
        venue, _ = make_venue()
        foreign = Instrument("BTC-USD", AssetClass.CRYPTO, VenueId.KRAKEN, "XBTUSD")
        with self.assertRaises(OrderRejected) as ctx:
            venue.submit(Order(foreign, Side.BUY, D("0.1")))
        self.assertIn("submitted to", str(ctx.exception))

    def test_resubmitting_a_live_order_rejected(self):
        venue, _ = make_venue()
        order = venue.submit(Order(BTC, Side.BUY, D("0.1")))
        with self.assertRaises(OrderRejected) as ctx:
            venue.submit(order)
        self.assertIn("DRAFT", str(ctx.exception))

    def test_limit_price_snapped(self):
        venue, book = make_venue()
        book.move("50", "51")
        order = venue.submit(Order(BTC, Side.BUY, D("0.2"),
                                   order_type=OrderType.LIMIT, limit_price=D("40.06")))
        self.assertEqual(order.limit_price, D("40.1"))


class LiveGateTests(unittest.TestCase):
    """The gate must hold even when the venue itself is willing."""

    def test_paper_mode_is_never_gated(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))   # must not raise

    def test_live_mode_blocked_when_disarmed(self):
        from spintrader.core.config import LiveTradingDisarmed
        venue, _ = make_venue(settings=settings(mode=TradingMode.LIVE,
                                                live=LiveGate(enabled=False)))
        with self.assertRaises(LiveTradingDisarmed):
            venue.submit(Order(BTC, Side.BUY, D("0.1")))

    def test_notional_cap_applies_to_the_normalised_size(self):
        # The cap must be evaluated after snapping, on what actually goes out.
        from spintrader.core.config import LiveTradingDisarmed
        gate = LiveGate(enabled=True, venues=frozenset({VenueId.PAPER}),
                        max_order_notional={VenueId.PAPER: D("15")})
        venue, _ = make_venue(settings=settings(mode=TradingMode.LIVE, live=gate))
        venue.submit(Order(BTC, Side.BUY, D("0.1")))               # ~$10.10, ok
        with self.assertRaises(LiveTradingDisarmed):
            venue.submit(Order(BTC, Side.BUY, D("1")))             # ~$101

    def test_market_orders_are_capped_using_a_live_quote(self):
        # Market orders have no intrinsic price; without a quote lookup they
        # would slip past the notional cap entirely.
        from spintrader.core.config import LiveTradingDisarmed
        gate = LiveGate(enabled=True, venues=frozenset({VenueId.PAPER}),
                        max_order_notional={VenueId.PAPER: D("10")})
        venue, _ = make_venue(settings=settings(mode=TradingMode.LIVE, live=gate))
        with self.assertRaises(LiveTradingDisarmed):
            venue.submit(Order(BTC, Side.BUY, D("1"), order_type=OrderType.MARKET))

    def test_dry_run_transmits_nothing(self):
        venue, _ = make_venue(settings=settings(dry_run=True))
        order = venue.submit(Order(BTC, Side.BUY, D("0.1")))
        self.assertEqual(order.status, OrderStatus.REJECTED_BY_RISK)
        self.assertEqual(venue.fills, [])


class FillPricingTests(unittest.TestCase):
    def test_buys_cross_to_the_ask_not_the_mid(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        self.assertEqual(venue.fills[0].price, D("101"))

    def test_sells_cross_to_the_bid(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.2")))
        venue.submit(Order(BTC, Side.SELL, D("0.1")))
        self.assertEqual(venue.fills[1].price, D("99"))

    def test_fees_charged_at_the_instrument_rate(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        fill = venue.fills[0]
        self.assertEqual(fill.fee, D("0.1") * D("101") * D("0.0026"))

    def test_slippage_worsens_both_sides(self):
        venue, _ = make_venue(slippage=SlippageModel(base_bps=D("10"),
                                                     impact_coefficient=D("0")))
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        buy = venue.fills[0].price
        venue.submit(Order(BTC, Side.SELL, D("0.1")))
        sell = venue.fills[1].price
        self.assertGreater(buy, D("101"))     # paid more than the ask
        self.assertLess(sell, D("99"))        # received less than the bid

    def test_impact_grows_with_size(self):
        model = SlippageModel(base_bps=D("1"), impact_coefficient=D("10"),
                              reference_size=D("100000"))
        self.assertLess(model.slippage_bps(D("1000")), model.slippage_bps(D("100000")))

    def test_round_trip_at_a_flat_market_loses_money(self):
        # Spread plus fees must make a zero-move round trip unprofitable.
        # If this ever passes at break-even, the cost model is broken and every
        # backtest downstream is optimistic.
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        venue.submit(Order(BTC, Side.SELL, D("0.1")))
        self.assertLess(venue.realized_pnl - venue.total_fees, D("0"))


class AffordabilityTests(unittest.TestCase):
    def test_cannot_spend_more_cash_than_held(self):
        venue, _ = make_venue(cash="100")
        with self.assertRaises(InsufficientFunds) as ctx:
            venue.submit(Order(BTC, Side.BUY, D("10")))     # ~$1010
        self.assertIn("only", str(ctx.exception))

    def test_cash_decreases_by_gross_plus_fee(self):
        venue, _ = make_venue(cash="1000")
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        expected = D("1000") - (D("0.1") * D("101")) - (D("0.1") * D("101") * D("0.0026"))
        self.assertEqual(venue.snapshot().balances["USD"].total, expected)

    def test_cash_account_cannot_short(self):
        venue, _ = make_venue(settings=settings(enforce_cash_account=True))
        with self.assertRaises(OrderRejected) as ctx:
            venue.submit(Order(BTC, Side.SELL, D("0.1")))
        self.assertIn("cannot short", str(ctx.exception))


class SettlementTests(unittest.TestCase):
    """T+1 settlement, mirroring the real IBKR cash account.

    Without this a backtest can round-trip the same dollar repeatedly in one
    session and manufacture returns the real account cannot earn.
    """

    def setUp(self):
        self.venue, self.book = make_venue(
            cash="1000", settings=settings(enforce_cash_account=True), settlement_days=1
        )

    def test_sale_proceeds_are_not_immediately_available(self):
        self.venue.submit(Order(BTC, Side.BUY, D("0.5")))
        cash_after_buy = self.venue.snapshot().balances["USD"].available
        self.venue.submit(Order(BTC, Side.SELL, D("0.5")))
        available = self.venue.snapshot().balances["USD"].available
        self.assertEqual(available, cash_after_buy)   # proceeds still unsettled

    def test_total_includes_unsettled_but_available_does_not(self):
        self.venue.submit(Order(BTC, Side.BUY, D("0.5")))
        self.venue.submit(Order(BTC, Side.SELL, D("0.5")))
        bal = self.venue.snapshot().balances["USD"]
        self.assertGreater(bal.total, bal.available)
        self.assertGreater(bal.held, D("0"))

    def test_proceeds_settle_the_next_day(self):
        self.venue.submit(Order(BTC, Side.BUY, D("0.5")))
        self.venue.submit(Order(BTC, Side.SELL, D("0.5")))
        before = self.venue.snapshot().balances["USD"].available
        self.venue.set_time(T0 + timedelta(days=1, seconds=1))
        after = self.venue.snapshot().balances["USD"].available
        self.assertGreater(after, before)

    def test_cannot_rebuy_with_unsettled_proceeds(self):
        # The good-faith violation scenario, which a naive simulator permits.
        self.venue.submit(Order(BTC, Side.BUY, D("9")))       # ~$909 of $1000
        self.venue.submit(Order(BTC, Side.SELL, D("9")))
        with self.assertRaises(InsufficientFunds):
            self.venue.submit(Order(BTC, Side.BUY, D("8")))

    def test_settlement_disabled_when_not_a_cash_account(self):
        venue, _ = make_venue(cash="1000", settings=settings(enforce_cash_account=False))
        venue.submit(Order(BTC, Side.BUY, D("0.5")))
        venue.submit(Order(BTC, Side.SELL, D("0.5")))
        bal = venue.snapshot().balances["USD"]
        self.assertEqual(bal.total, bal.available)


class RestingOrderTests(unittest.TestCase):
    def test_non_marketable_limit_does_not_fill_immediately(self):
        # The classic backtest lie: filling a limit the market never reached.
        venue, _ = make_venue()
        order = venue.submit(Order(BTC, Side.BUY, D("0.1"),
                                   order_type=OrderType.LIMIT, limit_price=D("90")))
        self.assertEqual(order.status, OrderStatus.OPEN)
        self.assertEqual(venue.fills, [])

    def test_marketable_limit_fills_at_once(self):
        venue, _ = make_venue()
        order = venue.submit(Order(BTC, Side.BUY, D("0.1"),
                                   order_type=OrderType.LIMIT, limit_price=D("110")))
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_limit_never_fills_worse_than_its_price(self):
        venue, _ = make_venue(slippage=SlippageModel(base_bps=D("500")))
        order = venue.submit(Order(BTC, Side.BUY, D("0.1"),
                                   order_type=OrderType.LIMIT, limit_price=D("101")))
        self.assertLessEqual(venue.fills[0].price, D("101"))
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_resting_limit_fills_once_the_market_reaches_it(self):
        venue, book = make_venue()
        order = venue.submit(Order(BTC, Side.BUY, D("0.1"),
                                   order_type=OrderType.LIMIT, limit_price=D("90")))
        self.assertEqual(order.status, OrderStatus.OPEN)
        book.move("88", "89")
        venue.set_time(T0 + timedelta(minutes=1))
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_stop_triggers_on_adverse_move(self):
        venue, book = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.5")))
        stop = venue.submit(Order(BTC, Side.SELL, D("0.5"),
                                  order_type=OrderType.STOP, stop_price=D("95")))
        self.assertEqual(stop.status, OrderStatus.OPEN)
        book.move("94", "95")
        venue.set_time(T0 + timedelta(minutes=1))
        self.assertEqual(stop.status, OrderStatus.FILLED)

    def test_cancel_removes_a_resting_order(self):
        venue, book = make_venue()
        order = venue.submit(Order(BTC, Side.BUY, D("0.1"),
                                   order_type=OrderType.LIMIT, limit_price=D("90")))
        venue.cancel(order)
        self.assertEqual(order.status, OrderStatus.CANCELED)
        book.move("88", "89")
        venue.set_time(T0 + timedelta(minutes=1))
        self.assertEqual(venue.fills, [])


class ReconciliationTests(unittest.TestCase):
    def test_agreeing_books_report_no_divergence(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        actual = venue.snapshot().positions
        self.assertEqual(venue.reconcile(actual), [])

    def test_divergence_is_reported(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        wrong = {BTC.key: Position(BTC.key, qty=D("0.5"))}
        divergences = venue.reconcile(wrong)
        self.assertEqual(len(divergences), 1)
        self.assertIn("ledger says 0.5", divergences[0])

    def test_position_missing_from_the_ledger_is_caught(self):
        venue, _ = make_venue()
        venue.submit(Order(BTC, Side.BUY, D("0.1")))
        self.assertEqual(len(venue.reconcile({})), 1)


if __name__ == "__main__":
    unittest.main()
