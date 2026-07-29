"""Tests for the portfolio ledger.

Three areas get disproportionate coverage, all chosen because the bug they
guard against is silent:

* **Fill idempotency.** A re-delivered fill double-counts a position and
  invents P&L, and nothing raises.
* **Multi-currency equity.** The real book holds USD at Kraken and CAD at
  IBKR. Adding them unconverted is a ~40% error that looks like a plausible
  number.
* **Settlement.** Spending unsettled proceeds is a good-faith violation the
  broker enforces and a naive ledger permits.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.core.types import (
    Balance, Fill, Position, Side, TradingMode, VenueId,
)
from spintrader.portfolio.ledger import (
    CashAccount, Ledger, SettlementBucket, UnknownRate,
)

D = Decimal
T0 = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
BTC = "kraken:BTC-USD"
XYZ = "ibkr:XYZ"


def fill(side=Side.BUY, qty="1", price="100", fee="0", key=BTC,
         offset_s=0, fill_id=None, fee_currency="USD"):
    kwargs = dict(
        order_id="o1", instrument_key=key, side=side, qty=D(qty), price=D(price),
        ts=T0 + timedelta(seconds=offset_s), fee=D(fee), fee_currency=fee_currency,
    )
    if fill_id is not None:
        kwargs["fill_id"] = fill_id
    return Fill(**kwargs)


def ledger(**kw) -> Ledger:
    params = dict(base_currency="USD", opening_cash={"USD": D("1000")})
    params.update(kw)
    return Ledger(**params)


class CashAccountTests(unittest.TestCase):
    def test_credit_and_debit_settled(self):
        acct = CashAccount("USD")
        acct.credit(D("100"))
        acct.debit(D("30"))
        self.assertEqual(acct.settled, D("70"))
        self.assertEqual(acct.available, D("70"))

    def test_pending_counts_toward_total_but_not_available(self):
        acct = CashAccount("USD")
        acct.credit(D("50"))
        acct.credit(D("100"), available_at=T0 + timedelta(days=1))
        self.assertEqual(acct.total, D("150"))
        self.assertEqual(acct.available, D("50"))
        self.assertEqual(acct.unsettled, D("100"))

    def test_release_matured_buckets_only(self):
        acct = CashAccount("USD")
        acct.credit(D("10"), available_at=T0 + timedelta(days=1))
        acct.credit(D("20"), available_at=T0 + timedelta(days=3))
        released = acct.release_settled(T0 + timedelta(days=2))
        self.assertEqual(released, D("10"))
        self.assertEqual(acct.available, D("10"))
        self.assertEqual(acct.unsettled, D("20"))

    def test_negative_balance_is_permitted(self):
        # Fees can be charged against an empty account; clamping to zero would
        # hide a reconciliation break rather than surface it.
        acct = CashAccount("USD")
        acct.debit(D("5"))
        self.assertEqual(acct.settled, D("-5"))


class IdempotencyTests(unittest.TestCase):
    """Venues re-deliver fills. Applying one twice must be a no-op."""

    def test_duplicate_fill_id_ignored(self):
        book = ledger()
        f = fill(qty="1", price="100", fill_id="F1")
        self.assertTrue(book.apply_fill(f))
        self.assertFalse(book.apply_fill(f))
        self.assertEqual(book.position(BTC).qty, D("1"))
        self.assertEqual(book.fill_count, 1)

    def test_duplicate_does_not_move_cash(self):
        book = ledger()
        f = fill(qty="1", price="100", fill_id="F1")
        book.apply_fill(f)
        cash_after_first = book.account("USD").settled
        book.apply_fill(f)
        self.assertEqual(book.account("USD").settled, cash_after_first)

    def test_distinct_fills_with_same_economics_both_apply(self):
        # Two genuine fills that happen to be identical in size and price must
        # not be mistaken for a duplicate.
        book = ledger()
        book.apply_fill(fill(qty="1", price="100", fill_id="F1"))
        book.apply_fill(fill(qty="1", price="100", fill_id="F2"))
        self.assertEqual(book.position(BTC).qty, D("2"))

    def test_apply_fills_reports_applied_count(self):
        book = ledger()
        f = fill(fill_id="F1")
        applied = book.apply_fills([f, f, fill(fill_id="F2")])
        self.assertEqual(applied, 2)


class CashFlowTests(unittest.TestCase):
    def test_buy_reduces_cash_by_gross(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="2", price="100"))
        self.assertEqual(book.account("USD").settled, D("800"))

    def test_fee_reduces_cash_and_is_tracked_separately(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100", fee="0.26"))
        self.assertEqual(book.account("USD").settled, D("899.74"))
        self.assertEqual(book.fees_paid, D("0.26"))

    def test_sell_credits_proceeds_immediately_without_settlement(self):
        book = ledger(settlement_days=0)
        book.apply_fill(fill(Side.BUY, qty="2", price="100"))
        book.apply_fill(fill(Side.SELL, qty="2", price="110", offset_s=1))
        self.assertEqual(book.account("USD").settled, D("1020"))
        self.assertEqual(book.account("USD").unsettled, D("0"))

    def test_fees_charged_immediately_even_under_settlement(self):
        # The venue does not wait for T+1 to take its fee.
        book = ledger(settlement_days=1)
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        book.apply_fill(fill(Side.SELL, qty="1", price="100", fee="0.26", offset_s=1))
        self.assertEqual(book.fees_paid, D("0.26"))
        # Settled cash reflects the fee but not the proceeds.
        self.assertEqual(book.account("USD").settled, D("899.74"))


class SettlementTests(unittest.TestCase):
    def test_proceeds_unavailable_until_settled(self):
        book = ledger(settlement_days=1)
        book.apply_fill(fill(Side.BUY, qty="5", price="100"))
        book.apply_fill(fill(Side.SELL, qty="5", price="100", offset_s=1))
        acct = book.account("USD")
        self.assertEqual(acct.available, D("500"))
        self.assertEqual(acct.unsettled, D("500"))
        self.assertEqual(acct.total, D("1000"))

    def test_settle_releases_after_the_window(self):
        book = ledger(settlement_days=1)
        book.apply_fill(fill(Side.BUY, qty="5", price="100"))
        book.apply_fill(fill(Side.SELL, qty="5", price="100", offset_s=1))
        released = book.settle(T0 + timedelta(days=1, seconds=2))
        self.assertEqual(released, D("500"))
        self.assertEqual(book.account("USD").available, D("1000"))

    def test_settle_is_a_noop_before_the_window(self):
        book = ledger(settlement_days=1)
        book.apply_fill(fill(Side.BUY, qty="5", price="100"))
        book.apply_fill(fill(Side.SELL, qty="5", price="100", offset_s=1))
        self.assertEqual(book.settle(T0 + timedelta(hours=1)), D("0"))


class PnLTests(unittest.TestCase):
    def test_realized_pnl_on_a_round_trip(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="2", price="100"))
        book.apply_fill(fill(Side.SELL, qty="2", price="120", offset_s=1))
        self.assertEqual(book.realized_pnl, D("40"))

    def test_unrealized_pnl_from_marks(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="2", price="100"))
        book.mark({BTC: D("110")})
        snapshot = book.value()
        self.assertEqual(snapshot.unrealized_pnl, D("20"))

    def test_net_pnl_subtracts_fees(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100", fee="0.26"))
        book.apply_fill(fill(Side.SELL, qty="1", price="110", fee="0.29", offset_s=1))
        snapshot = book.value()
        self.assertEqual(snapshot.realized_pnl, D("10"))
        self.assertEqual(snapshot.fees_paid, D("0.55"))
        self.assertEqual(snapshot.net_pnl, D("9.45"))

    def test_equity_unchanged_by_a_flat_round_trip_except_fees(self):
        book = ledger()
        before = book.value().equity
        book.apply_fill(fill(Side.BUY, qty="1", price="100", fee="0.26"))
        book.apply_fill(fill(Side.SELL, qty="1", price="100", fee="0.26", offset_s=1))
        after = book.value().equity
        self.assertEqual(before - after, D("0.52"))


class MultiCurrencyTests(unittest.TestCase):
    """The real book: USD at Kraken, CAD at IBKR."""

    def test_cad_converted_into_base_equity(self):
        book = Ledger(base_currency="USD",
                      opening_cash={"USD": D("700"), "CAD": D("100")})
        snapshot = book.value(rates={("CAD", "USD"): D("0.71")})
        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.equity, D("771.00"))

    def test_inverse_rate_accepted(self):
        book = Ledger(base_currency="USD", opening_cash={"CAD": D("141")})
        snapshot = book.value(rates={("USD", "CAD"): D("1.41")})
        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.equity, D("100"))

    def test_missing_rate_marks_equity_incomplete(self):
        # Reporting a smaller number as though it were total would make the
        # risk engine size against a fiction.
        book = Ledger(base_currency="USD",
                      opening_cash={"USD": D("700"), "CAD": D("100")})
        snapshot = book.value()
        self.assertFalse(snapshot.complete)
        self.assertIn("CAD", snapshot.unconvertible)

    def test_missing_mark_marks_equity_incomplete(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        book.positions[BTC].last_price = None
        snapshot = book.value()
        self.assertFalse(snapshot.complete)
        self.assertIn(BTC, snapshot.unpriced)

    def test_cad_position_converted(self):
        book = Ledger(base_currency="USD", opening_cash={"CAD": D("1000")})
        book.apply_fill(fill(Side.BUY, qty="10", price="14", key=XYZ,
                             fee_currency="CAD"), quote_currency="CAD")
        book.mark({XYZ: D("14")})
        snapshot = book.value(
            rates={("CAD", "USD"): D("0.71")},
            quote_currencies={XYZ: "CAD"},
        )
        self.assertTrue(snapshot.complete)
        # 1000 CAD total value regardless of the split -> 710 USD.
        self.assertEqual(snapshot.equity, D("710.00"))


class RiskStateTests(unittest.TestCase):
    def test_builds_state_for_the_risk_engine(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        book.mark({BTC: D("110")})
        state = book.to_risk_state()
        self.assertEqual(state.base_currency, "USD")
        self.assertEqual(state.available_cash, D("900"))
        self.assertEqual(state.trades_today, 1)

    def test_incomplete_valuation_refuses_to_produce_state(self):
        # Better to skip a cycle than to size against a book you cannot see.
        book = Ledger(base_currency="USD",
                      opening_cash={"USD": D("700"), "CAD": D("100")})
        with self.assertRaises(UnknownRate):
            book.to_risk_state()

    def test_peak_equity_tracked_for_drawdown(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        book.mark({BTC: D("200")})
        book.value()                       # equity peaks here
        book.mark({BTC: D("50")})
        state = book.to_risk_state()
        self.assertLess(state.drawdown, D("0"))

    def test_daily_counters_reset_on_a_new_day(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        self.assertEqual(book.to_risk_state().trades_today, 1)
        book.apply_fill(fill(Side.BUY, qty="1", price="100",
                             offset_s=86_400 * 2, fill_id="F-next-day"))
        self.assertEqual(book.to_risk_state().trades_today, 1)


class ReconciliationTests(unittest.TestCase):
    def test_agreement_reports_nothing(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        venue = {BTC: Position(BTC, qty=D("1"))}
        self.assertEqual(book.reconcile_positions(venue), [])

    def test_quantity_divergence_reported(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        venue = {BTC: Position(BTC, qty=D("2"))}
        divergences = book.reconcile_positions(venue)
        self.assertEqual(len(divergences), 1)
        self.assertIn("ledger 1", divergences[0])

    def test_position_only_at_the_venue_is_reported(self):
        book = ledger()
        venue = {BTC: Position(BTC, qty=D("1"))}
        self.assertEqual(len(book.reconcile_positions(venue)), 1)

    def test_cash_tolerance_ignores_sub_cent_noise(self):
        # Venues round fees in ways a ledger cannot reproduce; flagging that
        # trains you to ignore the alert.
        book = ledger()
        venue = {"USD": Balance("USD", D("1000.005"), D("1000.005"), VenueId.KRAKEN)}
        self.assertEqual(book.reconcile_cash(venue), [])

    def test_material_cash_divergence_reported(self):
        book = ledger()
        venue = {"USD": Balance("USD", D("900"), D("900"), VenueId.KRAKEN)}
        self.assertEqual(len(book.reconcile_cash(venue)), 1)

    def test_adopt_venue_state_overwrites_and_reports(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        venue_positions = {BTC: Position(BTC, qty=D("3"), avg_cost=D("105"))}
        venue_balances = {"USD": Balance("USD", D("500"), D("500"), VenueId.KRAKEN)}
        changes = book.adopt_venue_state(venue_positions, venue_balances)
        self.assertTrue(changes)
        self.assertEqual(book.position(BTC).qty, D("3"))
        self.assertEqual(book.account("USD").settled, D("500"))

    def test_adopt_preserves_basis_the_venue_does_not_report(self):
        # Kraken reports quantity only; discarding our cost basis would destroy
        # all P&L attribution.
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        venue_positions = {BTC: Position(BTC, qty=D("1"), avg_cost=D("0"))}
        book.adopt_venue_state(venue_positions, {})
        self.assertEqual(book.position(BTC).avg_cost, D("100"))

    def test_adopt_flattens_positions_absent_at_the_venue(self):
        book = ledger()
        book.apply_fill(fill(Side.BUY, qty="1", price="100"))
        book.adopt_venue_state({}, {})
        self.assertTrue(book.position(BTC).is_flat)


if __name__ == "__main__":
    unittest.main()
