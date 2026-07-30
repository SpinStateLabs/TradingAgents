"""Regression tests for sizing a risk-*reducing* trade.

The defect these pin down: :meth:`RiskEngine.evaluate` sized every trade from a
set of candidates that all answer "how much more risk may this take?" --
``kelly``, ``vol_target``, ``max_position_weight``, ``gross_exposure_headroom``,
``position_headroom``. Asked of a trade that removes risk, those candidates
invert their own purpose:

* at ``max_position_weight`` the position headroom is exactly zero, so an exit
  was rejected with "no room to add risk" -- the position most in need of
  closing was the only one that could not be closed, and a stop-loss could
  never fire; and
* below the cap, only the unused sliver of budget could be sold, so an exit
  dribbled out over many bars while the loss it existed to stop kept accruing.

Both are the kind of failure that looks like a merely conservative backtest
until a real position is trapped in a real drawdown. Every test below fails
against the pre-fix engine.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    AssetClass, Instrument, Position, Quote, Side, TradingMode, VenueId,
)
from spintrader.risk.engine import (
    Mandate, PortfolioState, RiskEngine, RiskVerdict, TradeIntent,
)

D = Decimal
NOW = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
PRICE = D("600")

SPY = Instrument(
    symbol="SPY", asset_class=AssetClass.EQUITY, venue=VenueId.PAPER,
    venue_symbol="SPY", quote_currency="USD",
    price_increment=D("0.01"), qty_increment=D("0.0001"),
    min_qty=D("0"), min_notional=D("1"), taker_fee=D("0.00005"),
)


def settings(**kw) -> Settings:
    base = dict(mode=TradingMode.BACKTEST, aggression=Aggression.MODERATE,
                live=LiveGate(enabled=False), base_currency="USD",
                enforce_cash_account=True)
    base.update(kw)
    return Settings(**base)


def quote() -> Quote:
    return Quote(SPY.key, NOW, bid=PRICE - D("0.05"), ask=PRICE + D("0.05"))


def mandate() -> Mandate:
    return Mandate(issued_at=NOW, expires_at=NOW + timedelta(hours=24),
                   permitted=frozenset({SPY.key}))


def book(weight, equity=D("10000"), price=PRICE):
    """A portfolio holding ``weight`` of its equity in SPY."""
    qty = (equity * D(str(weight))) / price
    position = Position(instrument_key=SPY.key, qty=qty, avg_cost=price,
                        last_price=price)
    state = PortfolioState(
        equity=equity, available_cash=equity - qty * price,
        positions={SPY.key: position}, base_currency="USD",
    )
    return state, qty


def exit_intent(**kw) -> TradeIntent:
    params = dict(instrument=SPY, side=Side.SELL, edge=D("0.06"),
                  confidence=D("1"), volatility=D("0.16"), quote=quote(),
                  strategy="exit")
    params.update(kw)
    return TradeIntent(**params)


class ExitSizingTests(unittest.TestCase):

    def setUp(self):
        self.engine = RiskEngine(settings=settings())
        self.cap = settings().risk.max_position_weight   # 0.15 for MODERATE

    def test_exit_at_the_position_cap_is_approved(self):
        # Pre-fix: rejected, binding_constraint == "position_headroom", qty 0.
        state, held = book(self.cap)
        decision = self.engine.evaluate(
            exit_intent(), state, mandate(), now=NOW,
        )
        self.assertTrue(
            decision.approved,
            f"exit rejected at the position cap: {decision.summary()}",
        )
        self.assertEqual(decision.qty, held)

    def test_exit_below_the_cap_sells_the_whole_position(self):
        # Pre-fix: sold only max_position_weight - existing_weight, i.e. 25% of
        # the position at a 12% weight against a 15% cap.
        state, held = book("0.12")
        decision = self.engine.evaluate(
            exit_intent(), state, mandate(), now=NOW,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.qty, held)

    def test_exit_is_full_across_the_whole_weight_range(self):
        for weight in ("0.001", "0.02", "0.05", "0.10", "0.12", "0.15", "0.20"):
            with self.subTest(weight=weight):
                state, held = book(weight)
                decision = self.engine.evaluate(
                    exit_intent(), state, mandate(), now=NOW,
                )
                self.assertTrue(decision.approved, decision.summary())
                self.assertEqual(decision.qty, held)

    def test_exit_is_labelled_a_reduction_not_an_addition(self):
        state, _ = book("0.10")
        decision = self.engine.evaluate(
            exit_intent(), state, mandate(), now=NOW,
        )
        # The held-quantity cap is the final binding constraint; either label is
        # honest, but neither may be a headroom constraint.
        self.assertNotIn(
            decision.binding_constraint,
            {"position_headroom", "gross_exposure_headroom", "kelly",
             "vol_target", "max_position_weight", "risk_budget"},
        )

    def test_exit_never_exceeds_the_position(self):
        state, held = book("0.15")
        decision = self.engine.evaluate(
            exit_intent(edge=D("10"), volatility=D("0.01")),
            state, mandate(), now=NOW,
        )
        self.assertLessEqual(decision.qty, held)

    def test_a_gross_exposure_ceiling_breach_does_not_trap_the_position(self):
        # The most dangerous case: the book is already over its gross ceiling,
        # which is precisely when it must be able to sell.
        equity = D("10000")
        qty = equity * D("2") / PRICE          # 200% gross, way over the 80% cap
        position = Position(instrument_key=SPY.key, qty=qty, avg_cost=PRICE,
                            last_price=PRICE)
        state = PortfolioState(equity=equity, available_cash=D("0"),
                               positions={SPY.key: position},
                               base_currency="USD")
        decision = self.engine.evaluate(
            exit_intent(), state, mandate(), now=NOW,
        )
        self.assertTrue(decision.approved, decision.summary())
        self.assertEqual(decision.qty, qty)

    def test_a_stop_loss_can_fire_at_the_cap(self):
        # End-to-end statement of the bug in the terms that matter.
        state, held = book(self.cap)
        stop = self.engine.stop_price(exit_intent(side=Side.BUY), PRICE)
        self.assertLess(stop, PRICE)
        decision = self.engine.evaluate(
            exit_intent(), state, mandate(), now=NOW,
        )
        self.assertGreater(
            decision.qty, D("0"),
            "the stop level is computable but the stop order is unsizeable",
        )


class ExitReachabilityUnderEntryGatesTests(unittest.TestCase):
    """Exits must not be blocked by ENTRY-only gates (lessons L1).

    The reduction-*sizing* fix above still left three entry controls -- the
    confidence floor, the daily-trade cap and the mandate's directional bias --
    checked before the reduction branch, so a stop-loss was still rejected once
    the day hit its trade cap or when an honest exit reported low conviction.
    Each test here fails against that incomplete engine and passes once the
    entry gates are skipped for reductions.
    """

    def setUp(self):
        self.engine = RiskEngine(settings=settings())
        self.profile = settings().risk

    def test_exit_approved_at_the_daily_trade_cap(self):
        # The critical case: the loop has traded up to its daily cap and a held
        # position then hits its stop. The exit must still fire.
        state, held = book("0.12")
        state.trades_today = self.profile.max_trades_per_day
        decision = self.engine.evaluate(exit_intent(), state, mandate(), now=NOW)
        self.assertTrue(
            decision.approved,
            f"exit trapped at the daily trade cap: {decision.summary()}",
        )
        self.assertEqual(decision.qty, held)

    def test_low_confidence_exit_is_still_approved(self):
        # An exit that honestly reports low conviction must not be vetoed by the
        # confidence floor; reachability cannot depend on a strategy hard-coding
        # confidence to 1.0.
        state, held = book("0.12")
        decision = self.engine.evaluate(
            exit_intent(confidence=D("0.10")), state, mandate(), now=NOW,
        )
        self.assertTrue(
            decision.approved,
            f"low-confidence exit trapped: {decision.summary()}",
        )
        self.assertEqual(decision.qty, held)

    def test_exit_survives_a_contradicting_mandate_bias(self):
        # A long-reducing SELL under a long-biased mandate is still an exit.
        state, held = book("0.12")
        biased = Mandate(
            issued_at=NOW, expires_at=NOW + timedelta(hours=24),
            permitted=frozenset({SPY.key}), directional_bias={SPY.key: D("0.9")},
        )
        decision = self.engine.evaluate(exit_intent(), state, biased, now=NOW)
        self.assertTrue(decision.approved, decision.summary())
        self.assertEqual(decision.qty, held)

    def test_entry_at_the_daily_cap_is_still_refused(self):
        # The cap must still gate ENTRIES; the fix loosens exits only.
        state = PortfolioState(
            equity=D("10000"), available_cash=D("10000"), base_currency="USD",
            trades_today=self.profile.max_trades_per_day,
        )
        entry = TradeIntent(
            instrument=SPY, side=Side.BUY, edge=D("0.05"), confidence=D("0.8"),
            volatility=D("0.20"), quote=quote(), strategy="entry",
        )
        decision = self.engine.evaluate(entry, state, mandate(), now=NOW)
        self.assertFalse(decision.approved)
        self.assertIn("daily trade limit", decision.reasons[0])


class EntrySizingUnchangedTests(unittest.TestCase):
    """The fix must not loosen entry sizing."""

    def setUp(self):
        self.engine = RiskEngine(settings=settings())

    def entry(self, **kw) -> TradeIntent:
        params = dict(instrument=SPY, side=Side.BUY, edge=D("0.05"),
                      confidence=D("0.8"), volatility=D("0.20"), quote=quote(),
                      strategy="entry")
        params.update(kw)
        return TradeIntent(**params)

    def test_position_headroom_still_caps_a_top_up(self):
        state, _ = book("0.14")
        decision = self.engine.evaluate(self.entry(), state, mandate(), now=NOW)
        weight = decision.qty * PRICE / state.equity
        self.assertLess(weight, D("0.02"))

    def test_entry_at_the_cap_is_still_refused(self):
        state, _ = book("0.15")
        decision = self.engine.evaluate(self.entry(), state, mandate(), now=NOW)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.binding_constraint, "position_headroom")

    def test_entry_from_flat_is_unaffected(self):
        state = PortfolioState(equity=D("10000"), available_cash=D("10000"),
                               base_currency="USD")
        decision = self.engine.evaluate(self.entry(), state, mandate(), now=NOW)
        self.assertTrue(decision.approved)
        weight = decision.qty * PRICE / state.equity
        self.assertLessEqual(weight, settings().risk.max_position_weight)

    def test_opening_a_short_is_still_blocked_under_a_cash_account(self):
        state = PortfolioState(equity=D("10000"), available_cash=D("10000"),
                               base_currency="USD")
        decision = self.engine.evaluate(
            exit_intent(), state, mandate(), now=NOW,
        )
        self.assertFalse(decision.approved)
        self.assertIn("short", decision.reasons[0])

    def test_confidence_floor_still_applies_to_an_entry(self):
        state = PortfolioState(equity=D("10000"), available_cash=D("10000"),
                               base_currency="USD")
        decision = self.engine.evaluate(
            self.entry(confidence=D("0.5")), state, mandate(), now=NOW,
        )
        self.assertFalse(decision.approved)


class ShortReductionTests(unittest.TestCase):
    """Closing a short must stop at flat rather than flip into a long."""

    def setUp(self):
        self.engine = RiskEngine(settings=settings(
            aggression=Aggression.AGGRESSIVE, enforce_cash_account=False,
        ))

    def test_closing_a_short_caps_at_flat(self):
        equity = D("10000")
        qty = -(equity * D("0.10")) / PRICE
        position = Position(instrument_key=SPY.key, qty=qty, avg_cost=PRICE,
                            last_price=PRICE)
        state = PortfolioState(equity=equity, available_cash=equity,
                               positions={SPY.key: position},
                               base_currency="USD")
        decision = self.engine.evaluate(
            TradeIntent(instrument=SPY, side=Side.BUY, edge=D("5"),
                        confidence=D("1"), volatility=D("0.05"), quote=quote(),
                        strategy="close_short"),
            state, mandate(), now=NOW,
        )
        self.assertTrue(decision.approved)
        self.assertEqual(decision.qty, -qty,
                         "a closing buy overshot into a long position")


if __name__ == "__main__":
    unittest.main()
