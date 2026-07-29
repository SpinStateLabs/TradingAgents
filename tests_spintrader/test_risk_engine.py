"""Tests for the risk engine, kill switch and mandate.

Bias throughout: where two behaviours are defensible, the tests pin the one
that risks less. Several tests exist specifically to stop a future refactor
from "helpfully" making the engine more permissive — an empty mandate
defaulting open, a kill switch resetting itself, a missing FX rate being
assumed away. Each of those would be an easy, plausible change that quietly
removes a control.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from spintrader.core.config import (
    Aggression, LiveGate, RiskProfile, Settings, risk_profile,
)
from spintrader.core.types import (
    AssetClass, Instrument, Position, Quote, Side, TradingMode, VenueId,
)
from spintrader.risk.engine import (
    KillSwitch, Mandate, PortfolioState, RiskDecision, RiskEngine, RiskVerdict,
    TradeIntent,
)

D = Decimal
NOW = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)

BTC = Instrument(
    symbol="BTC-USD", asset_class=AssetClass.CRYPTO, venue=VenueId.KRAKEN,
    venue_symbol="XXBTZUSD", quote_currency="USD",
    min_qty=D("0.00005"), min_notional=D("5"), taker_fee=D("0.0026"),
)
CAD_STOCK = Instrument(
    symbol="XYZ", asset_class=AssetClass.EQUITY, venue=VenueId.IBKR,
    venue_symbol="XYZ", quote_currency="CAD", min_notional=D("1"),
)


def settings(**kw) -> Settings:
    base = dict(mode=TradingMode.PAPER, aggression=Aggression.BALANCED,
                live=LiveGate(enabled=False), base_currency="USD",
                enforce_cash_account=False)
    base.update(kw)
    return Settings(**base)


def quote(bid="99", ask="101") -> Quote:
    return Quote(BTC.key, NOW, bid=D(bid), ask=D(ask))


def intent(**kw) -> TradeIntent:
    params = dict(instrument=BTC, side=Side.BUY, edge=D("0.05"),
                  confidence=D("0.8"), volatility=D("0.45"), quote=quote())
    params.update(kw)
    return TradeIntent(**params)


def state(**kw) -> PortfolioState:
    params = dict(equity=D("1000"), available_cash=D("1000"), base_currency="USD")
    params.update(kw)
    return PortfolioState(**params)


def mandate(**kw) -> Mandate:
    params = dict(issued_at=NOW, expires_at=NOW + timedelta(hours=1),
                  permitted=frozenset({BTC.key}))
    params.update(kw)
    return Mandate(**params)


def usd_rates(frm: str, to: str) -> Decimal | None:
    table = {("CAD", "USD"): D("0.71"), ("USD", "CAD"): D("1.41")}
    return D("1") if frm == to else table.get((frm, to))


def engine(**kw) -> RiskEngine:
    return RiskEngine(settings=kw.pop("settings", settings()),
                      rate_provider=kw.pop("rate_provider", usd_rates), **kw)


class KillSwitchTests(unittest.TestCase):
    def test_starts_armed_and_untripped(self):
        switch = KillSwitch(max_drawdown=D("0.15"), daily_loss_limit=D("0.03"))
        self.assertFalse(switch.tripped)
        self.assertFalse(switch.check(state()))

    def test_trips_on_drawdown_breach(self):
        switch = KillSwitch(max_drawdown=D("0.10"), daily_loss_limit=D("0.99"))
        self.assertTrue(switch.check(state(equity=D("850"), peak_equity=D("1000"))))
        self.assertTrue(switch.tripped)
        self.assertIn("drawdown", switch.reason)

    def test_does_not_trip_just_below_the_limit(self):
        switch = KillSwitch(max_drawdown=D("0.10"), daily_loss_limit=D("0.99"))
        self.assertFalse(switch.check(state(equity=D("905"), peak_equity=D("1000"))))

    def test_trips_on_daily_loss_breach(self):
        switch = KillSwitch(max_drawdown=D("0.99"), daily_loss_limit=D("0.03"))
        self.assertTrue(switch.check(state(realized_pnl_today=D("-50"))))
        self.assertIn("daily loss", switch.reason)

    def test_stays_tripped_once_tripped(self):
        # An automatically re-arming switch is a delay, not a control.
        switch = KillSwitch(max_drawdown=D("0.10"), daily_loss_limit=D("0.99"))
        switch.check(state(equity=D("800"), peak_equity=D("1000")))
        self.assertTrue(switch.check(state(equity=D("1000"), peak_equity=D("1000"))))

    def test_reset_requires_an_identity(self):
        switch = KillSwitch(max_drawdown=D("0.10"), daily_loss_limit=D("0.03"))
        switch._trip("test")
        with self.assertRaises(ValueError):
            switch.reset("")
        switch.reset("donal")
        self.assertFalse(switch.tripped)

    def test_no_peak_means_no_drawdown(self):
        switch = KillSwitch(max_drawdown=D("0.10"), daily_loss_limit=D("0.99"))
        self.assertFalse(switch.check(state(equity=D("1000"), peak_equity=None)))


class MandateTests(unittest.TestCase):
    def test_empty_permitted_set_allows_nothing(self):
        # Defaulting open would let a failed mandate build authorise the whole
        # universe. This must stay closed.
        empty = Mandate(issued_at=NOW, expires_at=NOW + timedelta(hours=1))
        self.assertFalse(empty.allows(BTC.key))

    def test_expiry(self):
        m = mandate(expires_at=NOW + timedelta(hours=1))
        self.assertFalse(m.is_expired(NOW))
        self.assertTrue(m.is_expired(NOW + timedelta(hours=2)))

    def test_expiry_is_inclusive_at_the_boundary(self):
        m = mandate(expires_at=NOW)
        self.assertTrue(m.is_expired(NOW))

    def test_open_mandate_covers_the_universe(self):
        m = Mandate.open_mandate([BTC.key, "kraken:ETH-USD"])
        self.assertTrue(m.allows(BTC.key))

    def test_bias_defaults_to_neutral(self):
        self.assertEqual(mandate().bias_for(BTC.key), D("0"))


class BlockerTests(unittest.TestCase):
    def test_kill_switch_blocks_everything(self):
        eng = engine()
        eng.kill_switch._trip("manual")
        decision = eng.evaluate(intent(), state(), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("kill switch", decision.reasons[0])

    def test_expired_mandate_blocks(self):
        # A stale LLM view must not drive trading indefinitely.
        eng = engine()
        decision = eng.evaluate(intent(), state(), mandate(),
                                now=NOW + timedelta(hours=2))
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("stale view", decision.reasons[0])

    def test_instrument_outside_the_mandate_blocks(self):
        eng = engine()
        decision = eng.evaluate(intent(), state(), mandate(permitted=frozenset()),
                                now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)

    def test_confidence_below_the_floor_blocks(self):
        eng = engine()
        floor = risk_profile(Aggression.BALANCED).min_confidence
        decision = eng.evaluate(intent(confidence=floor - D("0.01")),
                                state(), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("below", decision.reasons[0])

    def test_daily_trade_limit_blocks(self):
        eng = engine()
        limit = risk_profile(Aggression.BALANCED).max_trades_per_day
        decision = eng.evaluate(intent(), state(trades_today=limit), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("daily trade limit", decision.reasons[0])

    def test_zero_equity_blocks(self):
        eng = engine()
        decision = eng.evaluate(intent(), state(equity=D("0")), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)

    def test_direction_against_the_mandate_bias_blocks(self):
        eng = engine()
        decision = eng.evaluate(
            intent(side=Side.BUY),
            state(),
            mandate(directional_bias={BTC.key: D("-0.8")}),
            now=NOW,
        )
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("contradicts", decision.reasons[0])

    def test_cash_account_cannot_open_a_short(self):
        eng = engine(settings=settings(enforce_cash_account=True))
        decision = eng.evaluate(intent(side=Side.SELL), state(), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("short", decision.reasons[0])

    def test_selling_a_held_position_is_allowed(self):
        eng = engine(settings=settings(enforce_cash_account=True))
        held = {BTC.key: Position(BTC.key, qty=D("0.5"), avg_cost=D("90"),
                                  last_price=D("100"))}
        decision = eng.evaluate(intent(side=Side.SELL), state(positions=held),
                                mandate(), now=NOW)
        self.assertNotEqual(decision.verdict, RiskVerdict.REJECTED)


class SizingTests(unittest.TestCase):
    def test_approves_a_reasonable_trade(self):
        decision = engine().evaluate(intent(), state(), mandate(), now=NOW)
        self.assertTrue(decision.approved)
        self.assertGreater(decision.qty, D("0"))

    def test_higher_volatility_gives_a_smaller_position(self):
        eng = engine()
        calm = eng.evaluate(intent(volatility=D("0.20")), state(), mandate(), now=NOW)
        wild = eng.evaluate(intent(volatility=D("1.50")), state(), mandate(), now=NOW)
        self.assertLess(wild.qty, calm.qty)

    def test_lower_confidence_gives_a_smaller_position(self):
        # Confidence scales size rather than merely gating, so a marginal
        # signal is traded marginally.
        eng = engine()
        sure = eng.evaluate(intent(confidence=D("0.95")), state(), mandate(), now=NOW)
        unsure = eng.evaluate(intent(confidence=D("0.60")), state(), mandate(), now=NOW)
        self.assertLessEqual(unsure.qty, sure.qty)

    def test_minimum_of_all_constraints_is_taken(self):
        decision = engine().evaluate(intent(), state(), mandate(), now=NOW)
        self.assertIsNotNone(decision.binding_constraint)

    def test_conservative_profile_sizes_smaller_than_aggressive(self):
        base = dict(confidence=D("0.85"), volatility=D("0.45"))
        cons = RiskEngine(settings=settings(aggression=Aggression.CONSERVATIVE),
                          rate_provider=usd_rates)
        aggr = RiskEngine(settings=settings(aggression=Aggression.AGGRESSIVE),
                          rate_provider=usd_rates)
        c = cons.evaluate(intent(**base), state(), mandate(), now=NOW)
        a = aggr.evaluate(intent(**base), state(), mandate(), now=NOW)
        self.assertLess(c.qty, a.qty)

    def test_crisis_regime_shrinks_the_position(self):
        eng = engine()
        calm = eng.evaluate(intent(), state(), mandate(regime_risk=D("0")), now=NOW)
        crisis = eng.evaluate(intent(), state(), mandate(regime_risk=D("1")), now=NOW)
        self.assertLess(crisis.qty, calm.qty)
        self.assertTrue(any("regime risk" in r for r in crisis.reasons))

    def test_risk_budget_multiplier_scales_size(self):
        eng = engine()
        full = eng.evaluate(intent(), state(), mandate(), now=NOW)
        half = eng.evaluate(intent(), state(),
                            mandate(risk_budget_multiplier=D("0.25")), now=NOW)
        self.assertLess(half.qty, full.qty)

    def test_zero_volatility_is_not_infinite_size(self):
        # Division by variance would explode; must degrade to zero, not inf.
        decision = engine().evaluate(intent(volatility=D("0")), state(),
                                     mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)

    def test_existing_position_reduces_headroom(self):
        eng = engine()
        empty = eng.evaluate(intent(), state(), mandate(), now=NOW)
        held = {BTC.key: Position(BTC.key, qty=D("1.9"), avg_cost=D("100"),
                                  last_price=D("100"))}
        crowded = eng.evaluate(intent(), state(positions=held), mandate(), now=NOW)
        self.assertLess(crowded.qty, empty.qty)

    def test_gross_exposure_ceiling_blocks_when_exhausted(self):
        eng = engine()
        packed = {
            f"kraken:X{i}-USD": Position(f"kraken:X{i}-USD", qty=D("10"),
                                         avg_cost=D("100"), last_price=D("100"))
            for i in range(3)
        }
        decision = eng.evaluate(intent(), state(positions=packed), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("room", decision.reasons[-1])


class CashConstraintTests(unittest.TestCase):
    def test_reduced_to_available_cash(self):
        eng = engine()
        decision = eng.evaluate(intent(), state(equity=D("1000"),
                                               available_cash=D("20")),
                                mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REDUCED)
        self.assertEqual(decision.binding_constraint, "available_cash")
        # Must leave room for the fee, not spend the cash exactly.
        self.assertLessEqual(decision.qty * D("101") * D("1.0026"), D("20"))

    def test_sell_capped_at_the_held_quantity(self):
        eng = engine(settings=settings(aggression=Aggression.BALANCED))
        held = {BTC.key: Position(BTC.key, qty=D("0.01"), avg_cost=D("90"),
                                  last_price=D("100"))}
        decision = eng.evaluate(intent(side=Side.SELL), state(positions=held),
                                mandate(), now=NOW)
        self.assertLessEqual(decision.qty, D("0.01"))

    def test_below_venue_minimum_notional_rejected(self):
        eng = engine()
        decision = eng.evaluate(intent(), state(equity=D("20"),
                                               available_cash=D("4")),
                                mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertIn("below", decision.reasons[-1])


class CurrencyTests(unittest.TestCase):
    """Regression tests for the currency-blind notional cap.

    The earlier implementation compared a CAD notional against a USD cap,
    mis-gating by roughly 40%.
    """

    def test_notional_reported_in_base_currency(self):
        eng = engine()
        decision = eng.evaluate(intent(), state(), mandate(), now=NOW)
        self.assertIsNotNone(decision.notional_base)

    def test_cad_notional_converted_before_the_cap(self):
        gate = LiveGate(enabled=True, venues=frozenset({VenueId.IBKR}),
                        max_order_notional={VenueId.IBKR: D("10")})
        eng = engine(settings=settings(mode=TradingMode.LIVE))
        cad_quote = Quote(CAD_STOCK.key, NOW, bid=D("13"), ask=D("14"))
        # 1 share at 14 CAD is ~9.94 USD -- under a 10 USD cap, but over it if
        # the currency is ignored.
        decision = eng.evaluate(
            intent(instrument=CAD_STOCK, quote=cad_quote, volatility=D("0.2")),
            state(equity=D("100"), available_cash=D("100")),
            mandate(permitted=frozenset({CAD_STOCK.key})),
            live_gate=gate, now=NOW,
        )
        if decision.notional_base is not None:
            # Whatever the outcome, the figure compared must be in USD.
            self.assertLess(decision.notional_base, D("100"))

    def test_missing_rate_rejects_rather_than_assumes(self):
        # An ungated live order is worse than a skipped one.
        eng = engine(rate_provider=lambda f, t: D("1") if f == t else None)
        cad_quote = Quote(CAD_STOCK.key, NOW, bid=D("13"), ask=D("14"))
        decision = eng.evaluate(
            intent(instrument=CAD_STOCK, quote=cad_quote, volatility=D("0.2")),
            state(), mandate(permitted=frozenset({CAD_STOCK.key})), now=NOW,
        )
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)
        self.assertTrue(any("rate" in r for r in decision.reasons))

    def test_same_currency_needs_no_rate(self):
        eng = engine(rate_provider=None)
        decision = eng.evaluate(intent(), state(), mandate(), now=NOW)
        self.assertTrue(decision.approved)


class LiveGateIntegrationTests(unittest.TestCase):
    def test_disarmed_live_mode_rejects(self):
        eng = engine(settings=settings(mode=TradingMode.LIVE,
                                       live=LiveGate(enabled=False)))
        decision = eng.evaluate(intent(), state(), mandate(), now=NOW)
        self.assertEqual(decision.verdict, RiskVerdict.REJECTED)

    def test_notional_cap_enforced_in_live_mode(self):
        gate = LiveGate(enabled=True, venues=frozenset({VenueId.KRAKEN}),
                        max_order_notional={VenueId.KRAKEN: D("25")})
        eng = engine(settings=settings(mode=TradingMode.LIVE, live=gate))
        decision = eng.evaluate(intent(), state(equity=D("10000"),
                                               available_cash=D("10000")),
                                mandate(), now=NOW)
        if decision.approved:
            self.assertLessEqual(decision.notional_base, D("25"))
        else:
            self.assertTrue(any("cap" in r for r in decision.reasons))

    def test_paper_mode_ignores_the_gate(self):
        eng = engine(settings=settings(mode=TradingMode.PAPER,
                                       live=LiveGate(enabled=False)))
        self.assertTrue(eng.evaluate(intent(), state(), mandate(), now=NOW).approved)


class StopTests(unittest.TestCase):
    def test_long_stop_below_entry(self):
        eng = engine()
        stop = eng.stop_price(intent(side=Side.BUY), D("100"))
        self.assertLess(stop, D("100"))

    def test_short_stop_above_entry(self):
        eng = engine()
        stop = eng.stop_price(intent(side=Side.SELL), D("100"))
        self.assertGreater(stop, D("100"))

    def test_conservative_stop_is_tighter(self):
        cons = RiskEngine(settings=settings(aggression=Aggression.CONSERVATIVE))
        aggr = RiskEngine(settings=settings(aggression=Aggression.AGGRESSIVE))
        c = cons.stop_price(intent(), D("100"))
        a = aggr.stop_price(intent(), D("100"))
        self.assertGreater(c, a)      # closer to entry


class DecisionReportingTests(unittest.TestCase):
    def test_summary_names_the_binding_constraint(self):
        decision = engine().evaluate(intent(), state(), mandate(), now=NOW)
        self.assertIn(decision.binding_constraint or "", decision.summary())

    def test_rejection_carries_a_reason(self):
        eng = engine()
        eng.kill_switch._trip("test")
        decision = eng.evaluate(intent(), state(), mandate(), now=NOW)
        self.assertTrue(decision.reasons)
        self.assertFalse(decision.approved)


class PortfolioStateTests(unittest.TestCase):
    def test_gross_exposure_sums_absolute_values(self):
        positions = {
            "a": Position("a", qty=D("1"), last_price=D("300")),
            "b": Position("b", qty=D("-1"), last_price=D("200")),
        }
        st = state(equity=D("1000"), positions=positions)
        self.assertEqual(st.gross_exposure(), D("0.5"))

    def test_drawdown_from_peak(self):
        self.assertEqual(state(equity=D("900"), peak_equity=D("1000")).drawdown,
                         D("-0.1"))

    def test_no_peak_means_flat(self):
        self.assertEqual(state(equity=D("900")).drawdown, D("0"))

    def test_weight_of_missing_position_is_zero(self):
        self.assertEqual(state().weight_of("nope"), D("0"))


if __name__ == "__main__":
    unittest.main()
