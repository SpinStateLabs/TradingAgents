"""Risk engine: sizing, limits, and the kill switch.

Every order passes through :meth:`RiskEngine.evaluate` before it can reach a
venue. The engine is the only component permitted to decide *how much*, and it
is deliberately pessimistic at every branch -- when two readings of a situation
are defensible, it takes the smaller position.

Sizing
------
Three constraints are computed independently and the **minimum** is taken:

* **Fractional Kelly** on the agent-supplied edge. Kelly is growth-optimal only
  if the edge estimate is correct; on an LLM-estimated edge a mis-estimate at
  full Kelly is ruin, so the profile caps the fraction well below 1.
* **Volatility targeting**, which sizes so the position contributes a fixed
  share of portfolio volatility. This is what stops a 70%-vol crypto position
  and a 12%-vol equity position being treated as comparable risks.
* **Hard weight and notional caps**, which bound everything regardless.

Taking the minimum rather than blending means a single binding constraint is
always visible in the decision's audit trail.

Currency
--------
Notional caps are evaluated in the account's base currency. The earlier
implementation compared a CAD notional against a USD cap, mis-gating by ~40%.
Every comparison here converts first, and a missing rate is a *rejection*, not
an assumption -- an ungated order is worse than a skipped one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Callable, Mapping, Sequence

from spintrader.core.config import (
    LiveGate, LiveTradingDisarmed, RiskProfile, Settings,
)
from spintrader.core.types import (
    Instrument, Order, Position, Quote, Side, TradingMode, VenueId,
    to_decimal, utcnow,
)

log = logging.getLogger(__name__)

# A rate provider maps (from_currency, to_currency) -> rate, or None if unknown.
RateProvider = Callable[[str, str], Decimal | None]

ZERO = Decimal("0")
ONE = Decimal("1")


class RiskVerdict(str, Enum):
    APPROVED = "approved"
    REDUCED = "reduced"            # allowed, but smaller than requested
    REJECTED = "rejected"


@dataclass(slots=True)
class RiskDecision:
    """The engine's ruling on a proposed trade, with its reasoning.

    ``binding_constraint`` names the limit that actually determined the size.
    Without it, a system that always trades small is indistinguishable from one
    that is quietly broken.
    """
    verdict: RiskVerdict
    qty: Decimal = ZERO
    requested_qty: Decimal = ZERO
    reasons: list[str] = field(default_factory=list)
    binding_constraint: str | None = None
    notional_base: Decimal | None = None

    @property
    def approved(self) -> bool:
        return self.verdict is not RiskVerdict.REJECTED and self.qty > 0

    def reject(self, reason: str) -> "RiskDecision":
        self.verdict = RiskVerdict.REJECTED
        self.qty = ZERO
        self.reasons.append(reason)
        return self

    def summary(self) -> str:
        head = f"{self.verdict.value}: {self.qty}/{self.requested_qty}"
        if self.binding_constraint:
            head += f" (bound by {self.binding_constraint})"
        return head + ("; " + "; ".join(self.reasons) if self.reasons else "")


@dataclass(slots=True)
class PortfolioState:
    """What the risk engine needs to know about the book right now."""
    equity: Decimal
    available_cash: Decimal
    positions: Mapping[str, Position] = field(default_factory=dict)
    peak_equity: Decimal | None = None
    realized_pnl_today: Decimal = ZERO
    trades_today: int = 0
    base_currency: str = "USD"

    def gross_exposure(self, marks: Mapping[str, Decimal] | None = None) -> Decimal:
        """Sum of absolute position values, as a fraction of equity."""
        if self.equity <= 0:
            return ZERO
        marks = marks or {}
        total = ZERO
        for key, position in self.positions.items():
            price = marks.get(key, position.last_price)
            if price is not None:
                total += abs(position.qty * price)
        return total / self.equity

    def weight_of(self, instrument_key: str, mark: Decimal | None = None) -> Decimal:
        position = self.positions.get(instrument_key)
        if position is None or self.equity <= 0:
            return ZERO
        price = mark if mark is not None else position.last_price
        return ZERO if price is None else (position.qty * price) / self.equity

    @property
    def drawdown(self) -> Decimal:
        """Fractional drawdown from the peak. Zero or negative."""
        peak = self.peak_equity or self.equity
        if peak <= 0:
            return ZERO
        return (self.equity - peak) / peak


# --------------------------------------------------------------------------
# Kill switch
# --------------------------------------------------------------------------

@dataclass
class KillSwitch:
    """Trips on drawdown or daily-loss breach and refuses all new risk.

    Once tripped it stays tripped until explicitly reset by a human. Automatic
    re-arming is tempting and wrong: the conditions that trip it are exactly
    the conditions under which an unattended system should stop, and a switch
    that resets itself is a delay, not a control.
    """
    max_drawdown: Decimal
    daily_loss_limit: Decimal
    tripped: bool = False
    tripped_at: datetime | None = None
    reason: str | None = None

    def check(self, state: PortfolioState) -> bool:
        """Evaluate breach conditions. Returns True if tripped."""
        if self.tripped:
            return True

        drawdown = state.drawdown
        if drawdown < -self.max_drawdown:
            self._trip(
                f"drawdown {drawdown:.2%} breached the "
                f"{-self.max_drawdown:.2%} limit"
            )
            return True

        if state.equity > 0:
            daily = state.realized_pnl_today / state.equity
            if daily < -self.daily_loss_limit:
                self._trip(
                    f"daily loss {daily:.2%} breached the "
                    f"{-self.daily_loss_limit:.2%} limit"
                )
                return True

        return False

    def _trip(self, reason: str) -> None:
        self.tripped = True
        self.tripped_at = utcnow()
        self.reason = reason
        # Deliberately CRITICAL: this is the one log line that must not be
        # missed in a journal.
        log.critical("KILL SWITCH TRIPPED: %s", reason)

    def reset(self, acknowledged_by: str) -> None:
        """Manually re-arm. Requires naming who did it, for the audit trail."""
        if not acknowledged_by.strip():
            raise ValueError("kill switch reset requires an acknowledging identity")
        log.warning("kill switch reset by %s (was: %s)", acknowledged_by, self.reason)
        self.tripped = False
        self.tripped_at = None
        self.reason = None


# --------------------------------------------------------------------------
# Mandate (produced by the slow LLM loop, consumed by the fast loop)
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Mandate:
    """Constraints the slow LLM loop imposes on the fast quant loop.

    Mandates **expire**. A stale LLM view driving trading indefinitely is the
    failure mode this guards against: the model formed its opinion under
    conditions that no longer hold, and nothing in the fast loop would notice.
    """
    issued_at: datetime
    expires_at: datetime
    permitted: frozenset[str] = field(default_factory=frozenset)
    directional_bias: Mapping[str, Decimal] = field(default_factory=dict)  # -1..+1
    risk_budget_multiplier: Decimal = ONE
    thesis: Mapping[str, str] = field(default_factory=dict)
    regime_risk: Decimal = ZERO

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or utcnow()) >= self.expires_at

    def allows(self, instrument_key: str) -> bool:
        # An empty permitted set means "nothing permitted", not "everything".
        # Defaulting open would let a failed mandate build authorise the whole
        # universe.
        return instrument_key in self.permitted

    def bias_for(self, instrument_key: str) -> Decimal:
        return to_decimal(self.directional_bias.get(instrument_key, ZERO))

    @classmethod
    def open_mandate(cls, universe: Sequence[str], hours: int = 1) -> "Mandate":
        """A permissive mandate, for backtests and bootstrapping."""
        now = utcnow()
        return cls(
            issued_at=now,
            expires_at=now + timedelta(hours=hours),
            permitted=frozenset(universe),
        )


# --------------------------------------------------------------------------
# Risk engine
# --------------------------------------------------------------------------

@dataclass(slots=True)
class TradeIntent:
    """A proposed trade, before sizing."""
    instrument: Instrument
    side: Side
    edge: Decimal                 # expected return, as a fraction
    confidence: Decimal           # 0..1
    volatility: Decimal           # annualised, as a fraction
    quote: Quote
    strategy: str = "unknown"
    decision_id: str | None = None


class RiskEngine:
    """Sizes and vets every proposed trade."""

    def __init__(
        self,
        settings: Settings | None = None,
        kill_switch: KillSwitch | None = None,
        rate_provider: RateProvider | None = None,
    ) -> None:
        from spintrader.core.config import get_settings
        self.settings = settings or get_settings()
        profile = self.settings.risk
        self.kill_switch = kill_switch or KillSwitch(
            max_drawdown=profile.max_drawdown_limit,
            daily_loss_limit=profile.daily_loss_limit,
        )
        self._rates = rate_provider

    # -- currency ----------------------------------------------------------

    def to_base(self, amount: Decimal, currency: str, base: str) -> Decimal | None:
        """Convert to base currency. ``None`` means the rate is unknown."""
        if currency == base:
            return amount
        if self._rates is None:
            return None
        rate = self._rates(currency, base)
        return None if rate is None else amount * rate

    # -- sizing ------------------------------------------------------------

    def kelly_size(self, intent: TradeIntent, profile: RiskProfile) -> Decimal:
        """Fractional-Kelly weight from edge and volatility.

        Uses the continuous approximation f* = mu / sigma^2, scaled by the
        profile's Kelly fraction and by the agents' confidence. Confidence
        multiplies rather than gates, so a 0.6-confidence signal takes 60% of
        the size a certain one would -- gating on a threshold alone throws away
        the information that the signal was marginal.
        """
        if intent.volatility <= 0:
            return ZERO
        variance = intent.volatility * intent.volatility
        raw = intent.edge / variance
        return max(ZERO, raw * profile.kelly_fraction * intent.confidence)

    def vol_target_size(self, intent: TradeIntent, profile: RiskProfile) -> Decimal:
        """Weight that makes this position contribute the target volatility.

        Without this, a 73%-vol BTC position and a 27%-vol SPY position at the
        same weight represent wildly different risks, and the portfolio's
        realised volatility becomes whatever the most volatile holding decides.
        """
        if intent.volatility <= 0:
            return ZERO
        return profile.target_annual_vol / intent.volatility

    # -- main entry point --------------------------------------------------

    def evaluate(
        self,
        intent: TradeIntent,
        state: PortfolioState,
        mandate: Mandate,
        live_gate: LiveGate | None = None,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Size and vet a proposed trade.

        Checks run cheapest-and-most-fatal first, so a tripped kill switch
        never spends effort computing a size that will be discarded.
        """
        decision = RiskDecision(verdict=RiskVerdict.APPROVED)
        profile = self.settings.risk
        base = state.base_currency
        now = now or utcnow()

        # --- absolute blockers ------------------------------------------
        if self.kill_switch.check(state):
            return decision.reject(f"kill switch tripped: {self.kill_switch.reason}")

        if mandate.is_expired(now):
            return decision.reject(
                f"mandate expired at {mandate.expires_at:%Y-%m-%d %H:%M}; "
                f"refusing to trade on a stale view"
            )

        if not mandate.allows(intent.instrument.key):
            return decision.reject(
                f"{intent.instrument.symbol} is not in the current mandate"
            )

        if state.equity <= 0:
            return decision.reject("no equity")

        if intent.confidence < profile.min_confidence:
            return decision.reject(
                f"confidence {intent.confidence:.2f} below the "
                f"{profile.min_confidence:.2f} floor for "
                f"{profile.aggression.value}"
            )

        if state.trades_today >= profile.max_trades_per_day:
            return decision.reject(
                f"daily trade limit reached ({state.trades_today}/"
                f"{profile.max_trades_per_day})"
            )

        # Direction must agree with the mandate's bias when one is expressed.
        bias = mandate.bias_for(intent.instrument.key)
        if bias != ZERO:
            wants_long = intent.side is Side.BUY
            if (bias > ZERO) != wants_long:
                return decision.reject(
                    f"{intent.side.value} contradicts the mandate's "
                    f"{'long' if bias > 0 else 'short'} bias"
                )

        # --- shorting -----------------------------------------------------
        held = state.positions.get(intent.instrument.key)
        held_qty = held.qty if held else ZERO
        if intent.side is Side.SELL and held_qty <= ZERO:
            if not profile.allow_shorts or self.settings.enforce_cash_account:
                return decision.reject(
                    "opening a short is not permitted "
                    f"({'cash account' if self.settings.enforce_cash_account else profile.aggression.value})"
                )

        # --- regime-scaled profile ---------------------------------------
        scaled = profile.scaled_for_regime(mandate.regime_risk)
        if mandate.regime_risk > ZERO:
            decision.reasons.append(
                f"regime risk {mandate.regime_risk:.2f} scaled exposure to "
                f"{scaled.max_position_weight / profile.max_position_weight:.0%}"
            )

        # --- candidate sizes ----------------------------------------------
        candidates: dict[str, Decimal] = {
            "kelly": self.kelly_size(intent, scaled),
            "vol_target": self.vol_target_size(intent, scaled),
            "max_position_weight": scaled.max_position_weight,
        }
        candidates["risk_budget"] = (
            candidates["max_position_weight"] * mandate.risk_budget_multiplier
        )

        # Room left under the gross exposure ceiling.
        current_gross = state.gross_exposure()
        headroom = scaled.max_gross_exposure - current_gross
        candidates["gross_exposure_headroom"] = max(ZERO, headroom)

        # Room left in this specific name.
        existing_weight = abs(state.weight_of(intent.instrument.key, intent.quote.mid))
        candidates["position_headroom"] = max(
            ZERO, scaled.max_position_weight - existing_weight
        )

        binding = min(candidates, key=lambda k: candidates[k])
        target_weight = candidates[binding]
        decision.binding_constraint = binding

        if target_weight <= ZERO:
            return decision.reject(
                f"no room to add risk ({binding} is exhausted; "
                f"gross exposure {current_gross:.0%} of a "
                f"{scaled.max_gross_exposure:.0%} ceiling)"
            )

        # --- weight -> quantity -------------------------------------------
        price = intent.quote.ask if intent.side is Side.BUY else intent.quote.bid
        if price <= 0:
            return decision.reject(f"no usable price for {intent.instrument.symbol}")

        target_notional_base = state.equity * target_weight
        # Convert the base-currency budget into the instrument's quote currency.
        quote_ccy = intent.instrument.quote_currency
        if quote_ccy == base:
            target_notional_quote = target_notional_base
        else:
            rate = self._rates(base, quote_ccy) if self._rates else None
            if rate is None:
                return decision.reject(
                    f"cannot size a {quote_ccy}-quoted instrument without a "
                    f"{base}/{quote_ccy} rate"
                )
            target_notional_quote = target_notional_base * rate

        qty = target_notional_quote / price
        decision.requested_qty = qty

        # --- cash constraint ---------------------------------------------
        if intent.side is Side.BUY:
            fee_rate = intent.instrument.taker_fee
            affordable_notional = state.available_cash / (ONE + fee_rate)
            affordable_qty = affordable_notional / price if price > 0 else ZERO
            if affordable_qty < qty:
                qty = affordable_qty
                decision.binding_constraint = "available_cash"
                decision.reasons.append(
                    f"reduced to fit {state.available_cash:.2f} of settled cash"
                )
        else:
            # Never sell more than is held; the cash account cannot short.
            if held_qty > ZERO and qty > held_qty:
                qty = held_qty
                decision.binding_constraint = "position_size"
                decision.reasons.append("reduced to the held position")

        if qty <= ZERO:
            return decision.reject("sized to zero after cash and position limits")

        # --- venue minimums ----------------------------------------------
        notional_quote = qty * price
        if intent.instrument.min_notional > ZERO and notional_quote < intent.instrument.min_notional:
            return decision.reject(
                f"notional {notional_quote:.2f} below the "
                f"{intent.instrument.min_notional} minimum for "
                f"{intent.instrument.symbol}"
            )
        if qty < intent.instrument.min_qty:
            return decision.reject(
                f"quantity {qty} below the {intent.instrument.min_qty} minimum"
            )

        # --- live notional cap, in BASE currency -------------------------
        notional_base = self.to_base(notional_quote, quote_ccy, base)
        if notional_base is None:
            # Refusing beats assuming: an ungated live order is worse than a
            # skipped one.
            return decision.reject(
                f"cannot convert {quote_ccy} notional to {base}; refusing to "
                f"bypass the notional cap"
            )
        decision.notional_base = notional_base

        gate = live_gate or self.settings.live
        try:
            gate.check(intent.instrument.venue, self.settings.mode, notional_base)
        except LiveTradingDisarmed as exc:
            return decision.reject(str(exc))

        decision.qty = qty
        decision.verdict = (
            RiskVerdict.REDUCED if qty < decision.requested_qty else RiskVerdict.APPROVED
        )
        return decision

    # -- stops -------------------------------------------------------------

    def stop_price(self, intent: TradeIntent, entry: Decimal) -> Decimal:
        """Hard stop from the profile's per-position loss tolerance."""
        profile = self.settings.risk
        offset = entry * profile.stop_loss_pct
        return entry - offset if intent.side is Side.BUY else entry + offset


__all__ = [
    "KillSwitch", "Mandate", "PortfolioState", "RateProvider", "RiskDecision",
    "RiskEngine", "RiskVerdict", "TradeIntent",
]
