"""The portfolio ledger: positions, cash, P&L and reconciliation.

This module is the system's authoritative record of what it owns and what it
has earned. Everything downstream -- risk sizing, the scorecard, the
self-improvement loop's attribution -- trusts these numbers, so the failure
modes here are the expensive ones.

Four properties are enforced rather than assumed.

**Fills are idempotent.** Venues re-deliver fills: on WebSocket reconnect, on
REST reconciliation, on restart replaying a journal. Applying the same fill
twice double-counts a position and manufactures P&L from nothing, and it does
so silently. Every fill id is recorded and re-application is a no-op.

**Cash is per-currency and settlement-aware.** The book holds USD at Kraken and
CAD at IBKR. Summing them as though they were the same unit is a ~40% error.
Sale proceeds in a cash account are also unavailable until T+1, so settled and
unsettled cash are tracked separately -- spending unsettled proceeds is a
good-faith violation.

**Equity conversion never silently assumes.** A missing FX rate makes equity
unknowable, and reporting a smaller number as though it were complete would
make the risk engine size against a fiction. Conversion failures are surfaced.

**Fees are a first-class term.** They are tracked apart from trading P&L,
because at minute cadence fees are frequently the largest single line and
burying them inside net P&L hides the reason a strategy loses money.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from spintrader.core.types import (
    Balance, Fill, Order, Position, Side, TradingMode, VenueId, to_decimal,
    utcnow,
)
from spintrader.risk.engine import PortfolioState

log = logging.getLogger(__name__)

ZERO = Decimal("0")


class LedgerError(RuntimeError):
    """The ledger was asked to do something inconsistent."""


class UnknownRate(LedgerError):
    """A currency conversion is required but no rate is available."""


# --------------------------------------------------------------------------
# Cash
# --------------------------------------------------------------------------

@dataclass(slots=True)
class SettlementBucket:
    """Proceeds that become spendable at ``available_at``."""
    amount: Decimal
    available_at: datetime
    source_fill_id: str | None = None


@dataclass
class CashAccount:
    """Per-currency cash with T+N settlement tracking."""
    currency: str
    settled: Decimal = ZERO
    pending: list[SettlementBucket] = field(default_factory=list)

    @property
    def unsettled(self) -> Decimal:
        return sum((b.amount for b in self.pending), ZERO)

    @property
    def total(self) -> Decimal:
        return self.settled + self.unsettled

    @property
    def available(self) -> Decimal:
        """Spendable right now. This is what sizing must use."""
        return self.settled

    def credit(self, amount: Decimal, available_at: datetime | None = None,
               fill_id: str | None = None) -> None:
        if available_at is None:
            self.settled += amount
        else:
            self.pending.append(SettlementBucket(amount, available_at, fill_id))

    def debit(self, amount: Decimal) -> None:
        """Reduce settled cash. Allowed to go negative.

        A negative balance is a real state -- fees can be charged against an
        empty account -- and silently clamping it to zero would hide a
        reconciliation break rather than surface it.
        """
        self.settled -= amount

    def release_settled(self, now: datetime) -> Decimal:
        """Move matured buckets into settled cash. Returns the amount released."""
        released = ZERO
        still_pending: list[SettlementBucket] = []
        for bucket in self.pending:
            if bucket.available_at <= now:
                self.settled += bucket.amount
                released += bucket.amount
            else:
                still_pending.append(bucket)
        self.pending = still_pending
        return released

    def to_balance(self, venue: VenueId, ts: datetime) -> Balance:
        return Balance(currency=self.currency, total=self.total,
                       available=self.available, venue=venue, ts=ts)


# --------------------------------------------------------------------------
# Equity
# --------------------------------------------------------------------------

@dataclass(slots=True)
class EquitySnapshot:
    """A point-in-time valuation of the whole book.

    ``complete`` is False when something could not be valued -- a missing mark
    or a missing FX rate. A partial equity figure reported as though it were
    total is how a risk engine ends up sizing against a book bigger than it
    thinks, so the flag travels with the number.
    """
    ts: datetime
    base_currency: str
    cash: Decimal
    positions_value: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal
    gross_exposure: Decimal
    complete: bool = True
    unpriced: tuple[str, ...] = ()
    unconvertible: tuple[str, ...] = ()

    @property
    def net_pnl(self) -> Decimal:
        """Realized plus unrealized, net of fees."""
        return self.realized_pnl + self.unrealized_pnl - self.fees_paid


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------

class Ledger:
    """Authoritative position, cash and P&L record."""

    def __init__(
        self,
        base_currency: str = "USD",
        mode: TradingMode = TradingMode.PAPER,
        settlement_days: int = 0,
        opening_cash: Mapping[str, Decimal] | None = None,
    ) -> None:
        self.base_currency = base_currency.upper()
        self.mode = mode
        self.settlement_days = settlement_days
        self.positions: dict[str, Position] = {}
        self.cash: dict[str, CashAccount] = {}
        self.fees_by_currency: dict[str, Decimal] = {}
        self._applied_fills: set[str] = set()
        self._fill_count = 0
        self._peak_equity: Decimal | None = None
        self._realized_today: Decimal = ZERO
        self._trades_today: int = 0
        self._today: datetime | None = None

        for currency, amount in (opening_cash or {}).items():
            self.account(currency).credit(to_decimal(amount))

    # -- accessors ---------------------------------------------------------

    def account(self, currency: str) -> CashAccount:
        currency = currency.upper()
        if currency not in self.cash:
            self.cash[currency] = CashAccount(currency)
        return self.cash[currency]

    def position(self, instrument_key: str) -> Position:
        if instrument_key not in self.positions:
            self.positions[instrument_key] = Position(instrument_key)
        return self.positions[instrument_key]

    @property
    def realized_pnl(self) -> Decimal:
        """Total realized P&L, in instrument quote currencies.

        Summed unconverted. Correct only for a single-currency book; use
        :meth:`value` for a converted figure.
        """
        return sum((p.realized_pnl for p in self.positions.values()), ZERO)

    @property
    def fees_paid(self) -> Decimal:
        return sum(self.fees_by_currency.values(), ZERO)

    @property
    def fill_count(self) -> int:
        return self._fill_count

    # -- fills -------------------------------------------------------------

    def apply_fill(
        self,
        fill: Fill,
        quote_currency: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Apply a fill. Returns False if it was already applied.

        Idempotency is the point. Venues re-deliver fills on reconnect, on
        reconciliation and on restart, and applying one twice inflates the
        position and invents P&L without raising anything.
        """
        if fill.fill_id in self._applied_fills:
            log.debug("ignoring duplicate fill %s", fill.fill_id)
            return False

        currency = (quote_currency or fill.fee_currency or self.base_currency).upper()
        now = now or fill.ts
        self._roll_day(now)

        position = self.position(fill.instrument_key)
        realized = position.apply_fill(fill)

        cash = self.account(currency)
        gross = fill.qty * fill.price

        if fill.side is Side.BUY:
            # Purchases settle immediately against settled cash.
            cash.debit(gross)
        else:
            # Sale proceeds are unavailable until T+N in a cash account.
            available_at = (
                now + timedelta(days=self.settlement_days)
                if self.settlement_days > 0 else None
            )
            cash.credit(gross, available_at=available_at, fill_id=fill.fill_id)

        # Fees always come out of settled cash immediately, regardless of
        # settlement: the venue does not wait.
        if fill.fee:
            fee_currency = (fill.fee_currency or currency).upper()
            self.account(fee_currency).debit(fill.fee)
            self.fees_by_currency[fee_currency] = (
                self.fees_by_currency.get(fee_currency, ZERO) + fill.fee
            )

        self._applied_fills.add(fill.fill_id)
        self._fill_count += 1
        self._realized_today += realized
        self._trades_today += 1
        return True

    def apply_fills(self, fills: Iterable[Fill], **kw) -> int:
        """Apply many fills, skipping duplicates. Returns the number applied."""
        return sum(1 for fill in fills if self.apply_fill(fill, **kw))

    def _roll_day(self, now: datetime) -> None:
        """Reset per-day counters when the UTC date changes."""
        today = now.date()
        if self._today is None or self._today.date() != today:
            self._today = now
            self._realized_today = ZERO
            self._trades_today = 0

    # -- settlement --------------------------------------------------------

    def settle(self, now: datetime | None = None) -> Decimal:
        """Release matured proceeds across all currencies."""
        now = now or utcnow()
        return sum((acct.release_settled(now) for acct in self.cash.values()), ZERO)

    # -- valuation ---------------------------------------------------------

    def mark(self, prices: Mapping[str, Decimal]) -> None:
        """Update last prices on positions."""
        for key, price in prices.items():
            if key in self.positions:
                self.positions[key].last_price = to_decimal(price)

    def value(
        self,
        prices: Mapping[str, Decimal] | None = None,
        rates: Mapping[tuple[str, str], Decimal] | None = None,
        quote_currencies: Mapping[str, str] | None = None,
        now: datetime | None = None,
    ) -> EquitySnapshot:
        """Value the book in base currency.

        ``rates`` maps (from, to) to a multiplier. ``quote_currencies`` maps an
        instrument key to the currency its price is denominated in; anything
        absent is assumed to be base-denominated, which is correct for the
        crypto book and explicit for the CAD equity book.
        """
        now = now or utcnow()
        prices = prices or {}
        rates = rates or {}
        quote_currencies = quote_currencies or {}

        def convert(amount: Decimal, currency: str) -> Decimal | None:
            currency = currency.upper()
            if currency == self.base_currency or amount == ZERO:
                return amount
            rate = rates.get((currency, self.base_currency))
            if rate is not None:
                return amount * rate
            inverse = rates.get((self.base_currency, currency))
            if inverse and inverse != ZERO:
                return amount / inverse
            return None

        unconvertible: list[str] = []
        unpriced: list[str] = []
        complete = True

        # --- cash ---
        cash_total = ZERO
        for currency, account in self.cash.items():
            converted = convert(account.total, currency)
            if converted is None:
                unconvertible.append(currency)
                complete = False
                continue
            cash_total += converted

        # --- positions ---
        positions_value = ZERO
        unrealized = ZERO
        realized = ZERO
        gross = ZERO

        for key, position in self.positions.items():
            currency = quote_currencies.get(key, self.base_currency)

            realized_converted = convert(position.realized_pnl, currency)
            if realized_converted is None:
                unconvertible.append(currency)
                complete = False
            else:
                realized += realized_converted

            if position.is_flat:
                continue

            price = prices.get(key, position.last_price)
            if price is None:
                unpriced.append(key)
                complete = False
                continue

            price = to_decimal(price)
            market_value = convert(position.qty * price, currency)
            pnl = convert(position.unrealized_pnl(price), currency)
            if market_value is None or pnl is None:
                unconvertible.append(currency)
                complete = False
                continue

            positions_value += market_value
            unrealized += pnl
            gross += abs(market_value)

        # --- fees ---
        fees = ZERO
        for currency, amount in self.fees_by_currency.items():
            converted = convert(amount, currency)
            if converted is None:
                unconvertible.append(currency)
                complete = False
                continue
            fees += converted

        equity = cash_total + positions_value
        if complete:
            self._peak_equity = (
                equity if self._peak_equity is None else max(self._peak_equity, equity)
            )

        return EquitySnapshot(
            ts=now,
            base_currency=self.base_currency,
            cash=cash_total,
            positions_value=positions_value,
            equity=equity,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            fees_paid=fees,
            gross_exposure=(gross / equity) if equity > ZERO else ZERO,
            complete=complete,
            unpriced=tuple(sorted(set(unpriced))),
            unconvertible=tuple(sorted(set(unconvertible))),
        )

    # -- risk engine interface --------------------------------------------

    def to_risk_state(
        self,
        prices: Mapping[str, Decimal] | None = None,
        rates: Mapping[tuple[str, str], Decimal] | None = None,
        quote_currencies: Mapping[str, str] | None = None,
        now: datetime | None = None,
    ) -> PortfolioState:
        """Build the state the risk engine consumes.

        Raises when equity is incomplete. Handing the risk engine a partial
        equity figure would let it size against a book it cannot see, which is
        worse than refusing to trade this cycle.
        """
        snapshot = self.value(prices, rates, quote_currencies, now)
        if not snapshot.complete:
            raise UnknownRate(
                f"cannot value the book completely: unpriced={snapshot.unpriced} "
                f"unconvertible={snapshot.unconvertible}; refusing to size "
                f"against a partial view"
            )

        available = self.account(self.base_currency).available
        return PortfolioState(
            equity=snapshot.equity,
            available_cash=available,
            positions=dict(self.positions),
            peak_equity=self._peak_equity,
            realized_pnl_today=self._realized_today,
            trades_today=self._trades_today,
            base_currency=self.base_currency,
        )

    # -- reconciliation ----------------------------------------------------

    def reconcile_positions(
        self,
        venue_positions: Mapping[str, Position],
        tolerance: Decimal = ZERO,
    ) -> list[str]:
        """Compare our positions against the venue's.

        Silent divergence is the failure that matters: trading against a
        position you do not hold produces losses that look like strategy
        failure, and the real cause is invisible.
        """
        divergences: list[str] = []
        keys = set(self.positions) | set(venue_positions)
        for key in sorted(keys):
            ours = self.positions[key].qty if key in self.positions else ZERO
            theirs = venue_positions[key].qty if key in venue_positions else ZERO
            if abs(ours - theirs) > tolerance:
                divergences.append(
                    f"{key}: ledger {ours}, venue {theirs} (diff {ours - theirs})"
                )
        return divergences

    def reconcile_cash(
        self,
        venue_balances: Mapping[str, Balance],
        tolerance: Decimal = Decimal("0.01"),
    ) -> list[str]:
        """Compare cash against the venue, per currency.

        The tolerance defaults to one cent rather than zero: venues round fees
        and interest in ways a ledger cannot reproduce exactly, and flagging
        sub-cent noise as a break trains you to ignore the alert.
        """
        divergences: list[str] = []
        currencies = set(self.cash) | set(venue_balances)
        for currency in sorted(currencies):
            ours = self.cash[currency].total if currency in self.cash else ZERO
            theirs = venue_balances[currency].total if currency in venue_balances else ZERO
            if abs(ours - theirs) > tolerance:
                divergences.append(
                    f"{currency}: ledger {ours:.8f}, venue {theirs:.8f} "
                    f"(diff {ours - theirs:.8f})"
                )
        return divergences

    def adopt_venue_state(
        self,
        venue_positions: Mapping[str, Position],
        venue_balances: Mapping[str, Balance],
    ) -> list[str]:
        """Overwrite the ledger with the venue's view, returning what changed.

        Used at startup and after an unexplained divergence. The venue is
        authoritative about what is actually held -- our ledger is a model of
        it -- but note this DISCARDS cost basis the venue does not report,
        which is why it returns a change list for logging rather than doing it
        quietly.
        """
        changes = self.reconcile_positions(venue_positions) + \
            self.reconcile_cash(venue_balances)

        for key, position in venue_positions.items():
            existing = self.positions.get(key)
            adopted = Position(
                instrument_key=key,
                qty=position.qty,
                # Preserve our basis when the venue does not supply one;
                # Kraken reports quantity only.
                avg_cost=position.avg_cost or (existing.avg_cost if existing else ZERO),
                realized_pnl=existing.realized_pnl if existing else ZERO,
                fees_paid=existing.fees_paid if existing else ZERO,
                last_price=position.last_price or (existing.last_price if existing else None),
            )
            self.positions[key] = adopted

        for key in list(self.positions):
            if key not in venue_positions:
                self.positions[key].qty = ZERO

        for currency, balance in venue_balances.items():
            account = self.account(currency)
            account.settled = balance.available
            held = balance.total - balance.available
            account.pending = (
                [SettlementBucket(held, utcnow() + timedelta(days=1))]
                if held > ZERO else []
            )

        return changes


__all__ = [
    "CashAccount", "EquitySnapshot", "Ledger", "LedgerError",
    "SettlementBucket", "UnknownRate",
]
