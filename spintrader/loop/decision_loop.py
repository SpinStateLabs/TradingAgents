"""The two-tier decision loop -- the piece that makes the rest actually run.

Everything before this built components; this wires them into a running system.

Two tiers, at two cadences:

* **Fast loop, every minute, no LLM.** For each instrument it reads the trailing
  1-minute bars, asks a deterministic quant strategy for intents, and routes
  each intent through the *same* :class:`RiskEngine`, venue and :class:`Ledger`
  that the backtester replays through. That sameness is the point: a live path
  that differs from the backtested one measures a different program.

* **Slow loop, hourly, LLM.** It deliberates over the persona panel and emits a
  :class:`Mandate` -- which instruments may be opened, in which direction, under
  what regime -- and the fast loop obeys it until it expires.

Two safety rules are load-bearing here:

* **Exits are always reachable.** A reduce/close intent is evaluated against a
  fresh close-only mandate, so a held position can be exited even when the real
  mandate has expired or no longer permits the instrument. A stale macro view
  must be able to stop driving *new* risk without trapping risk already on. (It
  does not override the kill switch: a tripped kill switch halts everything
  pending a human reset, by design.)
* **The loop cannot arm live trading.** It runs in the configured mode -- paper
  by default -- and every order still passes the live gate inside
  ``Venue.submit``. Arming live is three switches a human flips, never this code.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Callable, Mapping, Sequence

from spintrader.agents.personas.spec import Horizon
from spintrader.core.config import Settings, get_settings
from spintrader.core.types import (
    Action, Bar, Decision, Instrument, Order, OrderType, Quote, Side,
    TimeInForce, TradingMode, to_decimal, utcnow,
)
from spintrader.loop.context import gather_contexts
from spintrader.loop.mandate import Deliberation, MandateService
from spintrader.portfolio.ledger import Ledger
from spintrader.risk.engine import Mandate, RiskDecision, RiskEngine
from spintrader.venues.base import Venue
from spintrader.venues.paper import PaperVenue, SlippageModel

log = logging.getLogger(__name__)

ZERO = Decimal("0")


class ExecutionStyle(str, Enum):
    """How the loop turns an approved intent into an order.

    ``TAKER`` sends a market order: immediate, certain, pays the taker fee.
    ``MAKER_FIRST`` posts a passive limit at the touch to earn the maker fee
    (0.16% vs 0.26% on Kraken -- decisive at minute cadence), and escalates to a
    taker market order if it has not filled within ``maker_timeout_ticks``.

    Two deliberate rules: exits are ALWAYS taker regardless of style -- a
    stop-loss that waits in the queue while the loss grows is the wrong trade --
    and backtests stay taker, so a strategy is promoted against the harder cost
    and live maker fills are a bonus, never a dependency.
    """
    TAKER = "taker"
    MAKER_FIRST = "maker_first"


# --------------------------------------------------------------------------
# Live cursor
# --------------------------------------------------------------------------

class LiveCursor:
    """The replay cursor's live twin: trailing history plus the current quote.

    A :class:`~spintrader.backtest.engine.Strategy` only ever calls
    ``history()`` and ``quote()``, so this exposes exactly those and nothing
    more, positioned at the most recent bar. The strategy therefore cannot tell
    whether it is running in a backtest or live -- which is what lets the same
    persona code drive both.

    The quote is supplied by the loop (from the live venue, or synthesised from
    the last bar in paper), not derived here, so sizing and filling agree on one
    price rather than two.
    """

    def __init__(self, bars: Sequence[Bar], quote: Quote) -> None:
        if not bars:
            raise ValueError("LiveCursor needs at least one bar")
        self._bars = list(bars)
        self._index = len(self._bars) - 1
        self._quote = quote

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def index(self) -> int:
        return self._index

    @property
    def current(self) -> Bar:
        return self._bars[self._index]

    @property
    def now(self) -> datetime:
        return self.current.ts

    def history(self, lookback: int | None = None) -> list[Bar]:
        end = self._index + 1
        start = 0 if lookback is None else max(0, end - lookback)
        return self._bars[start:end]

    def quote(self, spread_bps: Decimal = Decimal("5")) -> Quote:
        # The loop already chose the market price; spread_bps is accepted for
        # protocol compatibility but the supplied quote wins, so sizing uses the
        # exact price the venue will fill against.
        return self._quote


# --------------------------------------------------------------------------
# Execution result
# --------------------------------------------------------------------------

@dataclass(slots=True)
class ExecutionResult:
    """What happened to one intent, for tests, logs and the audit trail."""
    instrument_key: str
    side: Side
    reducing: bool
    risk: RiskDecision
    decision: Decision
    filled_qty: Decimal = ZERO
    submitted: bool = False
    error: str | None = None

    @property
    def approved(self) -> bool:
        return self.risk.approved


# --------------------------------------------------------------------------
# Decision loop
# --------------------------------------------------------------------------

def _synth_quote(bar: Bar, spread_bps: Decimal) -> Quote:
    """Synthesise a top-of-book from a bar close, matching the replay cursor.

    1-minute bars carry no bid/ask, so a spread is imposed. The half-spread is
    ``close * bps / 20000`` -- identical to ``ReplayCursor.quote`` -- so a paper
    loop and a backtest price a fill the same way.
    """
    half = bar.close * spread_bps / Decimal(20_000)
    return Quote(instrument_key=bar.instrument_key, ts=bar.ts,
                 bid=bar.close - half, ask=bar.close + half)


class DecisionLoop:
    """Fast quant execution governed by a slow, expiring LLM mandate."""

    def __init__(
        self,
        settings: Settings,
        store,
        venue: Venue,
        ledger: Ledger,
        risk: RiskEngine,
        strategies: Mapping[str, object],
        mandate_service: MandateService,
        instruments: Sequence[Instrument],
        *,
        interval: str = "1m",
        spread_bps: Decimal = Decimal("3"),
        fast_lookback: int = 250,
        context_lookback: int = 1_000,
        horizon: Horizon = Horizon.INTRADAY,
        continuous: bool = True,
        with_regime: bool = False,
        quote_provider: Callable[[Instrument, Bar], Quote] | None = None,
        data_keys: Mapping[str, str] | None = None,
        execution: ExecutionStyle = ExecutionStyle.TAKER,
        maker_timeout_ticks: int = 3,
    ) -> None:
        self.settings = settings
        self.store = store
        self.venue = venue
        self.ledger = ledger
        self.risk = risk
        self.strategies = dict(strategies)
        self.mandate_service = mandate_service
        self.instruments = list(instruments)
        self.interval = interval
        # Bars are stored under the ingesting venue's key (kraken:BTC-USD) while
        # the loop trades a PAPER twin (paper:BTC-USD). This maps a trading
        # instrument's key to the key its bars live under; identity when unset.
        self.data_keys = dict(data_keys or {})
        self.spread_bps = to_decimal(spread_bps)
        self.fast_lookback = fast_lookback
        self.context_lookback = context_lookback
        self.horizon = horizon
        self.continuous = continuous
        self.with_regime = with_regime
        self._quote_provider = quote_provider
        self.execution = execution
        self.maker_timeout_ticks = maker_timeout_ticks
        self._current_quotes: dict[str, Quote] = {}
        self._deliberation: Deliberation | None = None
        self._running = False
        self._tick = 0
        # instrument key -> the resting maker order working for it, and the tick
        # it was posted on, so a stale one can be escalated to a taker fill.
        self._working: dict[str, Order] = {}
        self._posted_tick: dict[str, int] = {}

    # -- market data helpers ----------------------------------------------

    def market_quote(self, instrument: Instrument) -> Quote:
        """Quote source for the paper venue: the loop's current-tick price."""
        quote = self._current_quotes.get(instrument.key)
        if quote is None:
            raise KeyError(f"no current quote for {instrument.key}")
        return quote

    def _quote_for(self, instrument: Instrument, bar: Bar) -> Quote:
        if self._quote_provider is not None:
            return self._quote_provider(instrument, bar)
        return _synth_quote(bar, self.spread_bps)

    # -- slow loop --------------------------------------------------------

    def refresh_mandate(self, now: datetime | None = None) -> Mandate:
        """Run a deliberation cycle and adopt the new mandate."""
        contexts = gather_contexts(
            self.store, self.instruments, self.interval, self.horizon,
            lookback=self.context_lookback, continuous=self.continuous,
            with_regime=self.with_regime, data_keys=self.data_keys,
        )
        self._deliberation = self.mandate_service.deliberate(contexts, now=now)
        return self._deliberation.mandate

    @property
    def deliberation(self) -> Deliberation | None:
        return self._deliberation

    # -- fast loop --------------------------------------------------------

    def fast_tick(
        self, mandate: Mandate, now: datetime | None = None,
    ) -> list[ExecutionResult]:
        """One minute-cadence pass over the universe."""
        now = now or utcnow()
        self._tick += 1

        # 1. Refresh every quote first, so the book can be fully valued before
        #    any single trade is sized. to_risk_state refuses a partial view, so
        #    a held instrument that is missing a mark would block the whole tick;
        #    pricing everything up front avoids that.
        bars_by_key: dict[str, list[Bar]] = {}
        marks: dict[str, Decimal] = {}
        for instrument in self.instruments:
            data_key = self.data_keys.get(instrument.key, instrument.key)
            raw = self.store.read_bars(data_key, self.interval, limit=self.fast_lookback)
            # Re-key the bars onto the trading instrument so the cursor, quotes,
            # marks, orders and ledger all agree on one instrument key.
            bars = (
                [replace(b, instrument_key=instrument.key) for b in raw]
                if data_key != instrument.key else raw
            )
            bars_by_key[instrument.key] = bars
            if bars:
                quote = self._quote_for(instrument, bars[-1])
                self._current_quotes[instrument.key] = quote
                marks[instrument.key] = quote.mid

        # Advance the (paper) venue clock and settle matured cash, then mark.
        # set_time fills any resting maker limits the market traded through, so
        # sync those into the ledger and retire filled/stale working orders
        # before sizing anything new.
        if isinstance(self.venue, PaperVenue):
            self.venue.set_time(now)
        self.ledger.settle(now)
        self.ledger.mark(marks)
        self._sync_fills(now)
        self._reconcile_working(now)
        self._sync_fills(now)

        # 2. Ask each strategy for intents and route them.
        results: list[ExecutionResult] = []
        for instrument in self.instruments:
            bars = bars_by_key[instrument.key]
            strategy = self.strategies.get(instrument.key)
            if strategy is None or not bars:
                continue
            warmup = int(getattr(strategy, "warmup_bars", 0) or 0)
            if len(bars) < warmup:
                continue

            cursor = LiveCursor(bars, self._current_quotes[instrument.key])
            try:
                intents = strategy.on_bar(cursor, instrument, mandate)
            except Exception as exc:                    # noqa: BLE001 - recorded
                log.warning("strategy for %s raised: %s", instrument.key, exc)
                continue

            for intent in intents or ():
                result = self._execute(intent, mandate, marks, now)
                if result is not None:
                    results.append(result)
        return results

    def _close_only_mandate(self, instrument_key: str, now: datetime) -> Mandate:
        """A fresh mandate that permits only exiting this one instrument.

        Issued at ``now`` so it is never expired at evaluation, permitting the
        held instrument with no directional bias and no regime damping -- an exit
        is a risk *reduction* and must not be sized down by a crisis regime it is
        responding to. This is how 'exits are always reachable' is implemented
        without touching the risk engine or its documented mandate semantics.
        """
        return Mandate(
            issued_at=now,
            expires_at=now + timedelta(hours=1),
            permitted=frozenset({instrument_key}),
        )

    def _execute(
        self, intent, mandate: Mandate, marks: Mapping[str, Decimal], now: datetime,
    ) -> ExecutionResult | None:
        key = intent.instrument.key

        try:
            state = self.ledger.to_risk_state(prices=marks, now=now)
        except Exception as exc:                        # noqa: BLE001 - recorded
            log.debug("skipping %s intent: book not fully valuable (%s)", key, exc)
            return None

        held_qty = state.positions[key].qty if key in state.positions else ZERO
        reducing = (
            (intent.side is Side.SELL and held_qty > ZERO)
            or (intent.side is Side.BUY and held_qty < ZERO)
        )
        # Exits always reachable: a reduction is judged against a fresh
        # close-only mandate, so a stale or restrictive real mandate cannot trap
        # a position. New risk still answers to the real mandate.
        effective = self._close_only_mandate(key, now) if reducing else mandate

        risk_decision = self.risk.evaluate(intent, state, effective, now=now)

        action = (
            Action.CLOSE if reducing
            else (Action.BUY if intent.side is Side.BUY else Action.SELL)
        )
        decision = Decision(
            instrument_key=key,
            action=action,
            confidence=intent.confidence,
            ts=now,
            horizon=self.horizon.value,
            regime=(f"{mandate.regime_risk:.2f}" if mandate.regime_risk else None),
            rationale=f"{intent.strategy}: {risk_decision.summary()}",
            contributions={
                "strategy": intent.strategy,
                "side": intent.side.value,
                "reducing": reducing,
                "edge": str(intent.edge),
                "risk_verdict": risk_decision.verdict.value,
                "binding_constraint": risk_decision.binding_constraint,
                "sized_qty": str(risk_decision.qty),
                "reasons": risk_decision.reasons,
            },
            metadata={
                "mode": self.settings.mode.value,
                "mandate_expired": mandate.is_expired(now),
                "permitted": key in mandate.permitted,
                "exit_path": reducing,
            },
        )

        result = ExecutionResult(
            instrument_key=key, side=intent.side, reducing=reducing,
            risk=risk_decision, decision=decision,
        )

        if not risk_decision.approved:
            self._persist(decision)
            return result

        # Maker-first posts a passive limit at the touch for ENTRIES only; exits
        # are always taker (a stop-loss must not wait in the queue). A new intent
        # supersedes any order still working for this instrument.
        maker = self.execution is ExecutionStyle.MAKER_FIRST and not reducing
        self._cancel_working(key, now)
        if maker:
            touch = intent.quote.bid if intent.side is Side.BUY else intent.quote.ask
            order = Order(
                instrument=intent.instrument, side=intent.side, qty=risk_decision.qty,
                order_type=OrderType.LIMIT, limit_price=touch,
                time_in_force=TimeInForce.GTC, mode=self.settings.mode,
                strategy=f"{intent.strategy}:maker", decision_id=decision.decision_id,
            )
        else:
            order = Order(
                instrument=intent.instrument, side=intent.side, qty=risk_decision.qty,
                order_type=OrderType.MARKET, mode=self.settings.mode,
                strategy=intent.strategy, decision_id=decision.decision_id,
            )

        try:
            self.venue.submit(order)
            result.submitted = True
        except Exception as exc:                        # noqa: BLE001 - recorded
            result.error = str(exc)
            decision.metadata = {**decision.metadata, "venue_error": str(exc)[:200]}
            log.warning("venue rejected %s order for %s: %s", intent.side.value, key, exc)
            self._persist(decision)
            return result

        # A maker limit that did not fill immediately rests in the book; track it
        # so it can be escalated to taker if it stays unfilled.
        if maker and not order.is_terminal:
            self._working[key] = order
            self._posted_tick[key] = self._tick

        # Fold fills into the ledger. apply_fill is idempotent, so re-scanning the
        # venue's fill list is safe; the order object carries its own fill total.
        self._sync_fills(now)
        result.filled_qty = order.filled_qty
        self._persist(decision)
        return result

    # -- order lifecycle ---------------------------------------------------

    def _sync_fills(self, now: datetime) -> None:
        """Fold every venue fill into the ledger (idempotent, dedup by fill id).

        Called after the venue clock advances (which fills resting maker limits)
        and after each submit, so the book stays current within a tick regardless
        of whether a fill was immediate or came from a rested order.
        """
        for fill in self._venue_fills():
            self.ledger.apply_fill(fill, now=now)

    def _reconcile_working(self, now: datetime) -> None:
        """Retire filled maker orders; escalate stale ones to a taker fill."""
        for key, order in list(self._working.items()):
            if order.is_terminal:
                self._working.pop(key, None)
                self._posted_tick.pop(key, None)
                continue
            age = self._tick - self._posted_tick.get(key, self._tick)
            if age >= self.maker_timeout_ticks:
                remaining = order.remaining_qty
                self._cancel_working(key, now)
                if remaining > ZERO:
                    taker = Order(
                        instrument=order.instrument, side=order.side, qty=remaining,
                        order_type=OrderType.MARKET, mode=self.settings.mode,
                        strategy=f"{order.strategy}:taker_escalation",
                        decision_id=order.decision_id,
                    )
                    try:
                        self.venue.submit(taker)
                        self._sync_fills(now)
                    except Exception as exc:            # noqa: BLE001 - recorded
                        log.warning("taker escalation for %s failed: %s", key, exc)

    def _cancel_working(self, key: str, now: datetime) -> None:
        order = self._working.pop(key, None)
        self._posted_tick.pop(key, None)
        if order is not None and not order.is_terminal:
            try:
                self.venue.cancel(order)
            except Exception as exc:                    # noqa: BLE001 - non-fatal
                log.debug("cancel working order for %s failed: %s", key, exc)

    def _venue_fills(self):
        # PaperVenue exposes .fills; a live venue reports fills through its own
        # channel. Only the paper path is wired here (the loop's default mode).
        return getattr(self.venue, "fills", ())

    def _persist(self, decision: Decision) -> None:
        """Record the decision, best-effort. A store hiccup must not stop trading."""
        writer = getattr(self.store, "write_decision", None)
        if writer is None:
            return
        try:
            writer(decision, self.settings.mode.value)
        except Exception as exc:                        # noqa: BLE001 - non-fatal
            log.debug("could not persist decision %s: %s", decision.decision_id, exc)

    # -- driver -----------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    def run(
        self,
        fast_interval_s: float = 60.0,
        slow_interval_s: float = 3_600.0,
        max_ticks: int | None = None,
        install_signal_handlers: bool = True,
    ) -> dict[str, object]:
        """Drive both tiers until stopped.

        The mandate is refreshed on the slow cadence *and* whenever the current
        one is within one fast interval of expiry, so the fast loop is never left
        acting on an expired mandate between scheduled refreshes. If a refresh
        fails, the old mandate simply ages out and the fast loop stops opening
        new risk -- the fail-safe direction.
        """
        self._running = True
        if install_signal_handlers:
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError, ValueError):
                    signal.signal(sig, lambda *_: self.stop())

        mandate = self.refresh_mandate()
        last_slow = time.monotonic()
        ticks = 0

        while self._running:
            now = utcnow()
            due = (time.monotonic() - last_slow) >= slow_interval_s
            expiring = mandate.expires_at - now <= timedelta(seconds=fast_interval_s)
            if due or expiring:
                try:
                    mandate = self.refresh_mandate(now=now)
                    last_slow = time.monotonic()
                except Exception as exc:                # noqa: BLE001 - keep trading exits
                    log.error("mandate refresh failed (%s); running on the existing one", exc)

            try:
                self.fast_tick(mandate, now=now)
            except Exception as exc:                    # noqa: BLE001 - one bad tick is not fatal
                log.error("fast tick failed: %s", exc)

            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            if self._running and fast_interval_s > 0:
                time.sleep(fast_interval_s)

        return {"ticks": ticks, "mandate": self._deliberation.summary() if self._deliberation else None}


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def _default_strategy_factory(interval: str, continuous: bool, spread_bps: Decimal):
    from spintrader.agents.personas.baseline_trend import BaselineTrendAgent

    def factory() -> BaselineTrendAgent:
        return BaselineTrendAgent(
            interval=interval, continuous=continuous, spread_bps=spread_bps,
        )
    return factory


def build_paper_loop(
    symbols: Sequence[str],
    *,
    settings: Settings | None = None,
    store=None,
    mandate_service: MandateService | None = None,
    starting_cash: Decimal | str | float | None = None,
    strategy_factory: Callable[[], object] | None = None,
    interval: str = "1m",
    with_regime: bool = False,
    slippage: SlippageModel | None = None,
    execution: ExecutionStyle = ExecutionStyle.TAKER,
    maker_timeout_ticks: int = 3,
) -> DecisionLoop:
    """Wire a paper-mode loop over Kraken crypto symbols.

    Trades a PAPER instrument per symbol -- carrying the same crypto cost model
    the backtester uses, so paper and backtest results are comparable -- while
    reading bars from the store under the Kraken ingest key (``kraken:BTC-USD``).
    Builds a :class:`PaperVenue`, :class:`Ledger` and :class:`RiskEngine` seeded
    with the same starting cash, so the risk state the engine sizes against and
    the cash the venue fills against never diverge. The mode is forced to PAPER
    and the live gate stays closed.

    The trading instrument is PAPER-venue on purpose: ``Venue.submit`` refuses an
    order whose instrument belongs to another venue, so trading Kraken-venue
    instruments through the PaperVenue would reject every order.
    """
    from spintrader.agents.personas.spec import Horizon
    from spintrader.backtest.runner import backtest_instrument, costs_for
    from spintrader.core.types import AssetClass

    base = settings or get_settings()
    # Force paper: this helper never builds a live-armed loop.
    settings = Settings(
        mode=TradingMode.PAPER,
        aggression=base.aggression,
        base_currency=base.base_currency,
        llm=base.llm, storage=base.storage,
        live=base.live,
        crypto_universe=base.crypto_universe,
        equity_universe=base.equity_universe,
        log_level=base.log_level,
        dry_run=base.dry_run,
        paper_equity_override=base.paper_equity_override,
        enforce_cash_account=base.enforce_cash_account,
    )

    if store is None:
        from spintrader.data.store import Store
        store = Store(settings.storage)
        store.connect()

    # Trade PAPER twins (correct venue for the PaperVenue, consistent costs),
    # and remember where each one's bars actually live in the store.
    instruments = [backtest_instrument(sym, AssetClass.CRYPTO) for sym in symbols]
    data_keys = {inst.key: f"kraken:{sym}" for inst, sym in zip(instruments, symbols)}

    cash = to_decimal(
        starting_cash if starting_cash is not None
        else (settings.paper_equity_override or Decimal("100"))
    )
    costs = costs_for(AssetClass.CRYPTO)

    loop_holder: dict[str, DecisionLoop] = {}

    def quote_source(instrument: Instrument) -> Quote:
        return loop_holder["loop"].market_quote(instrument)

    venue = PaperVenue(
        quote_source=quote_source,
        starting_cash=cash,
        currency=settings.base_currency,
        settings=settings,
        slippage=slippage or costs.slippage,
        settlement_days=1 if settings.enforce_cash_account else 0,
    )
    for instrument in instruments:
        venue.register(instrument)
    venue.connect()

    ledger = Ledger(
        base_currency=settings.base_currency,
        mode=TradingMode.PAPER,
        settlement_days=1 if settings.enforce_cash_account else 0,
        opening_cash={settings.base_currency: cash},
    )
    risk = RiskEngine(settings=settings)

    factory = strategy_factory or _default_strategy_factory(
        interval, continuous=True, spread_bps=costs.spread_bps,
    )
    strategies = {instrument.key: factory() for instrument in instruments}

    if mandate_service is None:
        from spintrader.agents.personas.roster import default_roster
        mandate_service = MandateService(default_roster())

    loop = DecisionLoop(
        settings=settings, store=store, venue=venue, ledger=ledger, risk=risk,
        strategies=strategies, mandate_service=mandate_service,
        instruments=instruments, interval=interval, spread_bps=costs.spread_bps,
        with_regime=with_regime, horizon=Horizon.INTRADAY, data_keys=data_keys,
        execution=execution, maker_timeout_ticks=maker_timeout_ticks,
    )
    loop_holder["loop"] = loop
    return loop


__all__ = [
    "DecisionLoop", "ExecutionResult", "LiveCursor", "build_paper_loop",
]
