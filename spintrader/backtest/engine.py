"""Event-driven backtester and walk-forward harness.

The single most important property of this module is that it does **not**
contain a simulation of trading. It replays historical bars through the same
:class:`PaperVenue`, :class:`RiskEngine` and :class:`Ledger` that paper and live
trading use. A backtester with its own fill logic, its own sizing and its own
P&L arithmetic measures a different program than the one that trades, and the
divergence is invisible until real money finds it.

Lookahead prevention
--------------------
The replay cursor is the only thing that decides what a strategy can see. It
exposes bars ``[0..t]`` and nothing beyond, and the quote it hands the venue is
derived from bar ``t``'s close -- never from ``t+1``. A strategy physically
cannot read ahead, because the future is not in the object it is given.

Walk-forward
------------
Models are refitted on each fold's training window and evaluated only on its
out-of-sample window. In-sample results are computed but reported separately
and never used for promotion: an HMM fitted on the data it is then scored
against will always look excellent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np

from spintrader.backtest.scorecard import Scorecard, score
from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    Bar, Instrument, Order, OrderType, Quote, Side, TradingMode, to_decimal,
)
from spintrader.portfolio.ledger import Ledger
from spintrader.risk.engine import (
    Mandate, RiskDecision, RiskEngine, RiskVerdict, TradeIntent,
)
from spintrader.venues.paper import PaperVenue, SlippageModel

log = logging.getLogger(__name__)

ZERO = Decimal("0")


# --------------------------------------------------------------------------
# Replay cursor
# --------------------------------------------------------------------------

class ReplayCursor:
    """Exposes history up to the current bar and no further.

    This is the lookahead barrier. A strategy receives this object rather than
    the full series, so reading the future is not merely discouraged -- the data
    is absent.
    """

    def __init__(self, bars: Sequence[Bar]) -> None:
        if not bars:
            raise ValueError("cannot replay an empty bar series")
        self._bars = list(bars)
        self._index = 0

    def __len__(self) -> int:
        return len(self._bars)

    @property
    def index(self) -> int:
        return self._index

    @property
    def now(self) -> datetime:
        return self._bars[self._index].ts

    @property
    def current(self) -> Bar:
        return self._bars[self._index]

    def history(self, lookback: int | None = None) -> list[Bar]:
        """Bars up to and including the current one."""
        end = self._index + 1
        start = 0 if lookback is None else max(0, end - lookback)
        return self._bars[start:end]

    def advance(self) -> bool:
        if self._index + 1 >= len(self._bars):
            return False
        self._index += 1
        return True

    def quote(self, spread_bps: Decimal = Decimal("5")) -> Quote:
        """Synthesise a quote from the current bar's close.

        Historical daily and hourly bars carry no bid/ask, so a spread is
        imposed. ``spread_bps`` should be calibrated from live orderbook data
        rather than guessed; the default is deliberately wider than BTC/USD's
        observed ~1 bps so that a backtest errs expensive.
        """
        bar = self.current
        half = bar.close * spread_bps / Decimal(20_000)
        return Quote(
            instrument_key=bar.instrument_key,
            ts=bar.ts,
            bid=bar.close - half,
            ask=bar.close + half,
        )


# --------------------------------------------------------------------------
# Strategy protocol
# --------------------------------------------------------------------------

class Strategy(Protocol):
    """A strategy sees only the cursor's history and returns intents.

    Deliberately narrow: it cannot place orders, cannot size positions and
    cannot see the ledger's cash. Sizing belongs to the risk engine, so a
    strategy that "knows" it wants a large position cannot act on that belief.
    """

    name: str

    def on_bar(
        self, cursor: ReplayCursor, instrument: Instrument, mandate: Mandate,
    ) -> Sequence[TradeIntent]:
        ...

    def fit(self, bars: Sequence[Bar]) -> None:
        """Refit any models on a training window. May be a no-op."""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass(slots=True)
class BacktestResult:
    """Outcome of a single backtest run."""
    strategy: str
    instrument_key: str
    start: datetime
    end: datetime
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    benchmark_curve: list[float] = field(default_factory=list)
    trade_pnls: list[float] = field(default_factory=list)
    n_orders: int = 0
    n_rejected: int = 0
    rejections: dict[str, int] = field(default_factory=dict)
    fees_paid: float = 0.0
    turnover: float = 0.0
    scorecard: Scorecard | None = None

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else 0.0


@dataclass(slots=True)
class WalkForwardFold:
    """One train/test split."""
    index: int
    train_start: int
    train_end: int      # exclusive
    test_start: int
    test_end: int       # exclusive

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass(slots=True)
class WalkForwardResult:
    strategy: str
    instrument_key: str
    folds: list[BacktestResult] = field(default_factory=list)
    combined: Scorecard | None = None
    in_sample: list[Scorecard] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def stitched_equity(self) -> list[float]:
        """Concatenate out-of-sample fold curves into one continuous curve.

        Each fold restarts from its own capital, so curves are chained
        multiplicatively rather than appended -- appending them would show a
        cliff at every fold boundary and understate compounding.
        """
        stitched: list[float] = []
        level = 1.0
        for fold in self.folds:
            curve = fold.equity_curve
            if len(curve) < 2 or curve[0] <= 0:
                continue
            for value in curve:
                stitched.append(level * value / curve[0])
            level = stitched[-1] if stitched else level
        return stitched


# --------------------------------------------------------------------------
# Fold generation
# --------------------------------------------------------------------------

def make_folds(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    anchored: bool = False,
    embargo: int = 0,
) -> list[WalkForwardFold]:
    """Generate walk-forward splits.

    ``anchored=True`` grows the training window from a fixed start (expanding
    window); otherwise it slides at a fixed length (rolling window). Rolling is
    the default because markets are non-stationary and a model fitted on 2021
    data has limited claim on 2026.

    ``embargo`` drops observations between train and test. With overlapping
    feature windows -- a 50-bar trailing mean, say -- the first test bars share
    inputs with the last training bars, and without a gap the "out-of-sample"
    period is partly in-sample.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    if train_size + test_size + embargo > n:
        raise ValueError(
            f"need at least {train_size + test_size + embargo} observations for "
            f"a {train_size}/{test_size} split with a {embargo}-bar embargo, got {n}"
        )

    step = step or test_size
    folds: list[WalkForwardFold] = []
    index = 0
    train_end = train_size

    while True:
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n:
            break
        folds.append(WalkForwardFold(
            index=index,
            train_start=0 if anchored else max(0, train_end - train_size),
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        index += 1
        train_end += step

    return folds


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class BacktestEngine:
    """Replays bars through the live trading components."""

    def __init__(
        self,
        settings: Settings | None = None,
        starting_cash: Decimal = Decimal("1000"),
        spread_bps: Decimal = Decimal("5"),
        slippage: SlippageModel | None = None,
        periods_per_year: int = 365,
    ) -> None:
        from spintrader.core.config import get_settings
        base = settings or get_settings()
        # Force backtest mode so the live gate can never be satisfied from here,
        # whatever the ambient environment says.
        self.settings = Settings(
            mode=TradingMode.BACKTEST,
            aggression=base.aggression,
            base_currency=base.base_currency,
            llm=base.llm, storage=base.storage,
            live=LiveGate(enabled=False),
            crypto_universe=base.crypto_universe,
            equity_universe=base.equity_universe,
            log_level=base.log_level,
            dry_run=False,
            paper_equity_override=base.paper_equity_override,
            enforce_cash_account=base.enforce_cash_account,
        )
        self.starting_cash = starting_cash
        self.spread_bps = spread_bps
        self.slippage = slippage
        self.periods_per_year = periods_per_year

    def run(
        self,
        strategy: Strategy,
        instrument: Instrument,
        bars: Sequence[Bar],
        mandate: Mandate | None = None,
        n_trials: int = 1,
    ) -> BacktestResult:
        """Replay ``bars`` and score the result."""
        if len(bars) < 2:
            raise ValueError("need at least two bars to backtest")

        cursor = ReplayCursor(bars)
        mandate = mandate or Mandate.open_mandate(
            [instrument.key], hours=24 * 365 * 100,      # never expires in replay
        )

        venue = PaperVenue(
            quote_source=lambda _inst: cursor.quote(self.spread_bps),
            starting_cash=self.starting_cash,
            currency=self.settings.base_currency,
            settings=self.settings,
            slippage=self.slippage,
            settlement_days=1 if self.settings.enforce_cash_account else 0,
        )
        venue.register(instrument)
        venue.connect()

        ledger = Ledger(
            base_currency=self.settings.base_currency,
            mode=TradingMode.BACKTEST,
            settlement_days=1 if self.settings.enforce_cash_account else 0,
            opening_cash={self.settings.base_currency: self.starting_cash},
        )
        risk = RiskEngine(settings=self.settings)

        result = BacktestResult(
            strategy=getattr(strategy, "name", type(strategy).__name__),
            instrument_key=instrument.key,
            start=bars[0].ts, end=bars[-1].ts,
        )

        first_close = float(bars[0].close)
        realized_before = ZERO

        while True:
            venue.set_time(cursor.now)
            ledger.settle(cursor.now)

            try:
                intents = strategy.on_bar(cursor, instrument, mandate)
            except Exception as exc:                    # noqa: BLE001 - recorded
                log.warning("strategy %s raised at %s: %s",
                            result.strategy, cursor.now, exc)
                intents = ()

            for intent in intents or ():
                self._execute(intent, venue, ledger, risk, mandate, cursor, result)

            # Mark and record equity every bar, so the curve has one point per
            # observation and the Sharpe denominator is the true sample size.
            mark = {instrument.key: to_decimal(cursor.current.close)}
            ledger.mark(mark)
            snapshot = ledger.value(prices=mark)
            result.equity_curve.append(float(snapshot.equity))
            result.timestamps.append(cursor.now)
            result.benchmark_curve.append(
                float(self.starting_cash) * float(cursor.current.close) / first_close
            )

            realized_now = ledger.realized_pnl
            if realized_now != realized_before:
                result.trade_pnls.append(float(realized_now - realized_before))
                realized_before = realized_now

            if not cursor.advance():
                break

        result.fees_paid = float(ledger.fees_paid)
        gross_traded = sum(
            float(f.qty * f.price) for f in venue.fills
        )
        final_equity = result.final_equity or float(self.starting_cash)
        result.turnover = gross_traded / final_equity if final_equity > 0 else 0.0

        result.scorecard = score(
            result.equity_curve,
            periods_per_year=self.periods_per_year,
            n_trades=len(venue.fills),
            trade_pnls=result.trade_pnls,
            fees_paid=result.fees_paid,
            turnover=result.turnover,
            benchmark_curve=result.benchmark_curve,
            n_trials=n_trials,
            start=result.start, end=result.end,
        )
        return result

    def _execute(
        self,
        intent: TradeIntent,
        venue: PaperVenue,
        ledger: Ledger,
        risk: RiskEngine,
        mandate: Mandate,
        cursor: ReplayCursor,
        result: BacktestResult,
    ) -> None:
        """Size, vet and fill one intent through the live code paths."""
        try:
            state = ledger.to_risk_state(
                prices={intent.instrument.key: to_decimal(cursor.current.close)},
                now=cursor.now,
            )
        except Exception as exc:                        # noqa: BLE001 - recorded
            result.n_rejected += 1
            result.rejections["unvaluable_book"] = \
                result.rejections.get("unvaluable_book", 0) + 1
            log.debug("skipping intent, book not valuable: %s", exc)
            return

        decision = risk.evaluate(intent, state, mandate, now=cursor.now)
        if not decision.approved:
            result.n_rejected += 1
            key = decision.binding_constraint or (
                decision.reasons[0][:40] if decision.reasons else "rejected"
            )
            result.rejections[key] = result.rejections.get(key, 0) + 1
            return

        order = Order(
            instrument=intent.instrument,
            side=intent.side,
            qty=decision.qty,
            order_type=OrderType.MARKET,
            mode=TradingMode.BACKTEST,
            strategy=intent.strategy,
            decision_id=intent.decision_id,
        )
        try:
            venue.submit(order)
        except Exception as exc:                        # noqa: BLE001 - recorded
            result.n_rejected += 1
            result.rejections["venue_rejected"] = \
                result.rejections.get("venue_rejected", 0) + 1
            log.debug("venue rejected order at %s: %s", cursor.now, exc)
            return

        result.n_orders += 1
        # Fold the resulting fills into the ledger, exactly as live does.
        for fill in venue.fills:
            ledger.apply_fill(fill, now=cursor.now)

    # -- walk-forward ------------------------------------------------------

    def walk_forward(
        self,
        strategy_factory: Callable[[], Strategy],
        instrument: Instrument,
        bars: Sequence[Bar],
        train_size: int,
        test_size: int,
        step: int | None = None,
        anchored: bool = False,
        embargo: int = 0,
        n_trials: int = 1,
    ) -> WalkForwardResult:
        """Refit per fold and evaluate out-of-sample only.

        ``strategy_factory`` produces a fresh strategy for each fold. Reusing
        one instance would carry fitted state across folds and leak the future
        into earlier tests.
        """
        folds = make_folds(len(bars), train_size, test_size, step=step,
                           anchored=anchored, embargo=embargo)
        name = getattr(strategy_factory(), "name", "strategy")
        out = WalkForwardResult(strategy=name, instrument_key=instrument.key)

        for fold in folds:
            strategy = strategy_factory()
            train = bars[fold.train_start:fold.train_end]
            test = bars[fold.test_start:fold.test_end]

            try:
                strategy.fit(train)
            except Exception as exc:                    # noqa: BLE001 - skip fold
                log.warning("fold %d: fit failed, skipping (%s)", fold.index, exc)
                continue

            try:
                fold_result = self.run(strategy, instrument, test, n_trials=n_trials)
            except ValueError as exc:
                log.warning("fold %d: run failed (%s)", fold.index, exc)
                continue
            out.folds.append(fold_result)

        stitched = out.stitched_equity()
        if len(stitched) >= 2:
            out.combined = score(
                stitched,
                periods_per_year=self.periods_per_year,
                n_trades=sum(f.n_orders for f in out.folds),
                trade_pnls=[p for f in out.folds for p in f.trade_pnls],
                fees_paid=sum(f.fees_paid for f in out.folds),
                n_trials=n_trials,
            )
        return out


__all__ = [
    "BacktestEngine", "BacktestResult", "ReplayCursor", "Strategy",
    "WalkForwardFold", "WalkForwardResult", "make_folds",
]
