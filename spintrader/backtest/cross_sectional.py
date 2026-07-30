"""Cross-sectional backtester: rank a universe, size the top of the ranking.

Why this exists
---------------
:class:`~spintrader.backtest.engine.BacktestEngine` replays *one* instrument. That
is the right shape for testing whether a signal has an edge in isolation, but it
structurally cannot express the decision that most diversified alpha comes from:
*given the whole universe right now, which names are the best to own?* A per-name
backtester run N times and averaged is not the same thing -- it never has to
choose between two names competing for the same capital, so it never pays the
opportunity cost that a real book pays, and it cannot respect a portfolio-wide
``max_positions`` or gross-exposure cap because each run only sees one name.

This module adds that missing shape **without touching the single-instrument
path**. It subclasses the engine so the fill model, the sizing, and the P&L
arithmetic are the *same code* -- every order still goes through
:meth:`BacktestEngine._execute`, the same :class:`PaperVenue`, :class:`RiskEngine`
and :class:`Ledger`. What is new is only the outer loop: it aligns a universe of
bar series on a shared timeline, asks a per-instrument strategy for an edge on
each name that exists at the current instant, ranks the names, and funds the top
of that ranking through the shared risk engine and book.

Causality and alignment
-----------------------
The universe is aligned on the **union** of all instruments' bar timestamps. At
an instant ``t`` an instrument is *rankable only if it actually has a bar at
t*; a name that is missing then is simply not a candidate -- it is never
forward-filled with a stale price into a ranking decision, because a decision
made on a price that has not printed is a decision made on the future. Each
instrument keeps its own :class:`ReplayCursor`, advanced only up to ``t``, so a
strategy physically cannot see any instrument's future bar. Truncating the
universe at ``t`` and replaying produces byte-identical decisions up to ``t``.

Long-only, cash account
-----------------------
Like the personas it drives, this is long-only against a cash account. Exits
(risk-reducing sells) are always processed, ahead of any entry, so a name can
always be closed (lessons L1). Only entries are ranked and capped: the book adds
new risk to the strongest signals first and refuses the rest once
``max_positions`` or the gross-exposure ceiling is reached -- both enforced by
the very same risk engine the live loop uses, never re-implemented here.

The output is an ordinary :class:`BacktestResult` / :class:`Scorecard`, so the
promotion gate and the whole self-improvement apparatus apply to a cross-sectional
strategy with no change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Mapping, Sequence

from spintrader.backtest.engine import (
    BacktestEngine, BacktestResult, ReplayCursor, WalkForwardResult, make_folds,
)
from spintrader.backtest.scorecard import score
from spintrader.core.config import Aggression, Settings
from spintrader.core.types import (
    AssetClass, Bar, Instrument, Side, TradingMode, to_decimal,
)
from spintrader.portfolio.ledger import Ledger
from spintrader.quant.features import periods_per_year
from spintrader.risk.engine import Mandate, RiskEngine, TradeIntent
from spintrader.venues.base import VenueError
from spintrader.venues.paper import PaperVenue

log = logging.getLogger(__name__)

ZERO = Decimal("0")


# --------------------------------------------------------------------------
# Per-instrument lane
# --------------------------------------------------------------------------

@dataclass(slots=True)
class _Track:
    """One instrument's inputs to a replay: its bars and its own strategy.

    Each instrument gets its *own* strategy instance because the personas carry
    per-name state (whether they are long, the entry mark, the trailing peak).
    One shared instance would smear one name's position state across the whole
    universe.
    """
    instrument: Instrument
    bars: Sequence[Bar]
    strategy: object


@dataclass(slots=True)
class _Lane:
    """Live replay state for one instrument, derived from a :class:`_Track`."""
    instrument: Instrument
    strategy: object
    cursor: ReplayCursor
    index_by_ts: dict[datetime, int]
    base_close: float


def _validate_series(instrument: Instrument, bars: Sequence[Bar]) -> None:
    """Reject a series that cannot be aligned deterministically.

    Bars must be non-empty and strictly ascending in time. A duplicate or
    out-of-order timestamp would make "the bar at ``t``" ambiguous, and the
    alignment silently wrong -- the exact failure this backtester exists to
    avoid.
    """
    if not bars:
        raise ValueError(f"{instrument.key}: empty bar series")
    prev: datetime | None = None
    for bar in bars:
        if prev is not None and bar.ts <= prev:
            raise ValueError(
                f"{instrument.key}: bars must be strictly ascending in ts "
                f"(offending timestamp {bar.ts})"
            )
        prev = bar.ts


def rank_key(intent: TradeIntent) -> Decimal:
    """The cross-sectional score: expected return weighted by conviction.

    ``edge * confidence`` ranks a strong-but-tentative signal below a
    slightly-weaker-but-certain one, which is the trade-off a book actually
    faces when two names compete for the same slot.
    """
    return intent.edge * intent.confidence


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class CrossSectionalEngine(BacktestEngine):
    """Ranks a universe each bar and funds the top-K through the shared book.

    Subclasses :class:`BacktestEngine` so that sizing, fills and P&L are the
    inherited code, not a parallel implementation. Only the replay loop is new.
    """

    def __init__(self, *args, top_k: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # ``top_k`` bounds how many names may be *entered* per bar; ``None``
        # defers to the profile's ``max_positions`` (enter up to the book cap).
        self.top_k = top_k
        # The last run's book and venue, kept for introspection and debugging.
        self._last_ledger: Ledger | None = None
        self._last_venue: PaperVenue | None = None

    # -- public entry points ----------------------------------------------

    def run_cross_sectional(
        self,
        strategy_factory: Callable[[], object],
        members: Sequence[tuple[Instrument, Sequence[Bar]]],
        n_trials: int = 1,
    ) -> BacktestResult:
        """Replay the whole universe once and score it.

        A fresh strategy is built per instrument (no fit); use
        :meth:`walk_forward_cross_sectional` when models must be refit per fold.
        """
        tracks = [_Track(inst, bars, strategy_factory()) for inst, bars in members]
        return self._replay(tracks, n_trials=n_trials)

    def walk_forward_cross_sectional(
        self,
        strategy_factory: Callable[[], object],
        members: Sequence[tuple[Instrument, Sequence[Bar]]],
        train_size: int,
        test_size: int,
        step: int | None = None,
        anchored: bool = False,
        embargo: int = 0,
        n_trials: int = 1,
    ) -> WalkForwardResult:
        """Refit per fold on the shared timeline, evaluate out-of-sample only.

        Folds are cut on the *union* timeline, then each instrument's bars are
        sliced to the fold's train and test windows. A name absent from a window
        simply does not participate in that fold -- the same "not rankable when
        missing" rule the inner loop applies, lifted to the fold level. Each
        fold builds fresh per-instrument strategies, fits them on the train
        slice, and runs them on the test slice, so no fitted state crosses a
        fold boundary.
        """
        timeline = _union_timeline(members)
        folds = make_folds(len(timeline), train_size, test_size, step=step,
                           anchored=anchored, embargo=embargo)
        name = getattr(strategy_factory(), "name", "strategy")
        out = WalkForwardResult(strategy=name, instrument_key="xsection")

        for fold in folds:
            train_lo, train_hi = timeline[fold.train_start], timeline[fold.train_end - 1]
            test_lo, test_hi = timeline[fold.test_start], timeline[fold.test_end - 1]

            tracks: list[_Track] = []
            for inst, bars in members:
                test_bars = [b for b in bars if test_lo <= b.ts <= test_hi]
                if not test_bars:
                    continue                      # not in this fold's window
                train_bars = [b for b in bars if train_lo <= b.ts <= train_hi]
                strategy = strategy_factory()
                try:
                    strategy.fit(train_bars)
                except Exception as exc:          # noqa: BLE001 - skip this name
                    log.warning("fold %d: fit failed for %s, skipping (%s)",
                                fold.index, inst.key, exc)
                    continue
                tracks.append(_Track(inst, test_bars, strategy))

            if not tracks:
                continue
            try:
                out.folds.append(self._replay(tracks, n_trials=n_trials))
            except ValueError as exc:
                log.warning("fold %d: run failed (%s)", fold.index, exc)

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

    # -- the replay loop ---------------------------------------------------

    def _replay(self, tracks: Sequence[_Track], n_trials: int = 1) -> BacktestResult:
        """Align, rank and fill a universe over its shared timeline.

        The mechanics mirror :meth:`BacktestEngine.run` bar-for-bar so that a
        one-name universe reproduces the single-instrument result exactly; the
        only additions are cross-instrument ranking and the portfolio caps.
        """
        if not tracks:
            raise ValueError("cross-sectional replay needs at least one instrument")

        lanes: list[_Lane] = []
        all_ts: set[datetime] = set()
        for track in tracks:
            bars = list(track.bars)
            _validate_series(track.instrument, bars)
            index_by_ts = {b.ts: i for i, b in enumerate(bars)}
            lanes.append(_Lane(
                instrument=track.instrument, strategy=track.strategy,
                cursor=ReplayCursor(bars), index_by_ts=index_by_ts,
                base_close=float(bars[0].close),
            ))
            all_ts.update(index_by_ts)

        timeline = sorted(all_ts)
        if len(timeline) < 2:
            raise ValueError("need at least two aligned timestamps to backtest")

        by_key = {lane.instrument.key: lane for lane in lanes}

        def quote_source(instrument: Instrument):
            lane = by_key.get(instrument.key)
            if lane is None:
                raise VenueError(f"no lane for {instrument.key}")
            return lane.cursor.quote(self.spread_bps)

        venue = PaperVenue(
            quote_source=quote_source,
            starting_cash=self.starting_cash,
            currency=self.settings.base_currency,
            settings=self.settings,
            slippage=self.slippage,
            settlement_days=1 if self.settings.enforce_cash_account else 0,
        )
        for lane in lanes:
            venue.register(lane.instrument)
        venue.connect()

        ledger = Ledger(
            base_currency=self.settings.base_currency,
            mode=TradingMode.BACKTEST,
            settlement_days=1 if self.settings.enforce_cash_account else 0,
            opening_cash={self.settings.base_currency: self.starting_cash},
        )
        risk = RiskEngine(settings=self.settings)
        mandate = Mandate.open_mandate(list(by_key), hours=24 * 365 * 100)
        profile = self.settings.risk
        top_k = self.top_k if self.top_k is not None else profile.max_positions

        result = BacktestResult(
            strategy=getattr(lanes[0].strategy, "name", type(lanes[0].strategy).__name__),
            instrument_key="xsection:" + ",".join(sorted(by_key)),
            start=timeline[0], end=timeline[-1],
        )

        appeared: set[str] = set()
        realized_before = ZERO

        for t in timeline:
            venue.set_time(t)
            ledger.settle(t)

            # Which names exist *at this instant*? Advance only those cursors,
            # only up to t. A missing name is simply not present -- not ranked,
            # not forward-filled.
            present: list[_Lane] = []
            for lane in lanes:
                i = lane.index_by_ts.get(t)
                if i is None:
                    continue
                while lane.cursor.index < i:
                    lane.cursor.advance()
                present.append(lane)
                appeared.add(lane.instrument.key)

            # Mark the book at the instant's known closes so the risk engine
            # values held names at current prices, not stale ones.
            marks = {lane.instrument.key: to_decimal(lane.cursor.current.close)
                     for lane in present}
            ledger.mark(marks)

            sells: list[tuple[TradeIntent, _Lane]] = []
            buys: list[tuple[TradeIntent, _Lane]] = []
            for lane in present:
                try:
                    intents = lane.strategy.on_bar(lane.cursor, lane.instrument, mandate)
                except Exception as exc:          # noqa: BLE001 - recorded
                    log.warning("strategy %s raised at %s on %s: %s",
                                result.strategy, t, lane.instrument.key, exc)
                    intents = ()
                for intent in intents or ():
                    (sells if intent.side is Side.SELL else buys).append((intent, lane))

            # Exits first, always: closing a name reduces risk and must never be
            # blocked by an entry cap (L1).
            for intent, lane in sells:
                self._execute(intent, venue, ledger, risk, mandate, lane.cursor, result)

            # Entries: rank by conviction-weighted edge, then fund the top of the
            # ranking subject to the book-wide position count. Gross exposure,
            # per-name weight and cash are enforced inside ``_execute`` by the
            # risk engine, so the ranking never has to duplicate them.
            ranked = sorted(buys, key=lambda pair: rank_key(pair[0]), reverse=True)[:top_k]
            open_keys = {k for k, p in ledger.positions.items() if not p.is_flat}
            for intent, lane in ranked:
                key = intent.instrument.key
                if key not in open_keys and len(open_keys) >= profile.max_positions:
                    result.n_rejected += 1
                    result.rejections["max_positions"] = \
                        result.rejections.get("max_positions", 0) + 1
                    continue
                self._execute(intent, venue, ledger, risk, mandate, lane.cursor, result)
                if not ledger.position(key).is_flat:
                    open_keys.add(key)

            # One equity point per instant, so the Sharpe denominator is the true
            # sample size. Absent names keep their last mark for valuation only.
            ledger.mark(marks)
            snapshot = ledger.value(prices=marks)
            result.equity_curve.append(float(snapshot.equity))
            result.timestamps.append(t)
            result.benchmark_curve.append(self._benchmark(lanes, appeared))

            realized_now = ledger.realized_pnl
            if realized_now != realized_before:
                result.trade_pnls.append(float(realized_now - realized_before))
                realized_before = realized_now

            if t == timeline[-1]:
                break

        result.fees_paid = float(ledger.fees_paid)
        gross_traded = sum(float(f.qty * f.price) for f in venue.fills)
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

        self._last_ledger = ledger
        self._last_venue = venue
        return result

    def _benchmark(self, lanes: Sequence[_Lane], appeared: set[str]) -> float:
        """Equal-weight buy-and-hold of the names that have appeared so far.

        Each name is indexed to its own first close, so a name that enters the
        universe late joins the index at 1.0 rather than distorting it. With a
        single name this is exactly that name's buy-and-hold, which is what makes
        the one-instrument case reproduce the single-instrument benchmark. Absent
        names use their last known close -- a valuation, not a decision, so no
        lookahead is introduced.
        """
        ratios = [
            float(lane.cursor.current.close) / lane.base_close
            for lane in lanes
            if lane.instrument.key in appeared and lane.base_close > 0
        ]
        if not ratios:
            return float(self.starting_cash)
        return float(self.starting_cash) * (sum(ratios) / len(ratios))


def _union_timeline(
    members: Sequence[tuple[Instrument, Sequence[Bar]]],
) -> list[datetime]:
    """The sorted union of every instrument's bar timestamps."""
    return sorted({b.ts for _, bars in members for b in bars})


# --------------------------------------------------------------------------
# Runner entry
# --------------------------------------------------------------------------

@dataclass(slots=True)
class CrossSectionalRun:
    """A cross-sectional backtest plus the context needed to interpret it.

    Mirrors :class:`~spintrader.backtest.runner.BacktestRun` in the attributes
    the improvement loop reads (``result``/``walk_forward``/``scorecard``), so a
    cross-sectional run is a drop-in wherever a single-instrument run was.
    """
    result: BacktestResult
    walk_forward: WalkForwardResult | None
    instruments: list[Instrument]
    settings: Settings
    starting_cash: Decimal
    n_bars: int

    @property
    def scorecard(self):
        return self.result.scorecard


def run_cross_sectional_backtest(
    strategy_factory: Callable[..., object],
    universe: Mapping[str, Sequence[Bar]],
    asset_class: AssetClass = AssetClass.EQUITY,
    aggression: Aggression | str = Aggression.MODERATE,
    starting_cash: Decimal | str | float = "1000",
    enforce_cash_account: bool = True,
    top_k: int | None = None,
    n_trials: int = 1,
    walk_forward: bool = True,
    train_size: int | None = None,
    test_size: int | None = None,
    embargo: int | None = None,
    strategy_kwargs: dict[str, object] | None = None,
) -> CrossSectionalRun:
    """Replay a universe through the live components and score the outcome.

    ``universe`` maps a symbol to its aligned bar series. The same cost model and
    backtest-pinned settings as :func:`~spintrader.backtest.runner.run_backtest`
    are used, so a cross-sectional run is comparable to the single-instrument
    runs the rest of the system produces; only the ranking-and-cap loop differs.
    """
    # Imported here rather than at module load to keep this file's dependency on
    # the runner one-directional and lazy (the runner does not import this).
    from spintrader.backtest.runner import (
        backtest_instrument, backtest_settings, costs_for,
    )

    if not universe:
        raise ValueError("cross-sectional backtest needs at least one instrument")

    costs = costs_for(asset_class)
    settings = backtest_settings(
        aggression=aggression, enforce_cash_account=enforce_cash_account,
    )
    cash = to_decimal(starting_cash)

    members: list[tuple[Instrument, Sequence[Bar]]] = []
    interval: str | None = None
    for symbol, bars in universe.items():
        series = list(bars)
        if not series:
            raise ValueError(f"{symbol}: empty bar series")
        members.append((backtest_instrument(symbol, asset_class, costs), series))
        interval = interval or series[0].interval

    annualisation = int(periods_per_year(
        interval, continuous=asset_class is AssetClass.CRYPTO,
    ))

    kwargs = dict(strategy_kwargs or {})
    kwargs.setdefault("spread_bps", costs.spread_bps)
    kwargs.setdefault("interval", interval)
    kwargs.setdefault("continuous", asset_class is AssetClass.CRYPTO)

    def factory():
        return strategy_factory(**kwargs)

    engine = CrossSectionalEngine(
        settings=settings, starting_cash=cash, spread_bps=costs.spread_bps,
        slippage=costs.slippage, periods_per_year=annualisation, top_k=top_k,
    )

    result = engine.run_cross_sectional(factory, members, n_trials=n_trials)

    wf: WalkForwardResult | None = None
    if walk_forward:
        timeline = _union_timeline(members)
        warmup = int(getattr(factory(), "warmup_bars", 0) or 0)
        train = train_size or min(756, max(252, len(timeline) // 4))
        test = test_size or max(warmup * 2, 504)
        gap = embargo if embargo is not None else warmup
        if train + test + gap <= len(timeline):
            wf = engine.walk_forward_cross_sectional(
                factory, members, train_size=train, test_size=test,
                embargo=gap, n_trials=n_trials,
            )
        else:
            log.warning(
                "skipping cross-sectional walk-forward: %d aligned bars is short "
                "of the %d needed for a %d/%d split with a %d-bar embargo",
                len(timeline), train + test + gap, train, test, gap,
            )

    return CrossSectionalRun(
        result=result, walk_forward=wf,
        instruments=[inst for inst, _ in members], settings=settings,
        starting_cash=cash, n_bars=len(_union_timeline(members)),
    )


__all__ = [
    "CrossSectionalEngine", "CrossSectionalRun", "rank_key",
    "run_cross_sectional_backtest",
]
