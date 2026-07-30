"""Wiring that turns a persona plus a bar series into a scored backtest.

The engine in :mod:`spintrader.backtest.engine` deliberately knows nothing about
where bars come from or what an instrument's fees are. This module supplies both,
in one place, so that every backtest in the project is run against the same cost
assumptions. Cost assumptions scattered across call sites are how two backtests
of the same strategy come to disagree.

Bar sources
-----------
Two, in priority order:

1. the TimescaleDB store, which is the system of record; and
2. a CSV file, for a machine that cannot reach it.

The CSV path is not a convenience. A backtest that can only run next to the
database cannot be reproduced by anyone reviewing it, and an unreproducible
backtest is an assertion, not evidence.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Callable, Sequence

from spintrader.backtest.engine import (
    BacktestEngine, BacktestResult, WalkForwardResult,
)
from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    AssetClass, Bar, Instrument, TradingMode, VenueId, ensure_utc, to_decimal,
)
from spintrader.venues.paper import SlippageModel

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Cost assumptions
# --------------------------------------------------------------------------
#
# Every figure below is set *worse* than the venue's observed cost, and the
# margin is stated. A backtest tuned to optimistic costs is a backtest of a
# market that does not exist, and the error is always in the flattering
# direction because low costs make high turnover look free.

@dataclass(frozen=True, slots=True)
class CostModel:
    """Round-trip trading friction for one asset class."""

    spread_bps: Decimal
    taker_fee: Decimal
    slippage: SlippageModel
    price_increment: Decimal
    qty_increment: Decimal
    min_notional: Decimal
    periods_per_year: int

    def round_trip_bps(self, notional: Decimal) -> Decimal:
        """Approximate all-in cost of entering and exiting, in bps."""
        per_side = (
            self.spread_bps / Decimal(2)
            + self.slippage.slippage_bps(notional)
            + self.taker_fee * Decimal(10_000)
        )
        return per_side * Decimal(2)


# SPY, retail, via IBKR Pro.
#   spread    -- observed ~0.17 bps ($0.01 on ~$600); modelled at 1 bp (~6x)
#   commission-- $0.0035/share tiered is ~0.06 bps at $600; modelled at
#                0.5 bps (~8x), which also absorbs exchange and regulatory fees
#   slippage  -- 0.5 bps floor plus a square-root impact term; on a $150 order
#                that is ~0.5 bps total, which for the most liquid ETF on earth
#                is generous against the trader
EQUITY_COSTS = CostModel(
    spread_bps=Decimal("1"),
    taker_fee=Decimal("0.00005"),
    slippage=SlippageModel(
        base_bps=Decimal("0.5"),
        impact_coefficient=Decimal("2"),
        reference_size=Decimal("1000000"),
    ),
    price_increment=Decimal("0.01"),
    qty_increment=Decimal("0.0001"),
    min_notional=Decimal("1"),
    periods_per_year=252,
)

# BTC/USD on Kraken.
#   spread    -- observed ~1 bp; modelled at 3 bps
#   taker fee -- Kraken's published taker tier is 26 bps at low volume
CRYPTO_COSTS = CostModel(
    spread_bps=Decimal("3"),
    taker_fee=Decimal("0.0026"),
    slippage=SlippageModel(
        base_bps=Decimal("2"),
        impact_coefficient=Decimal("8"),
        reference_size=Decimal("500000"),
    ),
    price_increment=Decimal("0.1"),
    qty_increment=Decimal("0.00000001"),
    min_notional=Decimal("5"),
    periods_per_year=365,
)


def costs_for(asset_class: AssetClass) -> CostModel:
    if asset_class is AssetClass.CRYPTO:
        return CRYPTO_COSTS
    return EQUITY_COSTS


# --------------------------------------------------------------------------
# Instruments
# --------------------------------------------------------------------------

def backtest_instrument(
    symbol: str, asset_class: AssetClass, costs: CostModel | None = None,
) -> Instrument:
    """Build the paper-venue instrument a replay trades.

    The venue is always ``PAPER``. :meth:`Venue.submit` rejects an order whose
    instrument belongs to a different venue, so a backtest instrument stamped
    ``KRAKEN`` or ``IBKR`` would be refused by the very venue the engine
    replays through.
    """
    model = costs or costs_for(asset_class)
    return Instrument(
        symbol=symbol,
        asset_class=asset_class,
        venue=VenueId.PAPER,
        venue_symbol=symbol,
        quote_currency="USD",
        price_increment=model.price_increment,
        qty_increment=model.qty_increment,
        min_qty=Decimal("0"),
        min_notional=model.min_notional,
        supports_fractional=True,
        maker_fee=Decimal("0"),
        taker_fee=model.taker_fee,
    )


# --------------------------------------------------------------------------
# Bar loading
# --------------------------------------------------------------------------

BAR_CSV_FIELDS = ("ts", "open", "high", "low", "close", "volume")


def load_bars_from_csv(
    path: str | Path, instrument_key: str, interval: str = "1d",
) -> list[Bar]:
    """Read bars from a CSV with an ISO-8601 ``ts`` column.

    Timestamps must be timezone-aware or explicitly UTC-suffixed; a naive
    timestamp is rejected rather than assumed, because guessing the zone of a
    daily close silently shifts every bar by up to a day.
    """
    rows: list[Bar] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [f for f in BAR_CSV_FIELDS if f not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )
        for line_no, row in enumerate(reader, start=2):
            raw = (row["ts"] or "").strip()
            if not raw:
                continue
            stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError(
                    f"{path}:{line_no} timestamp {raw!r} has no timezone"
                )
            rows.append(Bar(
                instrument_key=instrument_key,
                ts=ensure_utc(stamp),
                interval=interval,
                open=to_decimal(row["open"]),
                high=to_decimal(row["high"]),
                low=to_decimal(row["low"]),
                close=to_decimal(row["close"]),
                volume=to_decimal(row["volume"] or "0"),
            ))
    rows.sort(key=lambda b: b.ts)
    return rows


def load_bars_from_store(
    instrument_key: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    settings: Settings | None = None,
) -> list[Bar]:
    """Read bars from the TimescaleDB store."""
    from spintrader.core.config import get_settings
    from spintrader.data.store import Store

    resolved = settings or get_settings()
    with Store(resolved.storage) as store:
        return store.read_bars(instrument_key, interval, start=start, end=end)


def write_bars_csv(bars: Sequence[Bar], path: str | Path) -> Path:
    """Persist a bar series so a run can be reproduced byte-for-byte."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(BAR_CSV_FIELDS)
        for bar in bars:
            writer.writerow([
                bar.ts.isoformat(), bar.open, bar.high, bar.low,
                bar.close, bar.volume,
            ])
    return target


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

def backtest_settings(
    aggression: Aggression | str = Aggression.MODERATE,
    enforce_cash_account: bool = True,
    base: Settings | None = None,
) -> Settings:
    """Settings pinned to BACKTEST mode with the live gate closed.

    ``BacktestEngine`` re-pins the mode itself, but doing it here too means a
    caller who builds settings and then uses them for something else cannot
    accidentally carry a live-armed gate into a simulation.
    """
    resolved = Aggression(aggression) if isinstance(aggression, str) else aggression
    template = base or Settings()
    return Settings(
        mode=TradingMode.BACKTEST,
        aggression=resolved,
        base_currency=template.base_currency,
        llm=template.llm,
        storage=template.storage,
        live=LiveGate(enabled=False),
        crypto_universe=template.crypto_universe,
        equity_universe=template.equity_universe,
        log_level=template.log_level,
        dry_run=False,
        paper_equity_override=None,
        enforce_cash_account=enforce_cash_account,
    )


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------

@dataclass(slots=True)
class BacktestRun:
    """A single backtest plus the context needed to interpret it."""
    result: BacktestResult
    walk_forward: WalkForwardResult | None
    instrument: Instrument
    costs: CostModel
    settings: Settings
    starting_cash: Decimal
    strategy_config: dict[str, object]
    n_bars: int

    @property
    def scorecard(self):
        return self.result.scorecard


def run_backtest(
    strategy_factory: Callable[..., object],
    symbol: str,
    bars: Sequence[Bar],
    asset_class: AssetClass = AssetClass.EQUITY,
    aggression: Aggression | str = Aggression.MODERATE,
    starting_cash: Decimal | str | float = "1000",
    enforce_cash_account: bool = True,
    n_trials: int = 1,
    walk_forward: bool = True,
    train_size: int | None = None,
    test_size: int | None = None,
    embargo: int | None = None,
    strategy_kwargs: dict[str, object] | None = None,
) -> BacktestRun:
    """Replay ``bars`` through the live components and score the outcome."""
    if len(bars) < 2:
        raise ValueError("need at least two bars to backtest")

    costs = costs_for(asset_class)
    instrument = backtest_instrument(symbol, asset_class, costs)
    settings = backtest_settings(
        aggression=aggression, enforce_cash_account=enforce_cash_account,
    )
    cash = to_decimal(starting_cash)

    kwargs = dict(strategy_kwargs or {})
    # The persona must price off the same spread the venue fills at.
    kwargs.setdefault("spread_bps", costs.spread_bps)
    kwargs.setdefault("interval", bars[0].interval)
    kwargs.setdefault("continuous", asset_class is AssetClass.CRYPTO)

    def factory():
        return strategy_factory(**kwargs)

    engine = BacktestEngine(
        settings=settings,
        starting_cash=cash,
        spread_bps=costs.spread_bps,
        slippage=costs.slippage,
        periods_per_year=costs.periods_per_year,
    )

    strategy = factory()
    result = engine.run(strategy, instrument, bars, n_trials=n_trials)

    wf: WalkForwardResult | None = None
    if walk_forward:
        warmup = int(getattr(strategy, "warmup_bars", 0) or 0)
        train = train_size or min(756, max(252, len(bars) // 4))
        test = test_size or max(warmup * 2, 504)
        gap = embargo if embargo is not None else warmup
        if train + test + gap <= len(bars):
            wf = engine.walk_forward(
                factory, instrument, bars,
                train_size=train, test_size=test, embargo=gap,
                n_trials=n_trials,
            )
        else:
            log.warning(
                "skipping walk-forward: %d bars is short of the %d needed for "
                "a %d/%d split with a %d-bar embargo",
                len(bars), train + test + gap, train, test, gap,
            )

    return BacktestRun(
        result=result, walk_forward=wf, instrument=instrument, costs=costs,
        settings=settings, starting_cash=cash,
        strategy_config=(
            strategy.describe() if hasattr(strategy, "describe")
            else {"name": getattr(strategy, "name", "strategy")}
        ),
        n_bars=len(bars),
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def format_report(run: BacktestRun) -> str:
    """Human-readable summary.

    Reports the exposure-adjusted comparison alongside the raw one. A long-only
    persona capped at ``max_position_weight`` holds most of the book in cash, so
    measuring it against a fully-invested benchmark answers a question nobody
    asked -- of course 15% of an index underperforms 100% of it. The useful
    question is what the deployed capital earned.
    """
    card = run.result.scorecard
    profile = run.settings.risk
    lines: list[str] = []

    lines.append("=" * 74)
    lines.append(
        f"{run.result.strategy}  |  {run.instrument.symbol}  "
        f"({run.instrument.asset_class.value})"
    )
    lines.append("=" * 74)
    lines.append(
        f"period          {run.result.start:%Y-%m-%d} -> {run.result.end:%Y-%m-%d}  "
        f"({run.n_bars} bars, {run.costs.periods_per_year}/yr)"
    )
    lines.append(
        f"profile         {profile.aggression.value}  "
        f"(kelly {profile.kelly_fraction}, max position {profile.max_position_weight}, "
        f"stop {profile.stop_loss_pct}, min conf {profile.min_confidence})"
    )
    lines.append(
        f"capital         {run.starting_cash} {run.settings.base_currency}  "
        f"-> {run.result.final_equity:,.2f}"
    )
    lines.append(
        f"cash account    {run.settings.enforce_cash_account} "
        f"(T+1 settlement, shorts blocked)"
    )
    lines.append("")

    if card is None:
        lines.append("no scorecard: the run produced too few observations")
        return "\n".join(lines)

    lines.append("-- returns " + "-" * 62)
    lines.append(f"total return    {_pct(card.total_return)}")
    lines.append(f"CAGR            {_pct(card.cagr)}")
    lines.append(f"volatility      {card.volatility:.2%} annualised")
    lines.append(f"benchmark       {_pct(card.benchmark_return)}  (buy and hold)")
    lines.append(f"alpha (raw)     {_pct(card.alpha)}")

    # Exposure-adjusted benchmark: the same buy-and-hold, scaled to the largest
    # position the profile ever permits.
    cap = float(profile.max_position_weight)
    if card.benchmark_return is not None and cap > 0:
        scaled = card.benchmark_return * cap
        lines.append(
            f"benchmark x{cap:.0%}   {_pct(scaled)}  "
            f"(index scaled to the {cap:.0%} position cap)"
        )
        lines.append(f"alpha vs that   {_pct(card.total_return - scaled)}")
    lines.append("")

    lines.append("-- risk " + "-" * 65)
    low, high = card.sharpe_ci95
    lines.append(
        f"Sharpe          {card.sharpe:.2f}  "
        f"(95% CI {low:.2f} .. {high:.2f}, n={card.n_observations})"
    )
    lines.append(f"Sortino         {card.sortino:.2f}")
    lines.append(f"Calmar          {card.calmar:.2f}")
    lines.append(
        f"max drawdown    {card.max_drawdown:.2%}  "
        f"over {card.max_drawdown_days} observations"
    )
    lines.append(
        f"                limit for {profile.aggression.value} is "
        f"{-float(profile.max_drawdown_limit):.2%} "
        f"({'WITHIN' if card.max_drawdown >= -float(profile.max_drawdown_limit) else 'BREACHED'})"
    )
    lines.append(f"skew            {card.skew:+.2f}")
    lines.append(f"excess kurtosis {card.excess_kurtosis:+.2f}")
    lines.append("")

    lines.append("-- trading " + "-" * 62)
    lines.append(f"orders placed   {run.result.n_orders}")
    lines.append(f"fills           {card.n_trades}")
    lines.append(f"closed trades   {len(run.result.trade_pnls)}")
    lines.append(f"hit rate        {card.hit_rate:.1%}")
    pf = card.profit_factor
    lines.append(f"profit factor   {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    lines.append(f"turnover        {card.turnover:.2f}x final equity")
    lines.append(f"fees paid       {card.fees_paid:,.2f}")
    lines.append(f"cost drag       {card.cost_drag:.2%} of gross P&L")
    lines.append("")

    lines.append("-- rejections " + "-" * 59)
    if not run.result.rejections:
        lines.append("none")
    else:
        for key, count in sorted(
            run.result.rejections.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"{count:>6}  {key}")
    lines.append("")

    lines.append("-- significance " + "-" * 57)
    lines.append(f"PSR             {card.psr:.3f}  P(true Sharpe > 0)")
    lines.append(
        f"DSR             {card.deflated_sharpe:.3f}  "
        f"over {card.n_trials} trial(s)"
    )
    lines.append(
        f"verdict         "
        f"{'SIGNIFICANT' if card.is_significant else 'NOT SIGNIFICANT'}"
    )
    lines.append("")

    if run.walk_forward and run.walk_forward.folds:
        wf = run.walk_forward
        lines.append("-- walk-forward (out of sample) " + "-" * 41)
        lines.append(f"folds           {wf.n_folds}")
        for fold in wf.folds:
            fc = fold.scorecard
            lines.append(
                f"  {fold.start:%Y-%m-%d}..{fold.end:%Y-%m-%d}  "
                f"ret {_pct(fc.total_return if fc else None):>8}  "
                f"bench {_pct(fc.benchmark_return if fc else None):>8}  "
                f"maxDD {(fc.max_drawdown if fc else 0):>7.2%}  "
                f"trades {fold.n_orders:>3}"
            )
        if wf.combined:
            lines.append("")
            lines.append(f"stitched        {wf.combined.verdict()}")
        lines.append("")

    lines.append("=" * 74)
    return "\n".join(lines)


def export_run(run: BacktestRun, outdir: str | Path) -> dict[str, Path]:
    """Write the equity curve and scorecard to disk."""
    target = Path(outdir)
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{run.result.strategy}_{run.instrument.symbol}"

    curve_path = target / f"{stem}_equity.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "equity", "benchmark"])
        for ts, eq, bench in zip(
            run.result.timestamps, run.result.equity_curve,
            run.result.benchmark_curve,
        ):
            writer.writerow([ts.isoformat(), f"{eq:.6f}", f"{bench:.6f}"])

    card = run.result.scorecard
    payload: dict[str, object] = {
        "strategy": run.result.strategy,
        "strategy_config": run.strategy_config,
        "instrument": run.instrument.key,
        "asset_class": run.instrument.asset_class.value,
        "aggression": run.settings.aggression.value,
        "starting_cash": str(run.starting_cash),
        "final_equity": run.result.final_equity,
        "n_bars": run.n_bars,
        "start": run.result.start.isoformat(),
        "end": run.result.end.isoformat(),
        "n_orders": run.result.n_orders,
        "n_rejected": run.result.n_rejected,
        "rejections": run.result.rejections,
        "fees_paid": run.result.fees_paid,
        "turnover": run.result.turnover,
        "costs": {
            "spread_bps": str(run.costs.spread_bps),
            "taker_fee": str(run.costs.taker_fee),
            "slippage_base_bps": str(run.costs.slippage.base_bps),
            "slippage_impact": str(run.costs.slippage.impact_coefficient),
            "periods_per_year": run.costs.periods_per_year,
        },
        "scorecard": None,
    }
    if card is not None:
        card_dict = dict(card.as_dict())
        for key in ("start", "end"):
            value = card_dict.get(key)
            if isinstance(value, datetime):
                card_dict[key] = value.isoformat()
        payload["scorecard"] = card_dict
    if run.walk_forward:
        payload["walk_forward"] = {
            "n_folds": run.walk_forward.n_folds,
            "folds": [
                {
                    "start": f.start.isoformat(), "end": f.end.isoformat(),
                    "total_return": f.scorecard.total_return if f.scorecard else None,
                    "benchmark_return": (
                        f.scorecard.benchmark_return if f.scorecard else None
                    ),
                    "max_drawdown": f.scorecard.max_drawdown if f.scorecard else None,
                    "sharpe": f.scorecard.sharpe if f.scorecard else None,
                    "n_orders": f.n_orders,
                }
                for f in run.walk_forward.folds
            ],
        }

    card_path = target / f"{stem}_scorecard.json"
    card_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"equity": curve_path, "scorecard": card_path}


__all__ = [
    "BAR_CSV_FIELDS", "BacktestRun", "CRYPTO_COSTS", "CostModel",
    "EQUITY_COSTS", "backtest_instrument", "backtest_settings", "costs_for",
    "export_run", "format_report", "load_bars_from_csv", "load_bars_from_store",
    "run_backtest", "write_bars_csv",
]
