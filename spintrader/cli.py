"""SpinTrader command line.

    python -m spintrader.cli treasury convert --from CAD --to USD --amount all
    python -m spintrader.cli balances
    python -m spintrader.cli quote BTC-USD

Treasury operations are currency conversions that move the book's capital into
the currency strategies actually trade in. They are deliberately *not* part of
the strategy layer -- there is no thesis, no edge and no sizing decision, so
routing them through the signal path would pollute performance attribution
with trades no agent asked for.

Every order-placing command defaults to **validate only**. Kraken's
``validate`` flag makes the exchange parse and check the order and return its
interpretation without queueing anything, which exercises the entire path --
symbol resolution, rounding, risk checks, the live gate, authentication and
order construction -- with no market risk. Passing ``--execute`` is the
deliberate act that makes it real.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

from spintrader.core.config import (
    LiveTradingDisarmed, Settings, load_env_file,
)
from spintrader.core.types import (
    Order, OrderType, Side, TradingMode, to_decimal,
)
from spintrader.venues.base import OrderRejected, VenueError
from spintrader.venues.kraken import KrakenVenue

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


# --------------------------------------------------------------------------
# treasury convert
# --------------------------------------------------------------------------

def cmd_treasury_convert(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    venue = KrakenVenue(settings=settings)
    venue.connect()

    src, dst = args.source.upper(), args.dest.upper()

    # Kraken lists one direction of each fiat cross. Find whichever exists and
    # work out which side of it we are on, rather than assuming a direction.
    for symbol in (f"{dst}-{src}", f"{src}-{dst}"):
        try:
            instrument = venue.resolve(symbol)
            break
        except VenueError:
            continue
    else:
        print(f"{RED}kraken lists neither {dst}-{src} nor {src}-{dst}{RESET}")
        return 1

    # Buying the base spends the quote; selling the base receives it.
    buying_base = instrument.base_currency == dst
    side = Side.BUY if buying_base else Side.SELL

    snapshot = venue.snapshot()
    held = snapshot.cash(src)
    if held <= 0:
        print(f"{RED}no {src} balance to convert{RESET}")
        return 1

    amount = held if args.amount == "all" else to_decimal(args.amount)
    if amount > held:
        print(f"{RED}requested {amount} {src} but only {held} is held{RESET}")
        return 1

    quote = venue.get_quote(instrument)
    # Cross the spread in the direction that costs us: buying lifts the ask.
    rate = quote.ask if side is Side.BUY else quote.bid

    # Order volume is always denominated in the pair's BASE currency.
    volume = amount / rate if buying_base else amount
    received = volume if buying_base else volume * rate

    print("=" * 70)
    print(f"{BOLD}Treasury conversion{RESET}")
    print("=" * 70)
    print(f"  pair           {instrument.symbol}  (kraken: {instrument.venue_symbol})")
    print(f"  side           {side.value.upper()} {instrument.base_currency}")
    print(f"  rate           {rate}  ({'ask' if side is Side.BUY else 'bid'})")
    print(f"  spread         {quote.spread_bps:.1f} bps")
    print(f"  spending       {amount:,.2f} {src}")
    print(f"  receiving      ~{received:,.2f} {dst} (before fees)")
    fee = received * instrument.taker_fee
    print(f"  fee (taker)    ~{fee:,.2f} {dst}  ({instrument.taker_fee * 100:.2f}%)")
    print(f"  net            ~{received - fee:,.2f} {dst}")
    print()

    order = Order(
        instrument=instrument,
        side=side,
        qty=volume,
        order_type=OrderType.LIMIT,
        limit_price=rate,
        strategy="treasury",
    )

    if not args.execute:
        # Validate-only: proves the whole path without placing anything.
        try:
            venue._normalise(order)
            venue._check_tradability(order)
            result = venue.validate_order(order)
        except (OrderRejected, VenueError) as exc:
            print(f"  {RED}VALIDATION FAILED{RESET}: {exc}")
            return 1
        parsed = result.get("descr", {}).get("order", "(no description returned)")
        print(f"  {GREEN}VALIDATED{RESET} — kraken parsed the order and placed NOTHING:")
        print(f"    {parsed}")
        print()
        print(f"{YELLOW}  Nothing was executed. To place this order for real:{RESET}")
        print(f"    python -m spintrader.cli treasury convert "
              f"--from {src} --to {dst} --amount {args.amount} --execute")
        print("=" * 70)
        return 0

    # --- real execution path ---------------------------------------------
    if settings.mode is not TradingMode.LIVE:
        print(f"{RED}--execute requires SPINTRADER_MODE=live (currently "
              f"{settings.mode.value}){RESET}")
        return 1

    try:
        placed = venue.submit(order)
    except LiveTradingDisarmed as exc:
        print(f"{RED}live gate refused the order{RESET}: {exc}")
        print(f"{DIM}  Arm with SPINTRADER_LIVE_ENABLED=true and "
              f"SPINTRADER_LIVE_VENUES=kraken{RESET}")
        return 1
    except (OrderRejected, VenueError) as exc:
        print(f"{RED}order rejected{RESET}: {exc}")
        return 1

    print(f"  {GREEN}SUBMITTED{RESET} — venue order id {placed.venue_order_id}")
    print("=" * 70)
    return 0


# --------------------------------------------------------------------------
# balances / quote
# --------------------------------------------------------------------------

def cmd_balances(_args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    venue = KrakenVenue(settings=settings)
    venue.connect()
    snap = venue.snapshot()

    print("=" * 62)
    print(f"{BOLD}Kraken balances{RESET}")
    print("=" * 62)
    print(f"  {'ASSET':<10}{'AMOUNT':>22}{'MARK':>14}{'VALUE':>14}")
    print("-" * 62)
    for ccy, bal in sorted(snap.balances.items()):
        print(f"  {ccy:<10}{bal.total:>22,.8f}{'':>14}{'':>14}")
    for key, pos in sorted(snap.positions.items()):
        symbol = key.split(":", 1)[-1].split("-", 1)[0]
        mark = pos.last_price
        value = pos.market_value() if mark else None
        print(f"  {symbol:<10}{pos.qty:>22,.8f}"
              f"{(f'{mark:,.2f}' if mark else 'n/a'):>14}"
              f"{(f'{value:,.2f}' if value is not None else 'n/a'):>14}")
    print("-" * 62)
    print(f"  {'EQUITY':<10}{'':>22}{'':>14}"
          f"{snap.equity:>14,.2f} {settings.base_currency}")
    print("=" * 62)
    return 0


def cmd_data_backfill(args: argparse.Namespace) -> int:
    """Ingest history for the configured universe.

    Two sources with distinct roles: yfinance supplies deep daily history for
    model fitting, Kraken supplies recent intraday bars at the prices we will
    actually trade against. Kraken's 720-bar ceiling is reported per symbol so
    a thin result is visibly an API limit rather than a missing asset.
    """
    from spintrader.data import kraken_feed, yfinance_feed
    from spintrader.data.store import Store
    from spintrader.venues.kraken import KrakenVenue

    settings = Settings.from_env()
    store = Store(settings.storage)
    store.connect()
    if args.migrate:
        store.migrate()

    venue = KrakenVenue(settings=settings)
    venue.connect()

    crypto = list(args.symbols) or list(settings.crypto_universe)
    equities = [] if args.symbols else list(settings.equity_universe)

    print("=" * 78)
    print(f"{BOLD}Backfill{RESET}  crypto={len(crypto)}  equities={len(equities)}  "
          f"years={args.years}")
    print("=" * 78)
    print(f"  {'SYMBOL':<10}{'SOURCE':<10}{'INT':<5}{'NEW':>7}{'TOTAL':>8}  RANGE")
    print("-" * 78)

    failures = 0

    def report(symbol, source, interval, result):
        span = (f"{result['first']:%Y-%m-%d} .. {result['last']:%Y-%m-%d}"
                if result.get("first") else "empty")
        flag = f"  {YELLOW}[api limit]{RESET}" if result.get("reached_api_limit") else ""
        print(f"  {symbol:<10}{source:<10}{interval:<5}"
              f"{result['written']:>7}{result['bars']:>8}  {span}{flag}")

    for symbol in crypto:
        try:
            instrument = venue.resolve(symbol)
        except VenueError as exc:
            print(f"  {symbol:<10}{RED}unavailable{RESET}: {exc}")
            failures += 1
            continue
        try:
            report(symbol, "yfinance", "1d",
                   yfinance_feed.backfill(store, instrument, years=args.years))
        except Exception as exc:                        # noqa: BLE001 - reported
            print(f"  {symbol:<10}{'yfinance':<10}{RED}failed{RESET}: {str(exc)[:40]}")
            failures += 1
        try:
            report(symbol, "kraken", args.interval,
                   kraken_feed.backfill(store, venue, symbol, interval=args.interval))
        except Exception as exc:                        # noqa: BLE001 - reported
            print(f"  {symbol:<10}{'kraken':<10}{RED}failed{RESET}: {str(exc)[:40]}")
            failures += 1

    # Equities come from yfinance only: IBKR historical data needs market-data
    # subscriptions the account deliberately does not have.
    for symbol in equities:
        from spintrader.core.types import AssetClass, Instrument, VenueId
        instrument = Instrument(
            symbol=symbol, asset_class=AssetClass.EQUITY, venue=VenueId.IBKR,
            venue_symbol=symbol, quote_currency="USD",
            price_increment=Decimal("0.01"), qty_increment=Decimal("0.0001"),
            min_notional=Decimal("1"),
        )
        try:
            report(symbol, "yfinance", "1d",
                   yfinance_feed.backfill(store, instrument, years=args.years))
        except Exception as exc:                        # noqa: BLE001 - reported
            print(f"  {symbol:<10}{'yfinance':<10}{RED}failed{RESET}: {str(exc)[:40]}")
            failures += 1

    store.close()
    print("=" * 78)
    print(f"{'RESULT: PASS' if not failures else f'RESULT: {failures} failure(s)'}")
    return 1 if failures else 0


def cmd_data_backfill_1m(args: argparse.Namespace) -> int:
    """Backfill deep 1-minute history from Kraken's /Trades endpoint.

    The OHLC endpoint reaches back only 720 bars and the WebSocket collector
    only goes forward, so neither can recover deep minute history. This pages
    /Trades forward from the requested start (or the last stored bar) toward the
    present, aggregating trades into complete 1-minute bars. It is long running
    -- 1000 trades is roughly an hour of history -- and resumable: re-running
    continues from where it stopped. The forming final minute is never written;
    the WebSocket collector owns the live edge.
    """
    from datetime import datetime, timezone

    from spintrader.data.kraken_trades import backfill_1m
    from spintrader.data.store import Store

    settings = Settings.from_env()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    store = Store(settings.storage)
    store.connect()
    if args.migrate:
        store.migrate()

    venue = KrakenVenue(settings=settings)
    venue.connect()

    symbols = list(args.symbols) or list(settings.crypto_universe)

    start: object
    if args.since:
        start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) \
            if "T" not in args.since and "+" not in args.since \
            else datetime.fromisoformat(args.since.replace("Z", "+00:00"))
    elif args.years is not None and not args.resume_only:
        start = datetime.now(timezone.utc) - timedelta(days=int(365.25 * args.years))
    else:
        start = None    # resume from the last stored bar, else genesis
    end = (datetime.fromisoformat(args.end.replace("Z", "+00:00"))
           if args.end else None)

    print("=" * 78)
    print(f"{BOLD}1m trades backfill{RESET}  symbols={len(symbols)}  "
          f"start={start if start is not None else 'resume/genesis'}  "
          f"sleep={args.sleep}s")
    print("=" * 78)

    failures = 0
    for symbol in symbols:
        last_report = {"pages": 0}

        def progress(info):
            last_report.update(info)
            print(f"  {symbol:<10} page {info['pages']:>5}  "
                  f"written {info['written']:>8}  "
                  f"at {info['cursor_ts']:%Y-%m-%d %H:%M}", end="\r", flush=True)

        try:
            result = backfill_1m(
                store, venue, symbol,
                start=start, end=end,
                session=venue._session,
                max_pages=args.max_pages,
                sleep_s=args.sleep,
                resume=not args.no_resume,
                progress=progress,
            )
        except Exception as exc:                        # noqa: BLE001 - reported
            print(f"\n  {symbol:<10}{RED}failed{RESET}: {str(exc)[:60]}")
            failures += 1
            continue

        span = (f"{result['first']:%Y-%m-%d %H:%M} .. {result['last']:%Y-%m-%d %H:%M}"
                if result.get("first") else "empty")
        edge = f"  {GREEN}[live edge]{RESET}" if result.get("reached_live_edge") else \
               (f"  {YELLOW}[end]{RESET}" if result.get("hit_end") else
                f"  {YELLOW}[max-pages]{RESET}")
        print(f"\n  {GREEN}{symbol}{RESET}: {result['pages']} pages, "
              f"{result['written']} bars written, {result['bars']} total  "
              f"{span}{edge}")

    store.close()
    print("=" * 78)
    print(f"{'RESULT: PASS' if not failures else f'RESULT: {failures} failure(s)'}")
    return 1 if failures else 0


def cmd_data_collect(args: argparse.Namespace) -> int:
    """Run the WebSocket minute collector until interrupted.

    Intended to run continuously as a service. Kraken's REST OHLC endpoint
    only reaches back 720 bars — twelve hours at 1-minute resolution — so
    minute history can only be accumulated forward. Downtime is permanent
    data loss, not a delay.
    """
    import asyncio
    import logging

    from spintrader.data.kraken_ws import collect
    from spintrader.data.store import Store

    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    store = Store(settings.storage)
    store.connect()
    symbols = list(args.symbols) or list(settings.crypto_universe)
    print(f"collecting {args.interval}m bars for {', '.join(symbols)}")

    try:
        stats = asyncio.run(collect(symbols, store, interval_minutes=args.interval))
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 0
    finally:
        store.close()

    print(f"stats: {stats.as_dict()}")
    return 0


def cmd_quote(args: argparse.Namespace) -> int:
    venue = KrakenVenue(settings=Settings.from_env())
    venue.connect()
    instrument = venue.resolve(args.symbol)
    quote = venue.get_quote(instrument)
    print(f"{instrument.symbol}  bid {quote.bid}  ask {quote.ask}  "
          f"mid {quote.mid}  spread {quote.spread_bps:.1f} bps")
    return 0


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

STRATEGIES = {
    "baseline_trend": "spintrader.agents.personas.baseline_trend:BaselineTrendAgent",
    "mean_reversion": "spintrader.agents.personas.mean_reversion:MeanReversionAgent",
    "markov_chain": "spintrader.agents.personas.markov_chain:HighOrderMarkovAgent",
    "regime_switch": "spintrader.agents.personas.regime_switch:RegimeSwitchingAgent",
    "hedge_ensemble": "spintrader.agents.personas.hedge:HedgeEnsembleAgent",
}


def _load_strategy(name: str):
    """Resolve a registered persona, or a ``module:Class`` path."""
    target = STRATEGIES.get(name, name)
    if ":" not in target:
        raise SystemExit(
            f"unknown strategy {name!r}; registered: "
            f"{', '.join(sorted(STRATEGIES))}, or pass module:Class"
        )
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise SystemExit(f"{module_name} has no attribute {class_name}") from None


def cmd_backtest(args: argparse.Namespace) -> int:
    """Replay a persona through the paper venue and score it."""
    from spintrader.backtest.runner import (
        backtest_instrument, export_run, format_report,
        load_bars_from_csv, load_bars_from_store, run_backtest, write_bars_csv,
    )
    from spintrader.core.types import AssetClass

    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level)

    asset_class = AssetClass(args.asset_class)
    instrument = backtest_instrument(args.symbol, asset_class)

    if args.csv:
        bars = load_bars_from_csv(args.csv, instrument.key, args.interval)
        source = args.csv
    else:
        # The store keys bars by the *ingesting* venue, not the paper venue the
        # replay trades on, so the lookup key is built from the source venue.
        source_venue = "kraken" if asset_class is AssetClass.CRYPTO else "ibkr"
        store_key = f"{source_venue}:{args.symbol}"
        try:
            bars = load_bars_from_store(
                store_key, args.interval, settings=settings,
            )
        except Exception as exc:                        # noqa: BLE001 - reported
            print(f"could not read bars from the store: {exc}", file=sys.stderr)
            print(
                "pass --csv to run without the database "
                "(see scripts/fetch_bars_csv.py)", file=sys.stderr,
            )
            return 1
        source = store_key
        # Re-key onto the paper venue so the replay's instrument matches.
        bars = [replace(bar, instrument_key=instrument.key) for bar in bars]

    if len(bars) < 2:
        print(f"{source}: not enough bars ({len(bars)})", file=sys.stderr)
        return 1

    print(
        f"{source}: {len(bars)} {args.interval} bars  "
        f"{bars[0].ts:%Y-%m-%d} -> {bars[-1].ts:%Y-%m-%d}"
    )

    strategy_cls = _load_strategy(args.strategy)
    run = run_backtest(
        strategy_cls, args.symbol, bars,
        asset_class=asset_class,
        aggression=args.aggression or settings.aggression,
        starting_cash=args.cash,
        enforce_cash_account=not args.allow_margin,
        n_trials=args.trials,
        walk_forward=not args.no_walk_forward,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
    )
    print(format_report(run))

    if args.out:
        paths = export_run(run, args.out)
        if not args.csv:
            paths["bars"] = write_bars_csv(
                bars, Path(args.out) / f"{args.symbol}_{args.interval}.csv"
            )
        for label, path in paths.items():
            print(f"wrote {label}: {path}")

    card = run.result.scorecard
    if card is None:
        return 1
    # A non-zero exit on an insignificant result keeps a promotion script from
    # shipping noise. When a walk-forward ran, its stitched out-of-sample result
    # is the verdict of record: a full-sample DSR can read significant off a single
    # lucky fold (the BNB case), so the exit code follows the walk-forward, not the
    # in-sample scorecard. Fall back to full-sample only when no WF was run.
    wf = run.walk_forward
    decisive = wf.combined if (wf is not None and wf.combined is not None) else card
    return 0 if decisive.is_significant else 2


def _build_mandate_service(settings: Settings, use_llm: bool):
    """The slow loop's deliberation service: LLM panel or the quant bootstrap."""
    from spintrader.agents.personas.roster import default_roster
    from spintrader.loop.mandate import MandateService

    if use_llm:
        from spintrader.llm.router import LLMRouter
        from spintrader.loop.voting import LLMVoter
        voter = LLMVoter(LLMRouter(settings.llm))
    else:
        from spintrader.loop.voting import BootstrapVoter
        voter = BootstrapVoter()
    return MandateService(default_roster(), voter=voter)


def cmd_loop_mandate(args: argparse.Namespace) -> int:
    """Run one deliberation and print the mandate it would issue. No trading."""
    from spintrader.loop.decision_loop import build_paper_loop

    settings = Settings.from_env()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    symbols = list(args.symbols) or list(settings.crypto_universe)

    service = _build_mandate_service(settings, args.llm)
    loop = build_paper_loop(symbols, settings=settings, mandate_service=service,
                            with_regime=args.with_regime)
    mandate = loop.refresh_mandate()
    delib = loop.deliberation

    print("=" * 74)
    print(f"{BOLD}Deliberation{RESET}  ({'LLM panel' if args.llm else 'quant bootstrap'})")
    print("=" * 74)
    print(f"  {delib.summary()}")
    print("-" * 74)
    for key, verdict in sorted(delib.verdicts.items()):
        permitted = key in mandate.permitted
        flag = f"{GREEN}permitted{RESET}" if permitted else f"{DIM}not permitted{RESET}"
        print(f"  {key:<18}{verdict.summary()}  [{flag}]")
    print("=" * 74)
    return 0


def cmd_loop_run(args: argparse.Namespace) -> int:
    """Run the two-tier decision loop in PAPER mode until interrupted.

    Fast quant loop every minute; slow LLM (or bootstrap) loop hourly emitting
    an expiring Mandate. This helper is paper-only by construction -- arming live
    trading is the separate, deliberate act of flipping SPINTRADER_LIVE_ENABLED,
    populating SPINTRADER_LIVE_VENUES and clearing READ_ONLY_API, which no code
    path here performs.
    """
    from spintrader.loop.decision_loop import build_paper_loop, ExecutionStyle

    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    symbols = list(args.symbols) or list(settings.crypto_universe)

    service = _build_mandate_service(settings, args.llm)
    execution = ExecutionStyle.MAKER_FIRST if args.maker_first else ExecutionStyle.TAKER
    loop = build_paper_loop(symbols, settings=settings, mandate_service=service,
                            with_regime=args.with_regime, execution=execution)

    print("=" * 74)
    print(f"{BOLD}Decision loop{RESET}  mode=PAPER  symbols={', '.join(symbols)}")
    print(f"  mandate source: {'LLM panel' if args.llm else 'quant bootstrap'}  "
          f"regime={'on' if args.with_regime else 'off'}  "
          f"execution={execution.value}")
    print(f"  fast {args.fast_interval}s / slow {args.slow_interval}s"
          + (f" / max {args.max_ticks} ticks" if args.max_ticks else ""))
    print("=" * 74)

    try:
        out = loop.run(
            fast_interval_s=args.fast_interval,
            slow_interval_s=args.slow_interval,
            max_ticks=args.max_ticks,
        )
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 0
    print(f"stopped after {out['ticks']} ticks; {out['mandate']}")
    return 0


def cmd_improve(args: argparse.Namespace) -> int:
    """Run one improvement round: generate candidates, prove them, promote the best.

    Every candidate is counted as a lifetime trial before it can be promoted, and
    the deflated Sharpe rises with that count -- so this is deliberately hard to
    pass, and 'nothing promoted' is the common, correct outcome for a weak
    strategy family. The research memory is persisted (when reading from the
    store) so repeated runs continue the search rather than re-testing.
    """
    from dataclasses import replace as _replace

    from spintrader.backtest.runner import (
        backtest_instrument, load_bars_from_csv, load_bars_from_store,
    )
    from spintrader.core.types import AssetClass
    from spintrader.loop.improvement import ImprovementCycle
    from spintrader.loop.promotion import PromotionGate, TrialLedger
    from spintrader.research.factory import CandidateFactory, default_families
    from spintrader.research.memory import ResearchMemory

    settings = Settings.from_env()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    asset_class = AssetClass(args.asset_class)
    instrument = backtest_instrument(args.symbol, asset_class)

    store = None
    if args.csv:
        bars = load_bars_from_csv(args.csv, instrument.key, args.interval)
        source = args.csv
    else:
        source_venue = "kraken" if asset_class is AssetClass.CRYPTO else "ibkr"
        store_key = f"{source_venue}:{args.symbol}"
        try:
            bars = load_bars_from_store(store_key, args.interval, settings=settings)
        except Exception as exc:                        # noqa: BLE001 - reported
            print(f"could not read bars from the store: {exc}", file=sys.stderr)
            print("pass --csv to run without the database", file=sys.stderr)
            return 1
        bars = [_replace(b, instrument_key=instrument.key) for b in bars]
        source = store_key
        from spintrader.data.store import Store
        store = Store(settings.storage)
        store.connect()

    if len(bars) < 2:
        print(f"{source}: not enough bars ({len(bars)})", file=sys.stderr)
        return 1

    objective = args.objective or f"{args.symbol}_{args.interval}"
    gate = PromotionGate(ledger=TrialLedger())
    memory = ResearchMemory(store=store)
    loaded = memory.load(objective)
    factory = CandidateFactory(families=default_families())    # trend + mean reversion
    cycle = ImprovementCycle(gate=gate, memory=memory, factory=factory)

    print("=" * 74)
    print(f"{BOLD}Improvement round{RESET}  objective '{objective}'  "
          f"({len(bars)} {args.interval} bars, {loaded} prior trials loaded)")
    print("=" * 74)
    result = cycle.run_round(
        objective, args.symbol, bars, asset_class=asset_class,
        aggression=args.aggression or settings.aggression,
        starting_cash=args.cash, n_candidates=args.max_candidates,
        train_size=args.train_size, test_size=args.test_size,
        embargo=args.embargo,
    )
    print(f"  {result.summary()}")
    print("-" * 74)
    for verdict in result.verdicts:
        colour = GREEN if verdict.promoted else DIM
        print(f"  {colour}{verdict.summary()[:118]}{RESET}")
    print("-" * 74)
    summary = memory.summary(objective)
    print(f"  lifetime: {summary['evaluated']} evaluated, {summary['promoted']} promoted; "
          f"rejections {summary['rejections']}")
    print("=" * 74)

    if store is not None:
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spintrader", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    treasury = sub.add_parser("treasury", help="capital and currency operations")
    tsub = treasury.add_subparsers(dest="treasury_command", required=True)

    convert = tsub.add_parser(
        "convert", help="convert one currency to another (validate-only by default)"
    )
    convert.add_argument("--from", dest="source", required=True, help="currency to spend")
    convert.add_argument("--to", dest="dest", required=True, help="currency to receive")
    convert.add_argument("--amount", default="all", help="amount to convert, or 'all'")
    convert.add_argument(
        "--execute", action="store_true",
        help="actually place the order (requires live mode and an armed gate). "
             "Without this flag the order is validated by the exchange and discarded.",
    )
    convert.set_defaults(func=cmd_treasury_convert)

    balances = sub.add_parser("balances", help="show venue balances and equity")
    balances.set_defaults(func=cmd_balances)

    quote = sub.add_parser("quote", help="show top of book for a symbol")
    quote.add_argument("symbol")
    quote.set_defaults(func=cmd_quote)

    data = sub.add_parser("data", help="market data ingestion")
    dsub = data.add_subparsers(dest="data_command", required=True)
    backfill = dsub.add_parser("backfill", help="ingest history for the universe")
    backfill.add_argument("symbols", nargs="*", help="symbols (default: configured universe)")
    backfill.add_argument("--years", type=int, default=5, help="years of daily history")
    backfill.add_argument("--interval", default="1h", help="kraken intraday interval")
    backfill.add_argument("--migrate", action="store_true", help="apply the schema first")
    backfill.set_defaults(func=cmd_data_backfill)

    collect = dsub.add_parser("collect", help="stream and store live bars (long running)")
    collect.add_argument("symbols", nargs="*", help="symbols (default: crypto universe)")
    collect.add_argument("--interval", type=int, default=1, help="bar interval in minutes")
    collect.set_defaults(func=cmd_data_collect)

    backfill_1m = dsub.add_parser(
        "backfill-1m",
        help="backfill deep 1m history from Kraken /Trades (long running, resumable)",
        description="Pages Kraken's public Trades endpoint forward and stores "
                    "aggregated 1-minute bars. Unlike the OHLC endpoint (720 bars) "
                    "and the WebSocket collector (forward only), this reaches back "
                    "years. Resumable: re-run to continue from the last stored bar.",
    )
    backfill_1m.add_argument("symbols", nargs="*", help="symbols (default: crypto universe)")
    backfill_1m.add_argument(
        "--years", type=float, default=2.0,
        help="how far back to start when there is nothing to resume from (default 2)",
    )
    backfill_1m.add_argument(
        "--since", default=None,
        help="explicit start date (YYYY-MM-DD or ISO-8601); overrides --years",
    )
    backfill_1m.add_argument(
        "--end", default=None, help="stop at this date (YYYY-MM-DD or ISO-8601)",
    )
    backfill_1m.add_argument(
        "--resume-only", action="store_true",
        help="ignore --years; resume from the last stored bar, else genesis",
    )
    backfill_1m.add_argument(
        "--no-resume", action="store_true",
        help="do not resume from stored bars; use --since/--years as the start",
    )
    backfill_1m.add_argument("--max-pages", type=int, default=None, help="cap the page count")
    backfill_1m.add_argument(
        "--sleep", type=float, default=1.0, help="seconds between calls (rate limit)",
    )
    backfill_1m.add_argument("--migrate", action="store_true", help="apply the schema first")
    backfill_1m.set_defaults(func=cmd_data_backfill_1m)

    backtest = sub.add_parser(
        "backtest",
        help="replay a strategy through the paper venue and score it",
        description="Replays historical bars through the same PaperVenue, "
                    "RiskEngine and Ledger that paper and live trading use. "
                    "Exit code 2 means the result is not statistically "
                    "significant after the multiple-testing adjustment.",
    )
    backtest.add_argument("symbol", help="e.g. SPY or BTC-USD")
    backtest.add_argument(
        "--strategy", default="baseline_trend",
        help=f"registered persona ({', '.join(sorted(STRATEGIES))}) "
             f"or a module:Class path",
    )
    backtest.add_argument(
        "--asset-class", default="equity", choices=["equity", "crypto", "etf"],
        help="selects the cost model and the annualisation factor",
    )
    backtest.add_argument("--interval", default="1d", help="bar interval")
    backtest.add_argument(
        "--csv", default=None,
        help="read bars from a CSV instead of the store "
             "(see scripts/fetch_bars_csv.py)",
    )
    backtest.add_argument("--cash", default="1000", help="starting capital")
    backtest.add_argument(
        "--aggression", default=None,
        choices=["conservative", "moderate", "balanced", "growth", "aggressive"],
        help="risk profile (default: SPINTRADER_AGGRESSION)",
    )
    backtest.add_argument(
        "--allow-margin", action="store_true",
        help="disable cash-account rules (T+1 settlement and the short block)",
    )
    backtest.add_argument(
        "--trials", type=int, default=1,
        help="how many variants were tried to arrive at this one. Setting this "
             "honestly is what keeps the deflated Sharpe meaningful; leaving it "
             "at 1 after a parameter sweep is how a search gets reported as a "
             "discovery.",
    )
    backtest.add_argument(
        "--no-walk-forward", action="store_true", help="skip the fold analysis",
    )
    backtest.add_argument("--train-size", type=int, default=None)
    backtest.add_argument("--test-size", type=int, default=None)
    backtest.add_argument("--embargo", type=int, default=None)
    backtest.add_argument(
        "--out", default=None, help="directory for the equity curve and scorecard",
    )
    backtest.set_defaults(func=cmd_backtest)

    loop = sub.add_parser(
        "loop",
        help="the two-tier decision loop (fast quant + slow LLM mandate)",
    )
    lsub = loop.add_subparsers(dest="loop_command", required=True)

    loop_run = lsub.add_parser(
        "run",
        help="run the decision loop in PAPER mode (long running)",
        description="Fast quant loop every minute under a slow, expiring mandate "
                    "refreshed hourly. PAPER only; arming live trading is a "
                    "separate deliberate act and no code path here performs it.",
    )
    loop_run.add_argument("symbols", nargs="*", help="symbols (default: crypto universe)")
    loop_run.add_argument("--llm", action="store_true",
                          help="use the LLM persona panel (default: quant bootstrap)")
    loop_run.add_argument("--with-regime", action="store_true",
                          help="fit the HMM regime model (needs hmmlearn)")
    loop_run.add_argument("--maker-first", action="store_true",
                          help="post passive limit orders to earn the maker fee, "
                               "escalating to taker on timeout (exits stay taker)")
    loop_run.add_argument("--fast-interval", type=float, default=60.0,
                          help="seconds between fast ticks")
    loop_run.add_argument("--slow-interval", type=float, default=3600.0,
                          help="seconds between mandate refreshes")
    loop_run.add_argument("--max-ticks", type=int, default=None,
                          help="stop after this many fast ticks (default: run forever)")
    loop_run.set_defaults(func=cmd_loop_run)

    loop_mandate = lsub.add_parser(
        "mandate", help="run one deliberation and print the mandate (no trading)",
    )
    loop_mandate.add_argument("symbols", nargs="*", help="symbols (default: crypto universe)")
    loop_mandate.add_argument("--llm", action="store_true",
                              help="use the LLM persona panel (default: quant bootstrap)")
    loop_mandate.add_argument("--with-regime", action="store_true",
                              help="fit the HMM regime model (needs hmmlearn)")
    loop_mandate.set_defaults(func=cmd_loop_mandate)

    improve = sub.add_parser(
        "improve",
        help="run one self-improvement round (generate, prove, promote)",
        description="Generates candidate strategy configurations, evaluates each "
                    "through the walk-forward backtester and the promotion gate "
                    "(counting every one as a lifetime trial), and promotes the "
                    "best that clears the gate. Passing rarely -- the deflated "
                    "Sharpe rises with the trial count -- is the point.",
    )
    improve.add_argument("symbol", help="e.g. SPY or BTC-USD")
    improve.add_argument(
        "--asset-class", default="crypto", choices=["equity", "crypto", "etf"],
        help="selects the cost model and annualisation factor",
    )
    improve.add_argument("--interval", default="1m", help="bar interval")
    improve.add_argument(
        "--csv", default=None,
        help="read bars from a CSV instead of the store (no memory persistence)",
    )
    improve.add_argument("--cash", default="1000", help="starting capital")
    improve.add_argument(
        "--aggression", default=None,
        choices=["conservative", "moderate", "balanced", "growth", "aggressive"],
    )
    improve.add_argument(
        "--objective", default=None,
        help="trial-accounting bucket (default: <symbol>_<interval>)",
    )
    improve.add_argument(
        "--max-candidates", type=int, default=None,
        help="cap candidates evaluated this round (default: all fresh grid points)",
    )
    improve.add_argument(
        "--train-size", type=int, default=None,
        help="walk-forward train window in bars (default: auto-sized to the series)",
    )
    improve.add_argument(
        "--test-size", type=int, default=None,
        help="walk-forward test window in bars; smaller yields more folds -- needed "
             "to reach the 3-fold minimum on multi-year daily data",
    )
    improve.add_argument(
        "--embargo", type=int, default=None,
        help="bars skipped between train and test (default: the strategy warmup)",
    )
    improve.set_defaults(func=cmd_improve)

    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file(".env")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
