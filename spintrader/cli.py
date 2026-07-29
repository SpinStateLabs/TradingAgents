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
import sys
from decimal import Decimal

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

    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file(".env")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
