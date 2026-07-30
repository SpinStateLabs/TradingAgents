#!/usr/bin/env python3
"""Fetch daily bars to CSV without a database or an API key.

Why this exists alongside ``spintrader.data.yfinance_feed``: that module writes
into TimescaleDB, which is unreachable from a machine outside the LAN. A
backtest that can only be reproduced next to the database cannot be reviewed by
anyone else. This writes the same bar shape to a flat file that
``spintrader.backtest.runner.load_bars_from_csv`` reads.

The bar convention matches ``yfinance_feed.fetch_daily``: prices are raw
(unadjusted), and each session is stamped at 21:00:00 UTC, the US cash close.

    python scripts/fetch_bars_csv.py SPY --years 12 --out data/SPY_1d.csv

Unadjusted prices mean the series excludes dividends. For SPY that understates
total return by roughly 1.3%/yr. Both the strategy and its buy-and-hold
benchmark are computed from the same series, so the *comparison* is unaffected;
only the absolute CAGR of each is understated. Do not quote the absolute figure
as a total return.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
# Yahoo rejects the default urllib agent.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
CLOSE_HOUR_UTC = 21


class FetchError(RuntimeError):
    pass


def fetch(symbol: str, years: int, interval: str = "1d") -> list[dict[str, object]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(365.25 * years) + 5)
    query = (
        f"?period1={int(start.timestamp())}&period2={int(end.timestamp())}"
        f"&interval={interval}&includeAdjustedClose=true"
    )
    url = CHART_URL.format(symbol=symbol) + query

    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{symbol}: HTTP {exc.code} from Yahoo") from exc

    error = (payload.get("chart") or {}).get("error")
    if error:
        raise FetchError(f"{symbol}: {error}")
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        raise FetchError(f"{symbol}: empty result")

    block = results[0]
    stamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    opens, highs = quote.get("open") or [], quote.get("high") or []
    lows, closes = quote.get("low") or [], quote.get("close") or []
    volumes = quote.get("volume") or []

    rows: list[dict[str, object]] = []
    skipped = 0
    for i, epoch in enumerate(stamps):
        try:
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            v = volumes[i]
        except IndexError:
            skipped += 1
            continue
        if None in (o, h, l, c):
            skipped += 1
            continue
        session = datetime.fromtimestamp(epoch, tz=timezone.utc).date()
        ts = datetime(
            session.year, session.month, session.day,
            CLOSE_HOUR_UTC, 0, 0, tzinfo=timezone.utc,
        )
        # The schema enforces high >= open, close and low <= open, close. Yahoo
        # occasionally returns rows that violate this by a rounding cent; widen
        # rather than drop, so a real bar is not silently lost.
        hi = max(float(h), float(o), float(c))
        lo = min(float(l), float(o), float(c))
        rows.append({
            "ts": ts.isoformat(),
            "open": f"{float(o):.6f}",
            "high": f"{hi:.6f}",
            "low": f"{lo:.6f}",
            "close": f"{float(c):.6f}",
            "volume": f"{float(v or 0):.0f}",
        })

    if skipped:
        print(f"  skipped {skipped} malformed row(s)", file=sys.stderr)
    if not rows:
        raise FetchError(f"{symbol}: no usable bars")
    rows.sort(key=lambda r: r["ts"])
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ts", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol")
    parser.add_argument("--years", type=int, default=12)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    try:
        rows = fetch(args.symbol, args.years, args.interval)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out or f"data/{args.symbol}_{args.interval}.csv")
    write_csv(rows, out)
    print(
        f"{args.symbol}: {len(rows)} bars  "
        f"{rows[0]['ts'][:10]} -> {rows[-1]['ts'][:10]}  -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
