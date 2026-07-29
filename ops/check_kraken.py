"""Verify a Kraken API key: connectivity, permissions, and safety posture.

Run this immediately after creating the key, before wiring anything else up:

    python ops/check_kraken.py

What it does
------------
Probes each permission the platform needs and reports pass/fail per capability,
so a missing checkbox is identified by name instead of surfacing later as an
opaque ``EGeneral:Permission denied`` mid-cycle.

Safety
------
This script never moves money and never places a real order.

* The trading check uses ``AddOrder`` with ``validate=true``, which asks Kraken
  to parse and validate the order and return its interpretation *without
  submitting it*. Nothing is queued, nothing is filled.
* The withdrawal check calls ``WithdrawMethods``, a read-only listing. The
  actual ``/0/private/Withdraw`` endpoint is never referenced anywhere in this
  repository.
* Getting ``Permission denied`` on the withdrawal probe is the desired result
  and is reported as a PASS.

Secrets are read from the environment (or ``.env``) and are never printed.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from typing import Any

import requests

from spintrader.core.config import load_env_file, env_str
from spintrader.venues.kraken_auth import KrakenAuthError, KrakenCredentials

API_BASE = "https://api.kraken.com"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"


def call_public(endpoint: str, params: dict | None = None) -> dict[str, Any]:
    url = f"{API_BASE}/0/public/{endpoint}"
    response = requests.get(url, params=params or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def call_private(creds: KrakenCredentials, endpoint: str, params: dict | None = None) -> dict[str, Any]:
    path = f"/0/private/{endpoint}"
    headers, body = creds.signed_request(path, params)
    response = requests.post(f"{API_BASE}{path}", headers=headers, data=body, timeout=30)
    response.raise_for_status()
    return response.json()


def errors_of(payload: dict[str, Any]) -> list[str]:
    return payload.get("error") or []


def denied(payload: dict[str, Any]) -> bool:
    return any("Permission denied" in e for e in errors_of(payload))


def main() -> int:
    loaded = load_env_file(".env")
    print("=" * 70)
    print("Kraken API key check")
    print("=" * 70)
    print(f"{DIM}  loaded {loaded} vars from .env{RESET}")

    # --- credentials present and well-formed -----------------------------
    try:
        creds = KrakenCredentials(
            env_str("KRAKEN_API_KEY", ""), env_str("KRAKEN_API_SECRET", "")
        )
    except KrakenAuthError as exc:
        print(f"  {FAIL} credentials: {exc}")
        print("\n  Copy .env.spintrader.example to .env and fill in the key pair.")
        return 1
    print(f"  {OK} credentials parse (key {creds.api_key[:6]}...)")

    # --- public connectivity ---------------------------------------------
    try:
        server_time = call_public("Time")
        print(f"  {OK} public API reachable "
              f"({server_time['result']['rfc1123']})")
    except requests.RequestException as exc:
        print(f"  {FAIL} public API unreachable: {exc}")
        return 1

    failures: list[str] = []
    warnings: list[str] = []

    # --- Query Funds ------------------------------------------------------
    balances: dict[str, str] = {}
    try:
        payload = call_private(creds, "Balance")
        if errors_of(payload):
            print(f"  {FAIL} Query Funds: {errors_of(payload)}")
            failures.append("Query Funds")
            if any("Invalid key" in e for e in errors_of(payload)):
                print(f"{DIM}         'Invalid key' means either the key/secret is wrong, or an "
                      f"IP restriction is blocking this machine.{RESET}")
        else:
            balances = {k: v for k, v in payload["result"].items()
                        if float(v) > 0}
            print(f"  {OK} Query Funds ({len(balances)} non-zero balances)")
    except requests.RequestException as exc:
        print(f"  {FAIL} Query Funds: {exc}")
        failures.append("Query Funds")

    # --- Query Open / Closed Orders --------------------------------------
    for endpoint, label in (("OpenOrders", "Query Open Orders & Trades"),
                            ("ClosedOrders", "Query Closed Orders & Trades")):
        try:
            payload = call_private(creds, endpoint)
            if errors_of(payload):
                print(f"  {FAIL} {label}: {errors_of(payload)}")
                failures.append(label)
            else:
                print(f"  {OK} {label}")
        except requests.RequestException as exc:
            print(f"  {FAIL} {label}: {exc}")
            failures.append(label)

    # --- WebSockets token -------------------------------------------------
    try:
        payload = call_private(creds, "GetWebSocketsToken")
        if errors_of(payload):
            print(f"  {WARN} Access WebSockets API: {errors_of(payload)}")
            warnings.append("Access WebSockets API (needed for live price streams)")
        else:
            print(f"  {OK} Access WebSockets API")
    except requests.RequestException as exc:
        print(f"  {WARN} Access WebSockets API: {exc}")
        warnings.append("Access WebSockets API")

    # --- Modify Orders, via a VALIDATE-ONLY order -------------------------
    # validate=true makes Kraken parse and check the order, then return its
    # interpretation without submitting it. No order is created.
    #
    # Priced far below market so it could never be marketable even if the
    # validate flag were somehow dropped, but sized so price x volume clears
    # Kraken's ~$5 minimum order cost -- otherwise the request is rejected on
    # economics before the permission is exercised.
    try:
        payload = call_private(creds, "AddOrder", {
            "pair": "XBTUSD",
            "type": "buy",
            "ordertype": "limit",
            "price": "20000",       # far below market
            "volume": "0.0003",     # -> $6 notional, above the cost minimum
            "validate": "true",     # <-- nothing is placed
        })
        errs = errors_of(payload)
        if any("Permission denied" in e for e in errs):
            print(f"  {FAIL} Modify Orders: permission not granted")
            print(f"{DIM}         Tick 'Modify Orders' under Orders & Trades on the key.{RESET}")
            failures.append("Modify Orders")
        elif any(e.startswith("EOrder:") for e in errs):
            # The request authenticated and reached order validation, which is
            # exactly what we are testing. A rejection on economics (cost
            # minimum, tick size, insufficient funds) still proves the
            # permission is present.
            print(f"  {OK} Modify Orders (reached order validation: {errs[0]})")
        elif errs:
            print(f"  {WARN} Modify Orders: unexpected response {errs}")
            warnings.append(f"AddOrder returned {errs}")
        else:
            desc = payload.get("result", {}).get("descr", {}).get("order", "validated")
            print(f"  {OK} Modify Orders (validate-only, nothing placed)")
            print(f"{DIM}         kraken parsed: {desc}{RESET}")
    except requests.RequestException as exc:
        print(f"  {FAIL} Modify Orders: {exc}")
        failures.append("Modify Orders")

    # --- Withdrawal MUST be denied ---------------------------------------
    # Read-only probe. Denial here is the correct, desired outcome.
    try:
        payload = call_private(creds, "WithdrawMethods")
        if denied(payload):
            print(f"  {OK} Withdraw Funds is NOT granted (correct)")
        elif errors_of(payload):
            print(f"  {WARN} withdrawal probe inconclusive: {errors_of(payload)}")
        else:
            print(f"  {RED}RISK{RESET} Withdraw Funds IS granted on this key.")
            print(f"{DIM}         This platform never withdraws. Revoke the key and create a "
                  f"new one with withdrawal unticked.{RESET}")
            warnings.append("withdrawal permission is enabled -- recreate the key")
    except requests.RequestException as exc:
        print(f"  {WARN} withdrawal probe failed: {exc}")

    # --- summary ----------------------------------------------------------
    print("-" * 70)
    if balances:
        print("  balances:")
        for asset, amount in sorted(balances.items()):
            print(f"      {asset:<10} {amount}")
    else:
        print(f"{DIM}  no non-zero balances (expected until you transfer funds in){RESET}")

    if failures:
        print(f"\n  {FAIL} missing permissions: {', '.join(failures)}")
        print("  Fix: Kraken Pro -> Settings -> API -> edit the key -> tick the boxes above.")
    for w in warnings:
        print(f"  {WARN} {w}")

    print("=" * 70)
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS -- key is usable for paper and micro-live trading")
    return 0


if __name__ == "__main__":
    sys.exit(main())
