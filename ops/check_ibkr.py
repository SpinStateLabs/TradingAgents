"""Verify the IBKR Gateway connection and account access.

    PYTHONPATH=. .venv/bin/python ops/check_ibkr.py

Confirms the Gateway is listening, the API accepts a client connection, the
account is reachable, and -- importantly -- reports whether the connection is
pointed at the paper or the live account. Places no orders.
"""

from __future__ import annotations

import socket
import sys

from spintrader.core.config import env_int, env_str, load_env_file

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"

# IBKR convention: paper account identifiers begin with D (DU/DF), live with U.
PAPER_PREFIXES = ("DU", "DF")


def main() -> int:
    load_env_file(".env")
    host = env_str("IBKR_HOST", "127.0.0.1")
    port = env_int("IBKR_PORT", 4002)
    client_id = env_int("IBKR_CLIENT_ID", 17)

    print("=" * 70)
    print("IBKR Gateway check")
    print("=" * 70)
    print(f"{DIM}  target {host}:{port} (client id {client_id}){RESET}")
    expected = "paper" if port in (4002, 7497) else "live"
    print(f"{DIM}  port {port} conventionally means the {expected} account{RESET}")
    print()

    # --- socket reachable -------------------------------------------------
    try:
        with socket.create_connection((host, port), timeout=10):
            print(f"  {OK} {host}:{port} is listening")
    except OSError as exc:
        print(f"  {FAIL} cannot reach {host}:{port} -- {exc}")
        print(f"{DIM}         Is the container up?   docker ps | grep spintrader-ibgw")
        print(f"         Still logging in?      docker logs --tail 50 spintrader-ibgw")
        print(f"         First login needs VNC: ssh -N -L 5900:127.0.0.1:5900 spinner@10.0.0.62{RESET}")
        return 1

    # --- API handshake ----------------------------------------------------
    try:
        from ib_async import IB
    except ImportError:
        print(f"  {FAIL} ib_async not installed")
        print(f"{DIM}         uv pip install --python .venv/bin/python ib_async{RESET}")
        return 1

    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=30, readonly=True)
    except Exception as exc:                            # noqa: BLE001 - reported
        print(f"  {FAIL} API handshake failed: {type(exc).__name__}: {exc}")
        print(f"{DIM}         In Gateway: Configure -> Settings -> API -> Settings,")
        print(f"         tick 'Enable ActiveX and Socket Clients'.")
        print(f"         A duplicate client id also fails here -- try another IBKR_CLIENT_ID.{RESET}")
        return 1

    failures: list[str] = []
    try:
        print(f"  {OK} API connected (server version {ib.client.serverVersion()})")

        accounts = ib.managedAccounts()
        if not accounts:
            print(f"  {FAIL} no managed accounts returned")
            failures.append("accounts")
        else:
            print(f"  {OK} accounts: {', '.join(accounts)}")
            for acct in accounts:
                is_paper = acct.startswith(PAPER_PREFIXES)
                kind = "PAPER" if is_paper else f"{RED}LIVE{RESET}"
                print(f"         {acct} -> {kind}")
                if not is_paper and expected == "paper":
                    print(f"  {WARN} connected to a LIVE account on a paper port")
                    failures.append("live account on paper port")

        summary = ib.accountSummary()
        wanted = {"NetLiquidation", "TotalCashValue", "BuyingPower", "AvailableFunds"}
        rows = [v for v in summary if v.tag in wanted]
        if rows:
            print(f"  {OK} account summary readable")
            for v in sorted(rows, key=lambda r: r.tag):
                print(f"{DIM}         {v.tag:<18} {v.value} {v.currency}{RESET}")
        else:
            print(f"  {WARN} account summary empty (normal immediately after login)")

        positions = ib.positions()
        print(f"  {OK} positions readable ({len(positions)} open)")
        for p in positions:
            print(f"{DIM}         {p.contract.symbol:<8} {p.position} @ {p.avgCost}{RESET}")

    finally:
        ib.disconnect()

    print("=" * 70)
    if failures:
        print(f"RESULT: FAIL ({', '.join(failures)})")
        return 1
    print(f"RESULT: PASS -- IBKR {expected} account reachable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
