"""Verify the GB10 configuration in .env: database, Ollama, and data paths.

    PYTHONPATH=. .venv/bin/python ops/check_gb10_env.py

Checks every GB10-facing setting resolves and actually works, so a
misconfiguration is caught here rather than partway through a trading cycle.
Prints no secrets.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

from spintrader.core.config import (
    SecretFileError, Settings, load_env_file,
)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
OK, FAIL, WARN = f"{GREEN}PASS{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}"


def check_tcp(host: str, port: int, timeout: float = 5.0) -> str | None:
    """Return None on success, or an error string."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return None
    except OSError as exc:
        return str(exc)


def main() -> int:
    loaded = load_env_file(".env")
    print("=" * 70)
    print("GB10 environment check")
    print("=" * 70)
    print(f"{DIM}  loaded {loaded} vars from .env{RESET}")

    failures: list[str] = []
    warnings: list[str] = []

    # --- settings resolve -------------------------------------------------
    try:
        settings = Settings.from_env()
    except SecretFileError as exc:
        print(f"  {FAIL} database password: {exc}")
        print(f"{DIM}         On the GB10 this file is created by ops/provision.sh.\n"
              f"         From the Windows box, set SPINTRADER_DB_PASSWORD inline instead\n"
              f"         and leave SPINTRADER_DB_PASSWORD_FILE blank.{RESET}")
        return 1
    except ValueError as exc:
        print(f"  {FAIL} settings did not resolve: {exc}")
        return 1

    print(f"  {OK} settings resolved")
    for key, value in settings.describe().items():
        print(f"{DIM}         {key:<22} {value}{RESET}")
    print()

    # --- database ---------------------------------------------------------
    storage = settings.storage
    if not storage.password:
        print(f"  {FAIL} database password is empty")
        print(f"{DIM}         Set SPINTRADER_DB_PASSWORD_FILE (on the GB10) or "
              f"SPINTRADER_DB_PASSWORD (elsewhere).{RESET}")
        failures.append("db password")
    else:
        print(f"  {OK} database password loaded ({len(storage.password)} chars, not shown)")

    err = check_tcp(storage.host, storage.port)
    if err:
        print(f"  {FAIL} cannot reach {storage.host}:{storage.port} -- {err}")
        print(f"{DIM}         Is the container up?  docker ps | grep spintrader-db{RESET}")
        failures.append("db reachable")
    else:
        print(f"  {OK} {storage.host}:{storage.port} reachable")

        # Only attempt a real connection if the port answered.
        try:
            import psycopg
        except ImportError:
            print(f"  {WARN} psycopg not installed; skipping the live query")
            warnings.append("psycopg not installed (uv pip install 'psycopg[binary,pool]')")
        else:
            try:
                with psycopg.connect(storage.dsn, connect_timeout=10) as conn:
                    row = conn.execute("SELECT version()").fetchone()
                    print(f"  {OK} authenticated: {row[0].split(' on ')[0]}")
                    ts = conn.execute(
                        "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                    ).fetchone()
                    if ts:
                        print(f"  {OK} timescaledb {ts[0]} available")
                    else:
                        print(f"  {WARN} timescaledb extension not enabled in this database")
                        warnings.append("timescaledb extension missing")
            except Exception as exc:                    # noqa: BLE001 - reported
                print(f"  {FAIL} database connection failed: {type(exc).__name__}: {exc}")
                failures.append("db connect")

    # --- ollama -----------------------------------------------------------
    from spintrader.llm.router import LLMRouter          # imported late: needs requests

    health = LLMRouter(config=settings.llm).health_check()
    if health["ok"]:
        print(f"  {OK} ollama at {health['base_url']}")
        print(f"{DIM}         quick {health['quick_model']}{RESET}")
        print(f"{DIM}         deep  {health['deep_model']}{RESET}")
    else:
        print(f"  {FAIL} ollama: {health['error']}")
        if "models" in health:
            print(f"{DIM}         pulled: {health['models']}{RESET}")
        failures.append("ollama")

    # --- data root --------------------------------------------------------
    root = Path(storage.data_root)
    if root.is_dir():
        print(f"  {OK} data root {root}")
    else:
        # Only meaningful on the GB10; the Windows box has no such path.
        print(f"  {WARN} data root {root} does not exist here")
        warnings.append(f"data root missing (expected when not on the GB10)")

    # --- summary ----------------------------------------------------------
    print("-" * 70)
    for w in warnings:
        print(f"  {WARN} {w}")
    print("=" * 70)
    if failures:
        print(f"RESULT: FAIL ({', '.join(failures)})")
        return 1
    print("RESULT: PASS -- GB10 is configured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
