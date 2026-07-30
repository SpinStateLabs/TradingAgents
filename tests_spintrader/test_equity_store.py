"""Equity persistence + the store-driven dashboard. No network, no real DB.

Three things are pinned here:

* :meth:`Store.write_equity_point` / :meth:`Store.read_equity_curve` round-trip
  through the *real* store code -- against a fake cursor that mimics the table's
  upsert-on-(mode,run_id,ts) semantics -- so the SQL, the Decimal/UTC handling and
  the idempotency are exercised, not a reimplementation of them;
* the decision loop writes one equity point per fast tick, best-effort, so a
  store without an equity writer (or a failing one) never stops trading; and
* :func:`build_dashboard_from_store` reconstructs the portfolio from the
  accumulated curve, the forecasts from the last decisions, and leaves the panels
  the store cannot supply honestly empty.
"""

from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from spintrader.agents.panel import PanelVerdict
from spintrader.backtest.runner import backtest_instrument, costs_for
from spintrader.core.config import Aggression, LiveGate, Settings
from spintrader.core.types import (
    Action, AssetClass, Bar, Decision, Position, Side, TradingMode, to_decimal,
)
from spintrader.dashboard.live import build_dashboard_from_store, forecasts_from_decisions
from spintrader.data.store import EquityRow, Store, _positions_payload
from spintrader.loop.decision_loop import DecisionLoop
from spintrader.portfolio.ledger import Ledger, EquitySnapshot
from spintrader.risk.engine import Mandate, RiskEngine, TradeIntent
from spintrader.venues.paper import PaperVenue

D = Decimal
UTC = timezone.utc
T0 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------
# A fake cursor that understands exactly the equity/decision statements
# --------------------------------------------------------------------------

class _FakeCursor:
    """Mimics the table semantics the real SQL relies on, nothing more."""

    def __init__(self, equity: dict, decisions: dict) -> None:
        self._equity = equity          # (mode, run_id, ts) -> insert params
        self._decisions = decisions    # decision_id -> insert params
        self._rows: list = []

    def execute(self, sql: str, params=None) -> None:
        p = list(params or [])
        s = " ".join(sql.split())

        if "INSERT INTO equity_curve" in s:
            self._equity[(p[1], p[2], p[0])] = p        # upsert on (mode,run_id,ts)
            self._rows = []
        elif "FROM equity_curve" in s:
            mode, run_id, i = p[0], p[1], 2
            start = end = limit = None
            if "ts >= %s" in s:
                start = p[i]; i += 1
            if "ts <= %s" in s:
                end = p[i]; i += 1
            if "LIMIT %s" in s:
                limit = p[i]; i += 1
            rows = [r for r in self._equity.values() if r[1] == mode and r[2] == run_id]
            if start is not None:
                rows = [r for r in rows if r[0] >= start]
            if end is not None:
                rows = [r for r in rows if r[0] <= end]
            rows.sort(key=lambda r: r[0], reverse=True)     # ORDER BY ts DESC
            if limit is not None:
                rows = rows[:limit]
            self._rows = [
                (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], json.loads(r[9]))
                for r in rows
            ]
        elif "INSERT INTO decisions" in s:
            self._decisions.setdefault(p[0], p)             # ON CONFLICT DO NOTHING
            self._rows = []
        elif "FROM decisions" in s:
            mode, i = p[0], 1
            inst = start = end = limit = None
            if "instrument_key = %s" in s:
                inst = p[i]; i += 1
            if "ts >= %s" in s:
                start = p[i]; i += 1
            if "ts <= %s" in s:
                end = p[i]; i += 1
            if "LIMIT %s" in s:
                limit = p[i]; i += 1
            rows = [r for r in self._decisions.values() if r[11] == mode]
            if inst is not None:
                rows = [r for r in rows if r[1] == inst]
            if start is not None:
                rows = [r for r in rows if r[2] >= start]
            if end is not None:
                rows = [r for r in rows if r[2] <= end]
            rows.sort(key=lambda r: r[2], reverse=True)
            if limit is not None:
                rows = rows[:limit]
            self._rows = [
                (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
                 json.loads(r[9]), json.loads(r[10]), r[11])
                for r in rows
            ]
        else:
            self._rows = []

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeStore(Store):
    """The real Store, but every cursor is the in-memory fake above."""

    def __init__(self) -> None:                             # noqa: D107 - no DB
        self._equity: dict = {}
        self._decisions: dict = {}

    @contextmanager
    def cursor(self):
        yield _FakeCursor(self._equity, self._decisions)


def snapshot(**over) -> EquitySnapshot:
    kw = dict(
        ts=T0, base_currency="USD", cash=D("4000"), positions_value=D("6000"),
        equity=D("10000"), realized_pnl=D("0"), unrealized_pnl=D("0"),
        fees_paid=D("0"), gross_exposure=D("0.6"), complete=True,
    )
    kw.update(over)
    return EquitySnapshot(**kw)


def a_position(last=D("120")) -> Position:
    pos = Position(instrument_key="paper:BTC-USD", qty=D("0.5"), avg_cost=D("100"))
    pos.last_price = last
    return pos


# --------------------------------------------------------------------------
# write_equity_point / read_equity_curve
# --------------------------------------------------------------------------

class EquityStoreTests(unittest.TestCase):
    def test_round_trip_is_chronological_and_decimal(self):
        store = FakeStore()
        store.write_equity_point(snapshot(ts=T0 + timedelta(days=1), equity=D("10100")),
                                 "paper", "paper")
        store.write_equity_point(snapshot(ts=T0, equity=D("10000")), "paper", "paper")

        rows = store.read_equity_curve("paper", "paper")
        self.assertEqual([r.equity for r in rows], [D("10000"), D("10100")])  # sorted asc
        self.assertIsInstance(rows[0], EquityRow)
        self.assertIsInstance(rows[0].equity, Decimal)
        self.assertEqual(rows[0].ts.tzinfo, UTC)

    def test_positions_round_trip_as_json(self):
        store = FakeStore()
        store.write_equity_point(snapshot(), "paper", "paper",
                                 {"paper:BTC-USD": a_position(), "paper:FLAT":
                                  Position(instrument_key="paper:FLAT", qty=D("0"))})
        row = store.read_equity_curve("paper", "paper")[0]
        self.assertIn("paper:BTC-USD", row.positions)
        self.assertNotIn("paper:FLAT", row.positions)          # flat dropped
        self.assertEqual(row.positions["paper:BTC-USD"]["qty"], "0.5")
        self.assertEqual(row.positions["paper:BTC-USD"]["last_price"], "120")

    def test_upsert_is_idempotent_on_the_key(self):
        store = FakeStore()
        store.write_equity_point(snapshot(equity=D("10000")), "paper", "paper")
        store.write_equity_point(snapshot(equity=D("12345")), "paper", "paper")  # same ts
        rows = store.read_equity_curve("paper", "paper")
        self.assertEqual(len(rows), 1)                         # one row, not two
        self.assertEqual(rows[0].equity, D("12345"))           # latest wins

    def test_run_id_and_mode_separate_curves(self):
        store = FakeStore()
        store.write_equity_point(snapshot(equity=D("100")), "paper", "runA")
        store.write_equity_point(snapshot(equity=D("200")), "paper", "runB")
        store.write_equity_point(snapshot(equity=D("300")), "live", "runA")
        self.assertEqual([r.equity for r in store.read_equity_curve("paper", "runA")], [D("100")])
        self.assertEqual([r.equity for r in store.read_equity_curve("paper", "runB")], [D("200")])
        self.assertEqual([r.equity for r in store.read_equity_curve("live", "runA")], [D("300")])

    def test_time_window_filters(self):
        store = FakeStore()
        for i in range(5):
            store.write_equity_point(
                snapshot(ts=T0 + timedelta(days=i), equity=D("10000") + D(i)),
                "paper", "paper",
            )
        rows = store.read_equity_curve(
            "paper", "paper", start=T0 + timedelta(days=1), end=T0 + timedelta(days=3),
        )
        self.assertEqual([r.ts for r in rows],
                         [T0 + timedelta(days=i) for i in (1, 2, 3)])

    def test_positions_payload_helper(self):
        payload = _positions_payload({
            "paper:BTC-USD": a_position(),
            "paper:FLAT": Position(instrument_key="paper:FLAT", qty=D("0")),
        })
        self.assertEqual(set(payload), {"paper:BTC-USD"})
        self.assertEqual(payload["paper:BTC-USD"],
                         {"qty": "0.5", "avg_cost": "100", "last_price": "120"})


# --------------------------------------------------------------------------
# The loop persists an equity point each tick, best-effort
# --------------------------------------------------------------------------

class _OneShotBuy:
    name = "oneshot"
    warmup_bars = 2

    def __init__(self):
        self._fired = False

    def on_bar(self, cursor, instrument, mandate):
        if self._fired:
            return []
        self._fired = True
        return [TradeIntent(
            instrument=instrument, side=Side.BUY, edge=D("0.05"),
            confidence=D("0.9"), volatility=D("0.20"),
            quote=cursor.quote(D("3")), strategy="oneshot",
        )]

    def fit(self, bars):
        pass


def _bars(n=30, key="paper:BTC-USD"):
    out = []
    for i in range(n):
        c = 100 + 0.05 * i + (0.03 if i % 2 else -0.03)
        out.append(Bar(
            instrument_key=key, ts=T0 + timedelta(minutes=i + 1), interval="1m",
            open=D(str(c - 0.02)), high=D(str(c + 0.2)), low=D(str(c - 0.2)),
            close=D(str(c)), volume=D("1"),
        ))
    return out


class _RecordingStore:
    """Serves bars, records decisions and equity points -- like the loop's store."""

    def __init__(self, bars, key):
        self.bars_by_key = {key: list(bars)}
        self.decisions = []
        self.equity_points = []

    def read_bars(self, key, interval, start=None, end=None, limit=None):
        b = self.bars_by_key.get(key, [])
        return b[-limit:] if limit else list(b)

    def write_decision(self, decision, mode):
        self.decisions.append((decision, mode))

    def write_equity_point(self, snapshot, mode, run_id="live", positions=None):
        self.equity_points.append(
            SimpleNamespace(snapshot=snapshot, mode=mode, run_id=run_id,
                            positions=dict(positions or {})))


class _BarelyAStore:
    """Has read_bars but NO write_equity_point -- the best-effort escape hatch."""

    def __init__(self, bars, key):
        self.bars_by_key = {key: list(bars)}

    def read_bars(self, key, interval, start=None, end=None, limit=None):
        b = self.bars_by_key.get(key, [])
        return b[-limit:] if limit else list(b)


def _make_loop(store, run_id="paper"):
    inst = backtest_instrument("BTC-USD", AssetClass.CRYPTO)
    settings = Settings(
        mode=TradingMode.PAPER, aggression=Aggression.MODERATE, base_currency="USD",
        live=LiveGate(enabled=False), enforce_cash_account=True,
    )
    costs = costs_for(AssetClass.CRYPTO)
    holder = {}
    venue = PaperVenue(
        quote_source=lambda i: holder["loop"].market_quote(i),
        starting_cash=to_decimal("10000"), currency="USD", settings=settings,
        slippage=costs.slippage, settlement_days=1,
    )
    venue.register(inst)
    venue.connect()
    ledger = Ledger(base_currency="USD", mode=TradingMode.PAPER, settlement_days=1,
                    opening_cash={"USD": to_decimal("10000")})
    from spintrader.agents.personas.spec import Horizon
    loop = DecisionLoop(
        settings=settings, store=store, venue=venue, ledger=ledger,
        risk=RiskEngine(settings=settings), strategies={inst.key: _OneShotBuy()},
        mandate_service=None, instruments=[inst], interval="1m",
        spread_bps=costs.spread_bps, horizon=Horizon.INTRADAY, run_id=run_id,
    )
    holder["loop"] = loop
    return loop, inst, ledger


def _mandate(key):
    now = datetime.now(UTC)
    return Mandate(issued_at=now, expires_at=now + timedelta(hours=1),
                   permitted=frozenset({key}))


class LoopPersistenceTests(unittest.TestCase):
    def test_each_tick_writes_one_equity_point(self):
        store = _RecordingStore(_bars(), "paper:BTC-USD")
        loop, inst, ledger = _make_loop(store, run_id="paper")

        loop.fast_tick(_mandate(inst.key), now=T0)
        loop.fast_tick(_mandate(inst.key), now=T0 + timedelta(minutes=1))

        self.assertEqual(len(store.equity_points), 2)          # one per tick
        for point in store.equity_points:
            self.assertEqual(point.mode, "paper")
            self.assertEqual(point.run_id, "paper")
            self.assertGreater(point.snapshot.equity, D("0"))
        # After the buy, the persisted point carries the open position.
        self.assertIn(inst.key, store.equity_points[-1].positions)

    def test_missing_equity_writer_does_not_stop_the_tick(self):
        store = _BarelyAStore(_bars(), "paper:BTC-USD")
        loop, inst, ledger = _make_loop(store, run_id="paper")
        # No write_equity_point on the store: the tick must still trade.
        results = loop.fast_tick(_mandate(inst.key), now=T0)
        self.assertTrue(any(r.submitted for r in results))
        self.assertGreater(ledger.position(inst.key).qty, D("0"))


# --------------------------------------------------------------------------
# The store-driven dashboard
# --------------------------------------------------------------------------

class StoreDashboardTests(unittest.TestCase):
    def _seed(self):
        store = FakeStore()
        store.write_equity_point(snapshot(ts=T0, equity=D("10000"), cash=D("4000")),
                                 "paper", "paper", {"paper:BTC-USD": a_position()})
        store.write_equity_point(
            snapshot(ts=T0 + timedelta(days=1), equity=D("10200"), cash=D("4000")),
            "paper", "paper", {"paper:BTC-USD": a_position()},
        )
        # Two strategies; 'trend' has a newer HOLD after an older BUY.
        store.write_decision(Decision(
            instrument_key="paper:BTC-USD", action=Action.BUY, confidence=D("0.7"),
            ts=T0, contributions={"strategy": "trend", "edge": "0.02", "side": "buy"}),
            "paper")
        store.write_decision(Decision(
            instrument_key="paper:BTC-USD", action=Action.HOLD, confidence=D("0.3"),
            ts=T0 + timedelta(minutes=1),
            contributions={"strategy": "trend", "edge": "0.01", "side": "buy"}),
            "paper")
        store.write_decision(Decision(
            instrument_key="paper:BTC-USD", action=Action.BUY, confidence=D("0.6"),
            ts=T0, contributions={"strategy": "reversion", "edge": "0.03", "side": "buy"}),
            "paper")
        return store

    def test_portfolio_comes_from_the_curve(self):
        store = self._seed()
        model = build_dashboard_from_store(
            store, mode="paper", run_id="paper", symbol="BTC-USD", objective="crypto",
        )
        self.assertEqual(model.portfolio.equity, D("10200"))       # latest mark
        self.assertEqual(len(model.portfolio.equity_curve), 2)
        self.assertEqual(model.portfolio.starting_equity, D("10000"))
        self.assertEqual(model.portfolio.total_return, D("10200") / D("10000") - D("1"))
        self.assertEqual(len(model.portfolio.positions), 1)
        self.assertEqual(model.portfolio.positions[0].market_value, D("60"))  # 0.5*120

    def test_forecasts_are_the_last_decision_per_strategy(self):
        store = self._seed()
        model = build_dashboard_from_store(
            store, mode="paper", run_id="paper", symbol="BTC-USD", objective="crypto",
        )
        by_name = {f.name: f for f in model.forecasts.forecasts}
        self.assertEqual(set(by_name), {"trend", "reversion"})
        self.assertEqual(by_name["trend"].direction, "flat")       # newest was HOLD
        self.assertFalse(by_name["trend"].active)
        self.assertEqual(by_name["reversion"].direction, "long")   # BUY
        self.assertEqual(by_name["reversion"].edge, D("0.03"))

    def test_panels_the_store_cannot_supply_are_empty(self):
        store = self._seed()
        model = build_dashboard_from_store(
            store, mode="paper", run_id="paper", symbol="BTC-USD", objective="crypto",
        )
        self.assertEqual(model.leaderboard.evaluated, 0)           # no research memory
        self.assertEqual(model.experts.participating, 0)           # no deliberation

    def test_deliberation_populates_the_expert_panel(self):
        store = self._seed()
        verdict = PanelVerdict(action=Action.BUY, confidence=D("0.5"),
                               net_direction=D("0.3"), participating=2)
        delib = SimpleNamespace(verdicts={"paper:BTC-USD": verdict})
        model = build_dashboard_from_store(
            store, mode="paper", run_id="paper", symbol="BTC-USD", objective="crypto",
            instrument_key="paper:BTC-USD", deliberation=delib,
        )
        self.assertEqual(model.experts.action, "buy")
        self.assertEqual(model.experts.participating, 2)

    def test_empty_store_yields_a_zeroed_but_valid_model(self):
        model = build_dashboard_from_store(
            FakeStore(), mode="paper", run_id="paper", symbol="BTC-USD",
            objective="crypto",
        )
        self.assertEqual(model.portfolio.equity, D("0"))
        self.assertEqual(len(model.portfolio.equity_curve), 0)
        self.assertEqual(len(model.forecasts.forecasts), 0)

    def test_forecasts_from_decisions_helper_dedups_by_strategy(self):
        rows = [
            SimpleNamespace(action="hold", confidence=D("0.3"), rationale="newer",
                            contributions={"strategy": "trend", "edge": "0.01"}),
            SimpleNamespace(action="buy", confidence=D("0.7"), rationale="older",
                            contributions={"strategy": "trend", "edge": "0.02"}),
        ]
        out = forecasts_from_decisions(rows)
        self.assertEqual(len(out), 1)                              # deduped
        self.assertEqual(out[0].note, "newer")                    # first (newest) kept


if __name__ == "__main__":
    unittest.main()
