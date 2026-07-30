"""Build the dashboard from the *real* book in the store, not a demo run.

The demo in :mod:`spintrader.dashboard.__main__` fabricates an equity curve and a
persona vote from a bundled CSV. This module does the opposite: it reads what the
running loop actually persisted -- the accumulated equity curve, the last
decisions, and (when the caller hands it over) the latest deliberation -- and
turns that into the same :class:`~spintrader.dashboard.model.DashboardModel` the
renderer already knows how to draw.

The split from the demo is deliberate. The three panels that have a durable home
in the store are read straight from it:

* **Portfolio** ← ``equity_curve``. The curve *is* the persisted history, so the
  dashboard shows growth that accumulates across restarts rather than a curve
  invented afresh each render. The latest row reconstructs the current book.
* **Forecasts** ← ``decisions``. Each recent decision carries the strategy that
  made it, its edge and its side, so the most recent decision per strategy is
  exactly that strategy's latest live read.
* **Leaderboard** ← the research memory (itself persisted in ``research_cache``),
  loaded best-effort so a store without it simply yields an empty board.

The **deliberation** is not persisted anywhere -- it lives in the running loop --
so it is passed in when available (``loop.deliberation``) and the expert panel is
empty without it, which is honest rather than fabricated. Every monetary field is
a :class:`~decimal.Decimal` and every timestamp is aware UTC, as everywhere else.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Sequence

from spintrader.core.types import Position, to_decimal, utcnow
from spintrader.dashboard.model import (
    DashboardModel, EquityPoint, ExpertPanel, StrategyForecast, StrategyLeaderboard,
    build_expert_panel, build_forecast_panel, build_leaderboard,
    build_portfolio_panel,
)
from spintrader.portfolio.ledger import EquitySnapshot

if TYPE_CHECKING:  # pragma: no cover - hints only
    from spintrader.data.store import DecisionRow, EquityRow

ZERO = Decimal("0")


# --------------------------------------------------------------------------
# Reconstructing the book from a persisted row
# --------------------------------------------------------------------------

def _snapshot_from_row(row: "EquityRow", base_currency: str) -> EquitySnapshot:
    """Rebuild an :class:`EquitySnapshot` from one persisted equity mark.

    ``positions_value`` is not stored separately -- it is ``equity - cash`` by
    construction -- and the row is only ever written from a complete valuation,
    so ``complete`` is True with no unpriced or unconvertible names.
    """
    return EquitySnapshot(
        ts=row.ts,
        base_currency=base_currency,
        cash=row.cash,
        positions_value=row.equity - row.cash,
        equity=row.equity,
        realized_pnl=row.realized_pnl,
        unrealized_pnl=row.unrealized_pnl,
        fees_paid=row.fees_paid,
        gross_exposure=row.gross_exposure,
        complete=True,
    )


def _positions_from_row(row: "EquityRow") -> dict[str, Position]:
    """Rebuild the open-position map from a row's stored JSONB.

    Enough to revalue the book on the dashboard -- qty, basis and last mark --
    which is exactly what :func:`build_portfolio_panel` reads off each position.
    """
    out: dict[str, Position] = {}
    for key, payload in (row.positions or {}).items():
        pos = Position(
            instrument_key=key,
            qty=to_decimal(payload.get("qty", 0)),
            avg_cost=to_decimal(payload.get("avg_cost", 0)),
        )
        last = payload.get("last_price")
        if last is not None:
            pos.last_price = to_decimal(last)
        out[key] = pos
    return out


def _empty_snapshot(base_currency: str, ts: datetime) -> EquitySnapshot:
    """The book before the loop has marked anything: zeroed but well-formed."""
    return EquitySnapshot(
        ts=ts, base_currency=base_currency, cash=ZERO, positions_value=ZERO,
        equity=ZERO, realized_pnl=ZERO, unrealized_pnl=ZERO, fees_paid=ZERO,
        gross_exposure=ZERO, complete=True,
    )


# --------------------------------------------------------------------------
# Forecasts from the last decisions
# --------------------------------------------------------------------------

def forecasts_from_decisions(
    decisions: Sequence["DecisionRow"],
) -> list[StrategyForecast]:
    """The latest decision per strategy, read as that strategy's current stance.

    ``decisions`` is newest-first (as the store returns it), so the first row seen
    for a strategy is its most recent one. A BUY is a live long; a SELL or CLOSE
    has stepped aside (flat); anything else is flat. The edge and confidence come
    straight from the decision's own attribution, so the forecast card reflects
    what the loop actually acted on rather than a re-derived guess.
    """
    out: list[StrategyForecast] = []
    seen: set[str] = set()
    for dec in decisions:
        contributions = dec.contributions or {}
        name = str(contributions.get("strategy") or dec.action or "strategy")
        if name in seen:
            continue
        seen.add(name)

        edge = to_decimal(contributions.get("edge", 0) or 0)
        active = dec.action == "buy"
        direction = "long" if active else "flat"
        out.append(StrategyForecast(
            name=name,
            family=str(contributions.get("family", "live")),
            edge=edge,
            confidence=dec.confidence,
            direction=direction,
            strength=float(edge),
            note=(dec.rationale or "").strip()[:160],
            active=active,
        ))
    return out


# --------------------------------------------------------------------------
# Empty panels (for parts the store cannot supply)
# --------------------------------------------------------------------------

def _empty_leaderboard(objective: str) -> StrategyLeaderboard:
    return StrategyLeaderboard(
        objective=objective, rows=(), evaluated=0, promoted=0,
        champion_key=None, champion_sharpe=None, rejection_profile={},
    )


def _empty_experts() -> ExpertPanel:
    return ExpertPanel(
        action="hold", confidence=ZERO, net_direction=ZERO, dispersion=ZERO,
        lens_diversity=0, participating=0, escalate=False,
    )


# --------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------

def build_dashboard_from_store(
    store: Any,
    *,
    mode: str,
    symbol: str,
    objective: str,
    run_id: str = "live",
    instrument_key: str | None = None,
    deliberation: Any | None = None,
    hedge_reading: Any | None = None,
    memory: Any | None = None,
    base_currency: str = "USD",
    decisions_limit: int = 50,
    title: str = "Portfolio & Forecast Dashboard",
    generated_at: datetime | None = None,
) -> DashboardModel:
    """Assemble a :class:`DashboardModel` from what the store has persisted.

    ``mode``/``run_id`` select which curve to read (a paper run and a live run are
    distinct curves under one book). ``deliberation`` is the loop's latest, passed
    in because it has no store home; ``memory`` overrides the store-backed
    research memory used for the leaderboard. Everything the store cannot supply
    degrades to an empty panel rather than a fabricated one.
    """
    now = generated_at or utcnow()

    # -- portfolio, from the accumulated equity curve --------------------
    rows = list(store.read_equity_curve(mode, run_id))
    if rows:
        latest = rows[-1]
        snapshot = _snapshot_from_row(latest, base_currency)
        positions = _positions_from_row(latest)
        curve = [EquityPoint(ts=r.ts, equity=r.equity) for r in rows]
        starting = rows[0].equity
    else:
        snapshot = _empty_snapshot(base_currency, now)
        positions = {}
        curve = []
        starting = None

    portfolio = build_portfolio_panel(
        snapshot, positions, curve, starting_equity=starting,
    )

    # -- forecasts, from the last decisions ------------------------------
    decisions = store.read_decisions(
        mode, instrument_key=instrument_key, limit=decisions_limit,
    )
    forecasts = build_forecast_panel(
        forecasts_from_decisions(decisions), symbol, snapshot.ts,
    )

    # -- leaderboard, from the (store-backed) research memory ------------
    if memory is None:
        try:
            from spintrader.research.memory import ResearchMemory
            memory = ResearchMemory(store=store)
            memory.load(objective)
        except Exception:                               # noqa: BLE001 - degrade
            memory = None
    leaderboard = (
        build_leaderboard(memory, objective) if memory is not None
        else _empty_leaderboard(objective)
    )

    # -- experts, from the latest deliberation (if the caller has one) ---
    verdict = None
    if deliberation is not None:
        verdicts = getattr(deliberation, "verdicts", {}) or {}
        verdict = verdicts.get(instrument_key) if instrument_key else None
        if verdict is None and verdicts:
            verdict = next(iter(verdicts.values()))
    experts = (
        build_expert_panel(verdict, hedge_reading) if verdict is not None
        else _empty_experts()
    )

    return DashboardModel(
        generated_at=now,
        title=title,
        symbol=symbol,
        objective=objective,
        portfolio=portfolio,
        forecasts=forecasts,
        leaderboard=leaderboard,
        experts=experts,
    )


__all__ = ["build_dashboard_from_store", "forecasts_from_decisions"]
