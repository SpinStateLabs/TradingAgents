"""Pure aggregation for the Portfolio & Forecast dashboard.

This module turns the system's *live* objects -- a research memory, a ledger, a
persona panel verdict, a handful of strategy readings -- into a single flat,
immutable :class:`DashboardModel` that a renderer can walk without ever touching
the trading stack again.

Why a separate model at all, rather than rendering straight from the live
objects?

* **The renderer must not compute.** A template that reaches into a ``Ledger`` to
  re-value the book, or into a ``ResearchMemory`` to re-rank candidates, has
  quietly become a second, unreviewed copy of that logic -- and the two copies
  drift. Every number the dashboard shows is decided *here*, once, where it can
  be unit-tested against fake inputs with no HTML in sight.
* **The model is honest about money.** Every monetary field stays a
  :class:`~decimal.Decimal` exactly as the ledger produced it; rounding is a
  presentation choice and belongs to the renderer, not to the aggregation. A
  float creeping in here would be the same silent ledger corruption the core
  types go to such lengths to prevent.
* **The inputs are duck-typed on purpose.** The builders read attributes
  (``.sharpe``, ``.confidence``, ``.edge``) rather than importing the concrete
  classes, so a test can feed a trivial stand-in and so the dashboard never
  becomes a reason the core types cannot change shape.

Timestamps are timezone-aware UTC throughout, per the project-wide convention;
:func:`build_dashboard_model` stamps ``generated_at`` with :func:`utcnow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from spintrader.core.types import to_decimal, utcnow

if TYPE_CHECKING:  # pragma: no cover - hints only, never imported at runtime
    from spintrader.agents.panel import PanelVerdict
    from spintrader.agents.personas.hedge import HedgeReading
    from spintrader.core.types import Position
    from spintrader.portfolio.ledger import EquitySnapshot
    from spintrader.research.memory import ResearchMemory

ZERO = Decimal("0")

# Reading fields that carry a strategy's signed directional conviction, in the
# order we prefer them. Different families expose different names for "how
# bullish am I": the trend follower reports ``trend_strength``, the Hedge
# ensemble a ``blended`` vote, the Markov chain an ``expected_return``, and so
# on. The first present wins, so one generic function can read them all.
_STRENGTH_FIELDS: tuple[str, ...] = (
    "trend_strength", "blended", "expected_return", "gap", "drift", "zscore",
)


# --------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One sample of the equity curve: a UTC instant and the book's value then."""
    ts: datetime
    equity: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioPosition:
    """A single open position, already valued at the snapshot's marks."""
    instrument_key: str
    qty: Decimal
    avg_cost: Decimal
    last_price: Decimal | None
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioPanel:
    """The book at a point in time, plus the curve that led there.

    Mirrors :class:`~spintrader.portfolio.ledger.EquitySnapshot` field-for-field
    so the dashboard reports exactly what the ledger valued -- including
    ``complete``, because a partial valuation shown as though it were total is
    the very failure the snapshot flag exists to surface.
    """
    as_of: datetime
    base_currency: str
    equity: Decimal
    cash: Decimal
    positions_value: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal
    net_pnl: Decimal
    gross_exposure: Decimal
    complete: bool
    positions: tuple[PortfolioPosition, ...] = ()
    equity_curve: tuple[EquityPoint, ...] = ()
    starting_equity: Decimal | None = None

    @property
    def total_return(self) -> Decimal | None:
        """Return over the curve, if a starting point is known."""
        if self.starting_equity is None or self.starting_equity == ZERO:
            return None
        return self.equity / self.starting_equity - Decimal("1")


# --------------------------------------------------------------------------
# Strategy leaderboard
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LeaderboardRow:
    """One evaluated candidate, with its verdict and the reason for it."""
    family: str
    config_key: str
    sharpe: float | None
    deflated_sharpe: float
    total_return: float | None
    max_drawdown: float | None
    n_trials: int
    promoted: bool
    reason: str
    is_champion: bool = False


@dataclass(frozen=True, slots=True)
class StrategyLeaderboard:
    """Every candidate the improvement loop evaluated for one objective.

    Ordered best-Sharpe-first, which is the order a reader scans for the
    recommended strategy. The rejection profile travels alongside because *why*
    the loop is saying no is the diagnostic on the loop itself -- all
    ``NOT_SIGNIFICANT`` means the generator is producing noise; a wall of
    ``COST_EXCEEDS_EDGE`` means it is finding real patterns too small to trade.
    """
    objective: str
    rows: tuple[LeaderboardRow, ...]
    evaluated: int
    promoted: int
    champion_key: str | None
    champion_sharpe: float | None
    rejection_profile: Mapping[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Expert panel (mixture-of-experts / chain-of-thought)
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExpertVote:
    """One persona-expert's contribution to the panel verdict.

    ``weight`` is the reliability x horizon-fit the panel actually applied, and
    ``effective`` is the signed, weighted push it added to the net direction --
    the two numbers that make later attribution possible. ``rationale`` is the
    expert's chain of thought; ``changed_by`` is the single piece of evidence it
    says would flip its view, which is what turns a vote into a falsifiable one.
    """
    persona_key: str
    action: str
    confidence: Decimal
    weight: Decimal
    effective: Decimal
    lenses: tuple[str, ...]
    rationale: str
    changed_by: str
    abstained: bool


@dataclass(frozen=True, slots=True)
class HedgeExpert:
    """One sub-expert of the no-regret Hedge ensemble (momentum/reversion/breakout).

    Distinct from a persona: these are the three deterministic signals *inside*
    the Hedge agent, shown with the multiplicative weight the meta-learner has
    assigned each and the vote it is currently casting, so the mixture itself is
    visible rather than just its blended output.
    """
    name: str
    weight: float
    vote: float


@dataclass(frozen=True, slots=True)
class ExpertPanel:
    """The aggregated panel verdict with the diagnostics needed to trust it.

    Everything a reader needs to judge *how much* to believe the headline
    action: the dispersion of opinion, how many distinct lenses actually agreed
    (five personas all reading momentum is one opinion, not five), whether the
    panel voted to escalate to the deep model, and who abstained and why.
    """
    action: str
    confidence: Decimal
    net_direction: Decimal
    dispersion: Decimal
    lens_diversity: int
    participating: int
    escalate: bool
    experts: tuple[ExpertVote, ...] = ()
    abstentions: tuple[tuple[str, str], ...] = ()
    hedge_experts: tuple[HedgeExpert, ...] = ()


# --------------------------------------------------------------------------
# Per-strategy forecasts
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StrategyForecast:
    """One strategy's latest read on an instrument: edge, confidence, direction.

    ``direction`` is a coarse label -- ``long`` / ``flat`` / ``short`` /
    ``aside`` -- while ``strength`` keeps the raw signed conviction the label was
    derived from, so a reader can see both the call and how close it was. A
    long-only persona that would not enter reports ``flat``; one whose model is
    unavailable (the regime family without ``hmmlearn``) reports ``aside``, which
    is honest rather than a fabricated neutral.
    """
    name: str
    family: str
    edge: Decimal
    confidence: Decimal
    direction: str
    strength: float
    note: str
    active: bool


@dataclass(frozen=True, slots=True)
class ForecastPanel:
    """The strategy forecasts, side by side, for one instrument."""
    symbol: str
    as_of: datetime
    forecasts: tuple[StrategyForecast, ...] = ()


# --------------------------------------------------------------------------
# The whole dashboard
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DashboardModel:
    """Everything the renderer needs, and nothing it would have to compute."""
    generated_at: datetime
    title: str
    symbol: str
    objective: str
    portfolio: PortfolioPanel
    forecasts: ForecastPanel
    leaderboard: StrategyLeaderboard
    experts: ExpertPanel


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------

def build_portfolio_panel(
    snapshot: "EquitySnapshot",
    positions: Mapping[str, "Position"] | None = None,
    equity_curve: Sequence[EquityPoint] | Sequence[tuple[datetime, Decimal]] = (),
    *,
    starting_equity: Decimal | None = None,
) -> PortfolioPanel:
    """Aggregate a ledger snapshot (plus optional detail) into a portfolio panel.

    ``positions`` is the ledger's live position map; only non-flat positions are
    surfaced, valued at the mark already stored on each. ``equity_curve`` accepts
    either :class:`EquityPoint`s or bare ``(ts, equity)`` pairs, so a caller can
    hand over whatever it recorded during a paper session without adapting it.
    """
    rows: list[PortfolioPosition] = []
    for key, pos in (positions or {}).items():
        if getattr(pos, "is_flat", pos.qty == ZERO):
            continue
        price = pos.last_price
        rows.append(PortfolioPosition(
            instrument_key=key,
            qty=to_decimal(pos.qty),
            avg_cost=to_decimal(pos.avg_cost),
            last_price=None if price is None else to_decimal(price),
            market_value=to_decimal(pos.market_value(price)),
            unrealized_pnl=to_decimal(pos.unrealized_pnl(price)),
        ))
    rows.sort(key=lambda r: abs(r.market_value), reverse=True)

    curve: list[EquityPoint] = []
    for point in equity_curve:
        if isinstance(point, EquityPoint):
            curve.append(point)
        else:
            ts, eq = point
            curve.append(EquityPoint(ts=ts, equity=to_decimal(eq)))

    start = starting_equity
    if start is None and curve:
        start = curve[0].equity

    return PortfolioPanel(
        as_of=snapshot.ts,
        base_currency=snapshot.base_currency,
        equity=snapshot.equity,
        cash=snapshot.cash,
        positions_value=snapshot.positions_value,
        realized_pnl=snapshot.realized_pnl,
        unrealized_pnl=snapshot.unrealized_pnl,
        fees_paid=snapshot.fees_paid,
        net_pnl=snapshot.net_pnl,
        gross_exposure=snapshot.gross_exposure,
        complete=snapshot.complete,
        positions=tuple(rows),
        equity_curve=tuple(curve),
        starting_equity=start,
    )


def build_leaderboard(memory: "ResearchMemory", objective: str) -> StrategyLeaderboard:
    """Rank every evaluated candidate for ``objective`` best-Sharpe-first.

    Reads only the research memory's public surface (``records``, ``best``,
    ``summary``), so the leaderboard is exactly the loop's own record of what it
    tried and why it refused most of it.
    """
    records = list(memory.records(objective))
    champion = memory.best(objective)
    champion_key = champion.config_key if champion else None

    rows: list[LeaderboardRow] = []
    for rec in records:
        rows.append(LeaderboardRow(
            family=rec.family,
            config_key=rec.config_key,
            sharpe=rec.sharpe,
            deflated_sharpe=rec.deflated_sharpe,
            total_return=rec.total_return,
            max_drawdown=rec.max_drawdown,
            n_trials=rec.n_trials,
            promoted=rec.promoted,
            reason=_verdict_reason(rec),
            is_champion=(rec.config_key == champion_key),
        ))
    # Highest Sharpe first; a missing Sharpe sorts to the bottom rather than
    # crashing the comparison.
    rows.sort(key=lambda r: (-1e18 if r.sharpe is None else r.sharpe), reverse=True)

    summary = memory.summary(objective)
    return StrategyLeaderboard(
        objective=objective,
        rows=tuple(rows),
        evaluated=int(summary.get("evaluated", len(records))),
        promoted=int(summary.get("promoted", sum(1 for r in records if r.promoted))),
        champion_key=champion_key,
        champion_sharpe=(champion.sharpe if champion else None),
        rejection_profile=dict(summary.get("rejections", {})),
    )


def _verdict_reason(rec: Any) -> str:
    """A one-line reason for a candidate's verdict, promoted or not."""
    if rec.promoted:
        note = (rec.note or "").strip()
        return note or "cleared all promotion gates"
    reasons = list(rec.rejections or ())
    if reasons:
        return ", ".join(r.replace("_", " ") for r in reasons)
    return (rec.note or "rejected").strip()


def build_expert_panel(
    verdict: "PanelVerdict",
    hedge_reading: "HedgeReading | None" = None,
) -> ExpertPanel:
    """Turn a persona-panel verdict into the mixture-of-experts view.

    Every vote is carried, abstentions included and flagged, because an
    abstention with a stated cause is information -- knowing *who could not
    speak* is part of trusting the ones who did. If a Hedge reading is supplied,
    its three internal experts are attached so the no-regret sub-ensemble is
    visible next to the persona ensemble.
    """
    experts: list[ExpertVote] = []
    for vote in verdict.votes:
        experts.append(ExpertVote(
            persona_key=vote.persona_key,
            action=_enum_value(vote.action),
            confidence=to_decimal(vote.confidence),
            weight=to_decimal(vote.weight),
            effective=to_decimal(vote.effective),
            lenses=tuple(_enum_value(l) for l in vote.lenses),
            rationale=vote.rationale,
            changed_by=vote.changed_by,
            abstained=bool(vote.abstained),
        ))
    # Heaviest, most influential experts first: that is the reading order for a
    # weight bar chart.
    experts.sort(key=lambda e: e.weight, reverse=True)

    hedge_experts: tuple[HedgeExpert, ...] = ()
    if hedge_reading is not None:
        from spintrader.agents.personas.hedge import EXPERTS
        hedge_experts = tuple(
            HedgeExpert(name=name, weight=float(w), vote=float(v))
            for name, w, v in zip(EXPERTS, hedge_reading.weights, hedge_reading.votes)
        )

    return ExpertPanel(
        action=_enum_value(verdict.action),
        confidence=to_decimal(verdict.confidence),
        net_direction=to_decimal(verdict.net_direction),
        dispersion=to_decimal(verdict.dispersion),
        lens_diversity=int(verdict.lens_diversity),
        participating=int(verdict.participating),
        escalate=bool(verdict.escalate),
        experts=tuple(experts),
        abstentions=tuple(sorted(verdict.abstentions.items())),
        hedge_experts=hedge_experts,
    )


def forecast_from_reading(
    name: str, family: str, reading: Any,
) -> StrategyForecast:
    """Read edge / confidence / direction off any strategy's latest reading.

    Deliberately attribute-driven rather than type-driven: the personas expose
    different field names for the same ideas, and reading them by name lets one
    function serve trend, reversion, Markov, regime and Hedge alike -- and lets a
    test pass a two-field stand-in. A ``fitted`` field that is False (the regime
    family with no ``hmmlearn``) yields ``aside``: the strategy stood down rather
    than voiced a neutral it does not hold.
    """
    edge = to_decimal(getattr(reading, "edge", 0) or 0)
    confidence = to_decimal(getattr(reading, "confidence", 0) or 0)

    fitted = getattr(reading, "fitted", True)
    would_enter = bool(
        getattr(reading, "bullish", False) or getattr(reading, "oversold", False)
    )

    strength = 0.0
    strength_label = ""
    for field_name in _STRENGTH_FIELDS:
        value = getattr(reading, field_name, None)
        if value is not None:
            strength = float(value)
            strength_label = field_name.replace("_", " ")
            break

    if fitted is False:
        direction, active = "aside", False
    elif would_enter:
        direction, active = "long", True
    else:
        direction, active = "flat", False

    note = _forecast_note(reading, strength_label, strength, fitted)
    return StrategyForecast(
        name=name, family=family, edge=edge, confidence=confidence,
        direction=direction, strength=strength, note=note, active=active,
    )


def _forecast_note(reading: Any, strength_label: str, strength: float,
                   fitted: Any) -> str:
    """A short human descriptor of the reading, for the forecast card."""
    if fitted is False:
        return "regime model unavailable (no hmmlearn); standing aside"
    bits: list[str] = []
    if strength_label:
        bits.append(f"{strength_label} {strength:+.3f}")
    vol = getattr(reading, "annual_vol", None)
    if vol is not None:
        bits.append(f"vol {float(vol):.0%}")
    support = getattr(reading, "support", None)
    if support is not None:
        bits.append(f"support {int(support)}")
    return ", ".join(bits)


def build_forecast_panel(
    forecasts: Sequence[StrategyForecast],
    symbol: str,
    as_of: datetime | None = None,
) -> ForecastPanel:
    """Wrap already-built forecasts into a panel (active strategies first)."""
    ordered = sorted(forecasts, key=lambda f: (not f.active, f.name))
    return ForecastPanel(
        symbol=symbol, as_of=as_of or utcnow(), forecasts=tuple(ordered),
    )


def build_dashboard_model(
    *,
    memory: "ResearchMemory",
    objective: str,
    snapshot: "EquitySnapshot",
    verdict: "PanelVerdict",
    forecasts: Sequence[StrategyForecast],
    symbol: str,
    positions: Mapping[str, "Position"] | None = None,
    equity_curve: Sequence[EquityPoint] | Sequence[tuple[datetime, Decimal]] = (),
    starting_equity: Decimal | None = None,
    hedge_reading: "HedgeReading | None" = None,
    title: str = "Portfolio & Forecast Dashboard",
    generated_at: datetime | None = None,
) -> DashboardModel:
    """Assemble the four panels into one immutable :class:`DashboardModel`.

    The single seam between the live trading stack and the renderer: everything
    above computes, everything below (the renderer) only formats.
    """
    return DashboardModel(
        generated_at=generated_at or utcnow(),
        title=title,
        symbol=symbol,
        objective=objective,
        portfolio=build_portfolio_panel(
            snapshot, positions, equity_curve, starting_equity=starting_equity,
        ),
        forecasts=build_forecast_panel(forecasts, symbol, snapshot.ts),
        leaderboard=build_leaderboard(memory, objective),
        experts=build_expert_panel(verdict, hedge_reading),
    )


def _enum_value(item: Any) -> str:
    """The ``.value`` of an enum, or the string of anything else."""
    return getattr(item, "value", str(item))


__all__ = [
    "DashboardModel", "EquityPoint", "ExpertPanel", "ExpertVote",
    "ForecastPanel", "HedgeExpert", "LeaderboardRow", "PortfolioPanel",
    "PortfolioPosition", "StrategyForecast", "StrategyLeaderboard",
    "build_dashboard_model", "build_expert_panel", "build_forecast_panel",
    "build_leaderboard", "build_portfolio_panel", "forecast_from_reading",
]
