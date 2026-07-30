"""Tests for the Portfolio & Forecast dashboard. No network, no LLM, no DB.

Two halves, matching the module's own split:

* the model builders are exercised against deliberately trivial fake inputs, so
  the aggregation is pinned down without dragging in a backtest or a ledger run;
* the renderer is checked as a smoke test -- non-empty, structurally complete,
  and above all *self-contained*: it must not reference ``http://``, ``https://``
  or a CDN, because a dashboard that reaches out to the network is not the
  offline artefact it claims to be.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from spintrader.agents.panel import PanelVerdict, PersonaVote
from spintrader.agents.personas.spec import Lens
from spintrader.core.types import Action, Position
from spintrader.portfolio.ledger import EquitySnapshot
from spintrader.research.memory import ResearchMemory, TrialRecord
from spintrader.dashboard.model import (
    EquityPoint, StrategyForecast, build_dashboard_model, build_expert_panel,
    build_forecast_panel, build_leaderboard, build_portfolio_panel,
    forecast_from_reading,
)
from spintrader.dashboard.render import render_html

D = Decimal
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _snapshot(**over) -> EquitySnapshot:
    kw = dict(
        ts=T0, base_currency="USD", cash=D("40000"), positions_value=D("62000"),
        equity=D("102000"), realized_pnl=D("1500"), unrealized_pnl=D("2000"),
        fees_paid=D("35"), gross_exposure=D("0.61"), complete=True,
    )
    kw.update(over)
    return EquitySnapshot(**kw)


def _positions():
    pos = Position(instrument_key="paper:SPY", qty=D("100"), avg_cost=D("600"))
    pos.last_price = D("620")
    flat = Position(instrument_key="paper:AAA", qty=D("0"), avg_cost=D("0"))
    return {"paper:SPY": pos, "paper:AAA": flat}


def _curve():
    return [EquityPoint(ts=T0 + timedelta(days=i), equity=D("100000") + D(i) * D("100"))
            for i in range(6)]


def _memory() -> ResearchMemory:
    m = ResearchMemory()
    m.record(TrialRecord(
        objective="obj", config_key="win", config={}, promoted=True,
        deflated_sharpe=1.2, n_trials=3, family="trend", sharpe=2.1,
        total_return=0.18, max_drawdown=-0.05, rejections=[], note="cleared all gates",
    ))
    m.record(TrialRecord(
        objective="obj", config_key="mid", config={}, promoted=False,
        deflated_sharpe=0.4, n_trials=3, family="mean_reversion", sharpe=1.0,
        total_return=0.05, max_drawdown=-0.08, rejections=["not_significant"],
    ))
    m.record(TrialRecord(
        objective="obj", config_key="bad", config={}, promoted=False,
        deflated_sharpe=0.0, n_trials=3, family="hedge", sharpe=-0.3,
        total_return=-0.04, max_drawdown=-0.2,
        rejections=["negative_return", "not_significant"],
    ))
    return m


def _verdict() -> PanelVerdict:
    votes = [
        PersonaVote(persona_key="simons", action=Action.BUY, confidence=D("0.7"),
                    rationale="statistical edge holds after costs", changed_by="edge decays",
                    lenses=(Lens.STATISTICAL,), weight=D("1.0")),
        PersonaVote(persona_key="taleb", action=Action.SELL, confidence=D("0.5"),
                    rationale="convex risk to the downside", changed_by="tail thins",
                    lenses=(Lens.TAIL,), weight=D("0.75")),
        PersonaVote(persona_key="turtle", action=Action.HOLD, confidence=D("0"),
                    rationale="no breakout", lenses=(Lens.STATISTICAL,), weight=D("0.5")),
    ]
    return PanelVerdict(
        action=Action.BUY, confidence=D("0.42"), net_direction=D("0.31"),
        votes=votes, abstentions={"berkshire": "needs a years horizon"},
        dispersion=D("0.45"), lens_diversity=2, participating=2, escalate=True,
    )


def _hedge_reading():
    return SimpleNamespace(weights=(0.5, 0.3, 0.2), votes=(1.0, -1.0, 0.0))


def _forecasts():
    return [
        forecast_from_reading("baseline_trend_v1", "trend", SimpleNamespace(
            edge=0.02, confidence=0.72, bullish=True, trend_strength=0.05, annual_vol=0.18)),
        forecast_from_reading("mean_reversion_v1", "mean_reversion", SimpleNamespace(
            edge=0.01, confidence=0.61, oversold=False, gap=0.004, annual_vol=0.2)),
        forecast_from_reading("regime_switch_v1", "regime_switch", SimpleNamespace(
            edge=0.0, confidence=0.0, fitted=False, drift=0.0)),
    ]


def _model():
    return build_dashboard_model(
        memory=_memory(), objective="obj", snapshot=_snapshot(), verdict=_verdict(),
        forecasts=_forecasts(), symbol="SPY", positions=_positions(),
        equity_curve=_curve(), starting_equity=D("100000"),
        hedge_reading=_hedge_reading(),
    )


# --------------------------------------------------------------------------
# Portfolio panel
# --------------------------------------------------------------------------

class PortfolioPanelTests(unittest.TestCase):
    def test_mirrors_snapshot_fields(self):
        panel = build_portfolio_panel(_snapshot(), _positions(), _curve(),
                                      starting_equity=D("100000"))
        self.assertEqual(panel.equity, D("102000"))
        self.assertEqual(panel.cash, D("40000"))
        self.assertEqual(panel.fees_paid, D("35"))
        # net_pnl = realized + unrealized - fees
        self.assertEqual(panel.net_pnl, D("1500") + D("2000") - D("35"))
        self.assertTrue(panel.complete)

    def test_flat_positions_are_dropped_and_valued(self):
        panel = build_portfolio_panel(_snapshot(), _positions())
        self.assertEqual(len(panel.positions), 1)          # the flat AAA is gone
        spy = panel.positions[0]
        self.assertEqual(spy.instrument_key, "paper:SPY")
        self.assertEqual(spy.market_value, D("62000"))     # 100 * 620
        self.assertEqual(spy.unrealized_pnl, D("2000"))    # (620-600)*100

    def test_total_return_from_starting_equity(self):
        panel = build_portfolio_panel(_snapshot(), None, _curve(),
                                      starting_equity=D("100000"))
        self.assertEqual(panel.total_return, D("102000") / D("100000") - D("1"))

    def test_equity_curve_accepts_tuples(self):
        panel = build_portfolio_panel(_snapshot(), None,
                                      [(T0, D("100")), (T0 + timedelta(days=1), D("110"))])
        self.assertEqual(len(panel.equity_curve), 2)
        self.assertEqual(panel.equity_curve[0].equity, D("100"))


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------

class LeaderboardTests(unittest.TestCase):
    def test_rows_are_sorted_by_sharpe_desc(self):
        board = build_leaderboard(_memory(), "obj")
        sharpes = [r.sharpe for r in board.rows]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))
        self.assertEqual(board.rows[0].config_key, "win")

    def test_champion_and_counts(self):
        board = build_leaderboard(_memory(), "obj")
        self.assertEqual(board.evaluated, 3)
        self.assertEqual(board.promoted, 1)
        self.assertEqual(board.champion_key, "win")
        self.assertTrue(board.rows[0].is_champion)

    def test_rejection_reason_is_carried(self):
        board = build_leaderboard(_memory(), "obj")
        bad = next(r for r in board.rows if r.config_key == "bad")
        self.assertFalse(bad.promoted)
        self.assertIn("negative return", bad.reason)
        self.assertIn("not significant", bad.reason)

    def test_promoted_reason_uses_note(self):
        board = build_leaderboard(_memory(), "obj")
        win = next(r for r in board.rows if r.config_key == "win")
        self.assertTrue(win.promoted)
        self.assertIn("cleared", win.reason)


# --------------------------------------------------------------------------
# Expert panel
# --------------------------------------------------------------------------

class ExpertPanelTests(unittest.TestCase):
    def test_verdict_diagnostics_carried(self):
        panel = build_expert_panel(_verdict())
        self.assertEqual(panel.action, "buy")
        self.assertEqual(panel.dispersion, D("0.45"))
        self.assertEqual(panel.lens_diversity, 2)
        self.assertTrue(panel.escalate)
        self.assertEqual(panel.participating, 2)

    def test_experts_sorted_by_weight_and_flags_abstention(self):
        panel = build_expert_panel(_verdict())
        weights = [e.weight for e in panel.experts]
        self.assertEqual(weights, sorted(weights, reverse=True))
        turtle = next(e for e in panel.experts if e.persona_key == "turtle")
        self.assertTrue(turtle.abstained)             # HOLD + 0 confidence
        self.assertEqual(panel.abstentions[0][0], "berkshire")

    def test_hedge_experts_attached(self):
        panel = build_expert_panel(_verdict(), _hedge_reading())
        self.assertEqual(len(panel.hedge_experts), 3)
        self.assertEqual(panel.hedge_experts[0].name, "momentum")
        self.assertAlmostEqual(panel.hedge_experts[0].weight, 0.5)


# --------------------------------------------------------------------------
# Forecasts
# --------------------------------------------------------------------------

class ForecastTests(unittest.TestCase):
    def test_bullish_reading_is_long(self):
        fc = forecast_from_reading("t", "trend", SimpleNamespace(
            edge=0.02, confidence=0.7, bullish=True, trend_strength=0.05))
        self.assertEqual(fc.direction, "long")
        self.assertTrue(fc.active)
        self.assertAlmostEqual(fc.strength, 0.05)

    def test_unfitted_regime_stands_aside(self):
        fc = forecast_from_reading("r", "regime_switch", SimpleNamespace(
            edge=0.0, confidence=0.0, fitted=False))
        self.assertEqual(fc.direction, "aside")
        self.assertFalse(fc.active)
        self.assertIn("standing aside", fc.note)

    def test_flat_when_no_entry_signal(self):
        fc = forecast_from_reading("m", "markov_chain", SimpleNamespace(
            edge=0.005, confidence=0.6, bullish=False, expected_return=-0.001))
        self.assertEqual(fc.direction, "flat")
        self.assertFalse(fc.active)

    def test_forecast_panel_orders_active_first(self):
        panel = build_forecast_panel(_forecasts(), "SPY", T0)
        self.assertTrue(panel.forecasts[0].active)
        self.assertEqual(panel.forecasts[-1].direction, "aside")


# --------------------------------------------------------------------------
# End-to-end model
# --------------------------------------------------------------------------

class DashboardModelTests(unittest.TestCase):
    def test_assembles_all_four_panels(self):
        model = _model()
        self.assertEqual(model.symbol, "SPY")
        self.assertEqual(model.portfolio.equity, D("102000"))
        self.assertEqual(model.leaderboard.champion_key, "win")
        self.assertEqual(model.experts.action, "buy")
        self.assertEqual(len(model.experts.hedge_experts), 3)
        self.assertGreaterEqual(len(model.forecasts.forecasts), 3)


# --------------------------------------------------------------------------
# Renderer smoke test
# --------------------------------------------------------------------------

class RenderTests(unittest.TestCase):
    HEADERS = (
        "Portfolio &amp; Forecast Dashboard", "Portfolio Summary", "Equity Curve",
        "Strategy Leaderboard", "Expert Panel", "Forecast Panel",
    )

    def setUp(self):
        self.html = render_html(_model())

    def test_non_empty_and_well_formed(self):
        self.assertGreater(len(self.html), 2000)
        self.assertTrue(self.html.lstrip().startswith("<!DOCTYPE html>"))
        self.assertIn("</html>", self.html)

    def test_no_external_references(self):
        lowered = self.html.lower()
        for needle in ("http://", "https://", "cdn"):
            self.assertNotIn(needle, lowered, f"found external reference {needle!r}")
        # No linked assets or remote scripts either.
        self.assertNotIn("<link", lowered)
        self.assertNotIn("src=", lowered)

    def test_contains_section_headers(self):
        for header in self.HEADERS:
            self.assertIn(header, self.html, f"missing section {header!r}")

    def test_shows_money_and_theme_support(self):
        self.assertIn("$", self.html)                       # money formatted
        self.assertIn("prefers-color-scheme", self.html)    # dark/light aware
        self.assertIn("viewBox", self.html)                 # inline SVG chart


if __name__ == "__main__":
    unittest.main()
