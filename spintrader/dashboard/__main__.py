"""Generate a populated dashboard from a real, local demo run.

``python -m spintrader.dashboard [--out artifacts/dashboard.html]``

Everything the dashboard shows is produced here, locally, with no database and no
GPU host:

* an :class:`~spintrader.loop.improvement.ImprovementCycle` over the bundled
  ``data/SPY_1d.csv`` daily bars, across a handful of strategy families, fills
  the leaderboard with real deflated-Sharpe verdicts;
* a small paper :class:`~spintrader.portfolio.ledger.Ledger` session -- buy, mark
  forward over a year of real closes, take a partial profit -- produces a genuine
  equity curve and P&L;
* the persona :class:`~spintrader.agents.panel.PersonaPanel`, voted offline by the
  :class:`~spintrader.loop.voting.BootstrapVoter`, produces the mixture-of-experts
  verdict; and
* each strategy's latest reading over the same bars produces the forecasts, with
  the Hedge ensemble's internal experts shown alongside.

The regime-switching family needs ``hmmlearn`` and stands aside on a machine
without it -- which the forecast panel reports honestly rather than faking a
neutral. No network is touched; the whole thing runs from one CSV.
"""

from __future__ import annotations

import argparse
import logging
from decimal import Decimal
from pathlib import Path

from spintrader.agents.panel import PersonaPanel
from spintrader.agents.personas.baseline_trend import BaselineTrendAgent
from spintrader.agents.personas.hedge import HedgeEnsembleAgent
from spintrader.agents.personas.markov_chain import HighOrderMarkovAgent
from spintrader.agents.personas.mean_reversion import MeanReversionAgent
from spintrader.agents.personas.regime_switch import RegimeSwitchingAgent
from spintrader.agents.personas.roster import default_roster
from spintrader.agents.personas.spec import Horizon
from spintrader.backtest.runner import backtest_instrument, costs_for, load_bars_from_csv
from spintrader.core.types import (
    AssetClass, Bar, Fill, Instrument, Quote, Side, TradingMode,
)
from spintrader.dashboard.model import EquityPoint, build_dashboard_model, forecast_from_reading
from spintrader.dashboard.render import render_html
from spintrader.loop.context import build_context
from spintrader.loop.decision_loop import LiveCursor
from spintrader.loop.improvement import ImprovementCycle
from spintrader.loop.promotion import PromotionGate, TrialLedger
from spintrader.loop.voting import BootstrapVoter, VoteItem
from spintrader.research.factory import (
    CandidateFactory, hedge_family, markov_family, mean_reversion_family, trend_family,
)
from spintrader.research.memory import ResearchMemory
from spintrader.risk.engine import Mandate

log = logging.getLogger(__name__)

# Strategies shown in the forecast panel: (display name, family label, class).
_FORECAST_STRATEGIES: tuple[tuple[str, str, type], ...] = (
    ("baseline_trend_v1", "trend", BaselineTrendAgent),
    ("mean_reversion_v1", "mean_reversion", MeanReversionAgent),
    ("markov_chain_v1", "markov_chain", HighOrderMarkovAgent),
    ("regime_switch_v1", "regime_switch", RegimeSwitchingAgent),
    ("hedge_ensemble_v1", "hedge", HedgeEnsembleAgent),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _demo_factory() -> CandidateFactory:
    """A few candidates across four pure families -- small enough to run fast.

    Kept deliberately narrow: the point is a representative leaderboard, not an
    exhaustive search, and every candidate is a real walk-forward backtest.
    """
    tf = trend_family()
    tf.grid = {"fast_window": (10, 20)}
    mf = mean_reversion_family()
    mf.grid = {"entry_z": ("1.0", "1.5")}
    mk = markov_family()
    mk.grid = {"order": (1,)}
    hf = hedge_family()
    hf.grid = {"eta": ("2.0",)}
    return CandidateFactory(families=[tf, mf, mk, hf])


def _run_improvement(bars, symbol: str, objective: str) -> ResearchMemory:
    """Evaluate the demo candidates and return the populated research memory."""
    gate = PromotionGate(ledger=TrialLedger())
    memory = ResearchMemory()
    cycle = ImprovementCycle(gate=gate, memory=memory, factory=_demo_factory())
    result = cycle.run_round(
        objective, symbol, bars, asset_class=AssetClass.EQUITY,
    )
    log.info("improvement demo: %s", result.summary())
    return memory


def _paper_session(bars, instrument: Instrument):
    """A small paper session over the last ~year of closes.

    Returns ``(snapshot, positions, equity_curve, starting_equity)``. Buys ~60%
    of the book, marks it forward bar by bar recording equity, then takes a
    partial profit three-quarters of the way through so realised P&L and a second
    fee are populated alongside the remaining unrealised position.
    """
    from spintrader.portfolio.ledger import Ledger

    key = instrument.key
    taker_fee = costs_for(AssetClass.EQUITY).taker_fee
    opening = Decimal("100000")
    ledger = Ledger(
        base_currency="USD", mode=TradingMode.PAPER, settlement_days=0,
        opening_cash={"USD": opening},
    )

    window = min(260, max(30, len(bars) // 4))
    entry_idx = len(bars) - window
    entry_bar: Bar = bars[entry_idx]
    price = entry_bar.close
    qty = (opening * Decimal("0.6") / price).quantize(Decimal("0.0001"))
    buy_fee = (qty * price * taker_fee).quantize(Decimal("0.01"))
    ledger.apply_fill(
        Fill(order_id="demo-buy", instrument_key=key, side=Side.BUY, qty=qty,
             price=price, ts=entry_bar.ts, fee=buy_fee, fee_currency="USD"),
        quote_currency="USD", now=entry_bar.ts,
    )

    sell_at = int(window * 0.75)
    curve: list[EquityPoint] = []
    for i, bar in enumerate(bars[entry_idx:]):
        ledger.mark({key: bar.close})
        if i == sell_at:
            sell_qty = (qty * Decimal("0.4")).quantize(Decimal("0.0001"))
            sell_fee = (sell_qty * bar.close * taker_fee).quantize(Decimal("0.01"))
            ledger.apply_fill(
                Fill(order_id="demo-sell", instrument_key=key, side=Side.SELL,
                     qty=sell_qty, price=bar.close, ts=bar.ts, fee=sell_fee,
                     fee_currency="USD"),
                quote_currency="USD", now=bar.ts,
            )
            ledger.mark({key: bar.close})
        snapshot = ledger.value(now=bar.ts)
        curve.append(EquityPoint(ts=bar.ts, equity=snapshot.equity))

    final = ledger.value(now=bars[-1].ts)
    return final, dict(ledger.positions), curve, opening


def _persona_panel(bars, instrument: Instrument):
    """Vote the persona panel offline and aggregate it into a verdict."""
    registry = default_roster()
    panel = PersonaPanel(registry)
    ctx = build_context(
        instrument, bars, interval="1d", horizon=Horizon.DAYS, continuous=False,
    )
    specs = registry.applicable(AssetClass.EQUITY, Horizon.DAYS)
    items = [VoteItem(spec=spec, context=ctx) for spec in specs]
    votes_by_key = BootstrapVoter().vote_all(items)
    votes = votes_by_key.get(instrument.key, [])
    return panel.aggregate(votes, AssetClass.EQUITY, Horizon.DAYS)


def _forecasts(bars, instrument: Instrument):
    """Run each strategy once over the bars; return forecasts and the Hedge reading."""
    last = bars[-1]
    half = last.close * Decimal("5") / Decimal(20_000)
    quote = Quote(instrument_key=instrument.key, ts=last.ts,
                  bid=last.close - half, ask=last.close + half)
    mandate = Mandate.open_mandate([instrument.key], hours=1)

    forecasts = []
    hedge_reading = None
    for name, family, cls in _FORECAST_STRATEGIES:
        agent = cls(interval="1d", continuous=False)
        cursor = LiveCursor(bars, quote)
        try:
            agent.on_bar(cursor, instrument, mandate)
        except Exception as exc:  # noqa: BLE001 - a demo forecast must not abort
            log.warning("forecast for %s failed: %s", name, exc)
        reading = agent.last_reading
        if reading is None:
            continue
        forecasts.append(forecast_from_reading(name, family, reading))
        if family == "hedge":
            hedge_reading = reading
    return forecasts, hedge_reading


def build_demo_dashboard(csv_path: Path, symbol: str, objective: str):
    """Assemble a fully populated :class:`DashboardModel` from a local demo run."""
    instrument = backtest_instrument(symbol, AssetClass.EQUITY)
    bars = load_bars_from_csv(csv_path, instrument.key, interval="1d")
    if len(bars) < 300:
        raise SystemExit(f"{csv_path} has only {len(bars)} bars; need at least 300")

    memory = _run_improvement(bars, symbol, objective)
    snapshot, positions, curve, starting = _paper_session(bars, instrument)
    verdict = _persona_panel(bars, instrument)
    forecasts, hedge_reading = _forecasts(bars, instrument)

    return build_dashboard_model(
        memory=memory, objective=objective, snapshot=snapshot, verdict=verdict,
        forecasts=forecasts, symbol=symbol, positions=positions,
        equity_curve=curve, starting_equity=starting, hedge_reading=hedge_reading,
    )


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        prog="python -m spintrader.dashboard",
        description="Generate a self-contained Portfolio & Forecast dashboard.",
    )
    parser.add_argument(
        "--out", type=Path, default=root / "artifacts" / "dashboard.html",
        help="output HTML path (default: artifacts/dashboard.html)",
    )
    parser.add_argument(
        "--csv", type=Path, default=root / "data" / "SPY_1d.csv",
        help="bar CSV to drive the demo (default: data/SPY_1d.csv)",
    )
    parser.add_argument("--symbol", default="SPY", help="instrument symbol (default: SPY)")
    parser.add_argument(
        "--objective", default="spy-daily", help="improvement objective label",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    print(f"Building dashboard from {args.csv} ...")
    model = build_demo_dashboard(args.csv, args.symbol, args.objective)
    html = render_html(model)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    board = model.leaderboard
    panel = model.experts
    print(f"Wrote {len(html):,} bytes to {out}")
    print(
        f"  leaderboard: {board.evaluated} evaluated, {board.promoted} promoted, "
        f"champion {board.champion_key or 'none'}"
    )
    print(
        f"  panel: {panel.action.upper()} conf {float(panel.confidence):.2f}, "
        f"{panel.participating} experts, dispersion {float(panel.dispersion):.2f}, "
        f"{'escalate' if panel.escalate else 'no escalation'}"
    )
    print(f"  portfolio: equity {float(model.portfolio.equity):,.2f} "
          f"{model.portfolio.base_currency}, {len(model.portfolio.positions)} position(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
