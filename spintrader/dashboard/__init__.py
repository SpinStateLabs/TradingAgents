"""Self-contained Portfolio & Forecast dashboard for SpinTrader.

Two clean halves:

* :mod:`spintrader.dashboard.model` -- pure aggregation from the live trading
  objects (research memory, ledger, persona panel, strategy readings) into an
  immutable :class:`~spintrader.dashboard.model.DashboardModel`. No HTML, fully
  unit-testable.
* :mod:`spintrader.dashboard.render` -- one :class:`DashboardModel` to one
  standalone HTML file, every style and chart inlined, no external resource.

Run ``python -m spintrader.dashboard`` to build a populated dashboard from a real
local demo (an improvement cycle over the bundled SPY bars, a small paper
session, a persona panel and each strategy's latest reading) -- no database and
no GPU host required.
"""

from __future__ import annotations

from spintrader.dashboard.model import (
    DashboardModel, ForecastPanel, ExpertPanel, PortfolioPanel,
    StrategyForecast, StrategyLeaderboard, build_dashboard_model,
)
from spintrader.dashboard.render import render_html

__all__ = [
    "DashboardModel", "ExpertPanel", "ForecastPanel", "PortfolioPanel",
    "StrategyForecast", "StrategyLeaderboard", "build_dashboard_model",
    "render_html",
]
