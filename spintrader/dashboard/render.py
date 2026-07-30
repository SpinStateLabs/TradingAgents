"""Render a :class:`DashboardModel` to one self-contained HTML file.

The whole document -- every style rule, the equity-curve chart, the weight bars
-- is emitted as a single string with no external reference of any kind: no
stylesheet link, no script tag pointing elsewhere, no remote font, no image URL.
That constraint is the point. This dashboard is produced on a dev box and opened
from disk, often with no network at all, and a file that silently depends on a
remote asset is a file that renders differently -- or not at all -- depending on
where and when it is opened.

Two consequences worth stating, because they look like omissions otherwise:

* **Inline SVG carries no ``xmlns``.** HTML5 parses inline SVG without a
  namespace declaration, and the usual ``xmlns="http://..."`` would smuggle a
  URL into a document that is meant to have none. So it is left off on purpose.
* **Interactivity is native, not scripted.** The per-expert reasoning expands
  through ``<details>``/``<summary>``; there is no JavaScript to inline because
  none is needed, which is the most self-contained a page can be.

The renderer only *formats*. Every figure it shows was already decided in
:mod:`spintrader.dashboard.model`; nothing here re-derives a number, so the
document cannot disagree with the model it was built from. Money arrives as
:class:`~decimal.Decimal` and is quantised for display here and only here.
"""

from __future__ import annotations

import html
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable

from spintrader.dashboard.model import (
    DashboardModel, ExpertPanel, ForecastPanel, PortfolioPanel,
    StrategyLeaderboard,
)

ZERO = Decimal("0")
_CENTS = Decimal("0.01")


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------

def _esc(value: object) -> str:
    """HTML-escape any value's string form."""
    return html.escape(str(value), quote=True)


def _money(value: Decimal | None, currency: str = "USD") -> str:
    """Format a Decimal as money with thousands separators and a sign.

    Negative amounts read ``-$1,234.56`` (sign outside the symbol), which is how
    a P&L line should read when it is losing money.
    """
    if value is None:
        return "n/a"
    amount = Decimal(value).quantize(_CENTS, rounding=ROUND_HALF_UP)
    sign = "-" if amount < ZERO else ""
    symbol = "$" if currency.upper() == "USD" else f"{_esc(currency)} "
    return f"{sign}{symbol}{abs(amount):,.2f}"


def _pct(value: object, places: int = 2, signed: bool = False) -> str:
    """Format a fraction (0.0123) as a percent (1.23%)."""
    if value is None:
        return "n/a"
    number = float(value) * 100.0
    return f"{number:+.{places}f}%" if signed else f"{number:.{places}f}%"


def _num(value: object, places: int = 2, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{number:+.{places}f}" if signed else f"{number:.{places}f}"


def _action_class(action: str) -> str:
    """A CSS class for an action, so buy/sell/hold read at a glance in colour."""
    action = (action or "").lower()
    if action == "buy":
        return "act-buy"
    if action in ("sell", "close"):
        return "act-sell"
    return "act-hold"


def _direction_class(direction: str) -> str:
    direction = (direction or "").lower()
    return {
        "long": "act-buy", "short": "act-sell",
    }.get(direction, "act-hold")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _stat(label: str, value: str, tone: str = "") -> str:
    cls = f"stat {tone}".strip()
    return (
        f'<div class="{cls}"><div class="stat-label">{_esc(label)}</div>'
        f'<div class="stat-value">{value}</div></div>'
    )


def _pnl_tone(value: Decimal | float | None) -> str:
    if value is None:
        return ""
    return "pos" if float(value) >= 0 else "neg"


def _portfolio_section(panel: PortfolioPanel) -> str:
    cur = panel.base_currency
    incomplete = (
        '' if panel.complete
        else '<span class="warn-pill">partial valuation</span>'
    )
    stats = "".join([
        _stat("Equity", _money(panel.equity, cur)),
        _stat("Cash", _money(panel.cash, cur)),
        _stat("Positions", _money(panel.positions_value, cur)),
        _stat("Realised P&amp;L", _money(panel.realized_pnl, cur),
              _pnl_tone(panel.realized_pnl)),
        _stat("Unrealised P&amp;L", _money(panel.unrealized_pnl, cur),
              _pnl_tone(panel.unrealized_pnl)),
        _stat("Net P&amp;L", _money(panel.net_pnl, cur), _pnl_tone(panel.net_pnl)),
        _stat("Fees paid", _money(panel.fees_paid, cur), "neg"),
        _stat("Gross exposure", _pct(panel.gross_exposure)),
    ])
    tr = panel.total_return
    ret_line = (
        f'<p class="muted">Total return since inception '
        f'<strong class="{_pnl_tone(tr)}">{_pct(tr, signed=True)}</strong> '
        f'&middot; started at {_money(panel.starting_equity, cur)}</p>'
        if tr is not None else ""
    )

    positions = _positions_table(panel) if panel.positions else (
        '<p class="muted">No open positions.</p>'
    )

    return f"""
    <section class="card">
      <h2>Portfolio Summary {incomplete}</h2>
      <div class="stat-grid">{stats}</div>
      {ret_line}
      <h3>Open positions</h3>
      {positions}
    </section>
    <section class="card">
      <h2>Equity Curve</h2>
      {_equity_curve_svg(panel)}
    </section>
    """


def _positions_table(panel: PortfolioPanel) -> str:
    cur = panel.base_currency
    body = "".join(
        f"<tr><td>{_esc(p.instrument_key)}</td>"
        f"<td class='num'>{_num(p.qty, 4)}</td>"
        f"<td class='num'>{_money(p.avg_cost, cur)}</td>"
        f"<td class='num'>{_money(p.last_price, cur)}</td>"
        f"<td class='num'>{_money(p.market_value, cur)}</td>"
        f"<td class='num {_pnl_tone(p.unrealized_pnl)}'>"
        f"{_money(p.unrealized_pnl, cur)}</td></tr>"
        for p in panel.positions
    )
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Instrument</th><th class="num">Qty</th><th class="num">Avg cost</th>
          <th class="num">Mark</th><th class="num">Market value</th>
          <th class="num">Unrealised</th>
        </tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>"""


def _equity_curve_svg(panel: PortfolioPanel) -> str:
    """A responsive inline SVG line chart of the equity curve.

    No ``xmlns`` (HTML5 inline SVG needs none) so the document stays URL-free;
    the ``viewBox`` plus a CSS ``width:100%`` makes it scale to any width.
    """
    points = panel.equity_curve
    if len(points) < 2:
        return '<p class="muted">Not enough samples to plot an equity curve.</p>'

    values = [float(p.equity) for p in points]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    width, height, pad = 1000.0, 280.0, 12.0
    inner_h = height - 2 * pad
    n = len(values)

    coords = []
    for i, v in enumerate(values):
        x = (i / (n - 1)) * width
        y = pad + (1.0 - (v - lo) / span) * inner_h
        coords.append((x, y))

    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    area = (
        f"M{coords[0][0]:.1f},{height - pad:.1f} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in coords)
        + f" L{coords[-1][0]:.1f},{height - pad:.1f} Z"
    )
    cur = panel.base_currency
    first, last = values[0], values[-1]
    tone = "pos" if last >= first else "neg"

    return f"""
    <div class="chart">
      <svg viewBox="0 0 {width:.0f} {height:.0f}" preserveAspectRatio="none"
           role="img" aria-label="Equity curve">
        <path class="area" d="{area}"></path>
        <polyline class="line" points="{line}"></polyline>
      </svg>
      <div class="chart-axis">
        <span>{_esc(points[0].ts.strftime('%Y-%m-%d'))} &middot; {_money(Decimal(str(first)), cur)}</span>
        <span class="{tone}">{_esc(points[-1].ts.strftime('%Y-%m-%d'))} &middot; {_money(Decimal(str(last)), cur)}</span>
      </div>
      <div class="chart-range muted">low {_money(Decimal(str(lo)), cur)} &middot; high {_money(Decimal(str(hi)), cur)}</div>
    </div>"""


def _leaderboard_section(board: StrategyLeaderboard) -> str:
    champ = (
        f'Champion <strong>{_esc(board.champion_key)}</strong> '
        f'(Sharpe {_num(board.champion_sharpe)})'
        if board.champion_key else "No candidate cleared the promotion gate."
    )
    rejections = " ".join(
        f'<span class="pill">{_esc(reason.replace("_", " "))}: {count}</span>'
        for reason, count in sorted(
            board.rejection_profile.items(), key=lambda kv: -kv[1]
        )
    ) or '<span class="muted">none recorded</span>'

    if board.rows:
        rows = "".join(_leaderboard_row(r) for r in board.rows)
        table = f"""
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>Family</th><th class="num">Sharpe</th><th class="num">DSR</th>
              <th class="num">Return</th><th class="num">Max DD</th>
              <th>Verdict</th><th>Reason</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""
    else:
        table = '<p class="muted">No candidates were evaluated.</p>'

    return f"""
    <section class="card">
      <h2>Strategy Leaderboard</h2>
      <p class="muted">{board.evaluated} evaluated &middot; {board.promoted} promoted
         &middot; objective <code>{_esc(board.objective)}</code></p>
      <p>{champ}</p>
      {table}
      <h3>Why candidates were refused</h3>
      <div class="pill-row">{rejections}</div>
    </section>"""


def _leaderboard_row(row) -> str:
    verdict = (
        '<span class="badge badge-ok">promoted</span>' if row.promoted
        else '<span class="badge badge-no">rejected</span>'
    )
    star = ' <span class="star" title="current champion">&#9733;</span>' if row.is_champion else ""
    return (
        f"<tr class='{'champ' if row.is_champion else ''}'>"
        f"<td>{_esc(row.family)}{star}</td>"
        f"<td class='num'>{_num(row.sharpe)}</td>"
        f"<td class='num'>{_num(row.deflated_sharpe, 3)}</td>"
        f"<td class='num {_pnl_tone(row.total_return)}'>{_pct(row.total_return, signed=True)}</td>"
        f"<td class='num'>{_pct(row.max_drawdown)}</td>"
        f"<td>{verdict}</td>"
        f"<td class='reason'>{_esc(row.reason)}</td></tr>"
    )


def _expert_section(panel: ExpertPanel) -> str:
    head = (
        f'<span class="badge {_action_class(panel.action)}">{_esc(panel.action.upper())}</span> '
        f'confidence <strong>{_num(panel.confidence)}</strong>'
    )
    escalate = (
        '<span class="warn-pill">escalate to deep model</span>'
        if panel.escalate else ""
    )
    diagnostics = "".join([
        _stat("Net direction", _num(panel.net_direction, 2, signed=True)),
        _stat("Dispersion", _num(panel.dispersion)),
        _stat("Lens diversity", str(panel.lens_diversity)),
        _stat("Participating", str(panel.participating)),
    ])

    max_weight = max((float(e.weight) for e in panel.experts), default=1.0) or 1.0
    experts = "".join(_expert_row(e, max_weight) for e in panel.experts) or (
        '<p class="muted">No experts voted.</p>'
    )

    hedge = _hedge_block(panel) if panel.hedge_experts else ""
    abst = _abstentions_block(panel) if panel.abstentions else ""

    return f"""
    <section class="card">
      <h2>Expert Panel</h2>
      <p class="muted">Mixture-of-experts verdict with each expert's chain of thought.</p>
      <p>{head} {escalate}</p>
      <div class="stat-grid small">{diagnostics}</div>
      <h3>Persona experts &middot; weight, vote &amp; reasoning</h3>
      <div class="experts">{experts}</div>
      {hedge}
      {abst}
    </section>"""


def _expert_row(expert, max_weight: float) -> str:
    pct = max(2.0, float(expert.weight) / max_weight * 100.0)
    lenses = ", ".join(expert.lenses) or "-"
    conf = _num(expert.confidence)
    state = "abstained" if expert.abstained else expert.action.upper()
    detail = (
        f"<details><summary>reasoning</summary>"
        f"<p class='cot'>{_esc(expert.rationale) or 'no rationale given'}</p>"
        f"<p class='cot muted'>Would change its mind on: "
        f"{_esc(expert.changed_by) or 'unstated'}</p></details>"
        if (expert.rationale or expert.changed_by) else ""
    )
    return f"""
      <div class="expert {'muted-row' if expert.abstained else ''}">
        <div class="expert-head">
          <span class="expert-name">{_esc(expert.persona_key)}</span>
          <span class="badge {_action_class(expert.action)}">{_esc(state)}</span>
        </div>
        <div class="bar"><span class="bar-fill {_action_class(expert.action)}"
             style="width:{pct:.1f}%"></span></div>
        <div class="expert-meta muted">weight {_num(expert.weight, 3)}
             &middot; effective {_num(expert.effective, 3, signed=True)}
             &middot; conf {conf} &middot; lenses: {_esc(lenses)}</div>
        {detail}
      </div>"""


def _hedge_block(panel: ExpertPanel) -> str:
    total = sum(float(h.weight) for h in panel.hedge_experts) or 1.0
    rows = ""
    for h in panel.hedge_experts:
        pct = float(h.weight) / total * 100.0
        vote_cls = "act-buy" if h.vote > 0 else "act-sell" if h.vote < 0 else "act-hold"
        rows += (
            f"<div class='expert'>"
            f"<div class='expert-head'><span class='expert-name'>{_esc(h.name)}</span>"
            f"<span class='muted'>vote {_num(h.vote, 2, signed=True)}</span></div>"
            f"<div class='bar'><span class='bar-fill {vote_cls}' "
            f"style='width:{max(2.0, pct):.1f}%'></span></div>"
            f"<div class='expert-meta muted'>Hedge weight {h.weight * 100:.1f}%</div>"
            f"</div>"
        )
    return f"""
      <h3>Hedge ensemble &middot; no-regret sub-experts</h3>
      <p class="muted">Multiplicative weights the meta-learner has placed on its
         momentum / reversion / breakout experts.</p>
      <div class="experts">{rows}</div>"""


def _abstentions_block(panel: ExpertPanel) -> str:
    items = "".join(
        f"<li><strong>{_esc(key)}</strong>: {_esc(reason)}</li>"
        for key, reason in panel.abstentions
    )
    return f"""
      <details class="abstain">
        <summary>{len(panel.abstentions)} personas abstained</summary>
        <ul class="abstain-list">{items}</ul>
      </details>"""


def _forecast_section(panel: ForecastPanel) -> str:
    if not panel.forecasts:
        cards = '<p class="muted">No strategy forecasts available.</p>'
    else:
        cards = "".join(_forecast_card(f) for f in panel.forecasts)
    return f"""
    <section class="card">
      <h2>Forecast Panel</h2>
      <p class="muted">Each strategy's latest read on {_esc(panel.symbol)}:
         edge, confidence and direction.</p>
      <div class="forecast-grid">{cards}</div>
    </section>"""


def _forecast_card(f) -> str:
    conf_pct = max(0.0, min(100.0, float(f.confidence) * 100.0))
    return f"""
      <div class="forecast">
        <div class="forecast-head">
          <span class="forecast-name">{_esc(f.name)}</span>
          <span class="badge {_direction_class(f.direction)}">{_esc(f.direction.upper())}</span>
        </div>
        <div class="forecast-body">
          <div class="fc-row"><span>edge</span><strong>{_pct(f.edge, signed=True)}</strong></div>
          <div class="fc-row"><span>confidence</span><strong>{_num(f.confidence)}</strong></div>
          <div class="bar"><span class="bar-fill {_direction_class(f.direction)}"
               style="width:{conf_pct:.1f}%"></span></div>
          <div class="fc-row"><span>family</span><code>{_esc(f.family)}</code></div>
          <p class="muted fc-note">{_esc(f.note)}</p>
        </div>
      </div>"""


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

def render_html(model: DashboardModel) -> str:
    """Render a :class:`DashboardModel` to one complete, standalone HTML string."""
    generated = model.generated_at.strftime("%Y-%m-%d %H:%M UTC")
    body = "".join([
        _portfolio_section(model.portfolio),
        _forecast_section(model.forecasts),
        _expert_section(model.experts),
        _leaderboard_section(model.leaderboard),
    ])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(model.title)} &middot; {_esc(model.symbol)}</title>
<style>{_CSS}</style>
</head>
<body>
<header class="masthead">
  <h1>{_esc(model.title)}</h1>
  <p class="sub">{_esc(model.symbol)} &middot; objective <code>{_esc(model.objective)}</code>
     &middot; generated {_esc(generated)}</p>
</header>
<main>
{body}
</main>
<footer class="foot muted">
  SpinTrader dashboard &middot; self-contained, no external resources &middot; all values in the ledger base currency.
</footer>
</body>
</html>
"""


# The entire stylesheet, inlined. Theme-aware via prefers-color-scheme, with an
# explicit data-theme override hook so a viewer's toggle can win in both
# directions; responsive via a fluid grid and a single narrow-screen breakpoint.
_CSS = """
:root{
  --bg:#f6f7f9; --card:#ffffff; --ink:#1a1d21; --muted:#5b6470;
  --line:#e3e7ec; --accent:#2f6df6; --accent-soft:rgba(47,109,246,.14);
  --pos:#12855a; --neg:#c8394a; --hold:#7a828c;
  --shadow:0 1px 3px rgba(16,24,40,.08),0 1px 2px rgba(16,24,40,.04);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0f1216; --card:#171b21; --ink:#e8ebef; --muted:#98a2b0;
    --line:#252b33; --accent:#5b8cff; --accent-soft:rgba(91,140,255,.16);
    --pos:#3ecf8e; --neg:#ff6b7d; --hold:#8b93a0;
    --shadow:0 1px 3px rgba(0,0,0,.5);
  }
}
:root[data-theme="light"]{
  --bg:#f6f7f9; --card:#ffffff; --ink:#1a1d21; --muted:#5b6470;
  --line:#e3e7ec; --accent:#2f6df6; --accent-soft:rgba(47,109,246,.14);
  --pos:#12855a; --neg:#c8394a; --hold:#7a828c; --shadow:0 1px 3px rgba(16,24,40,.08);
}
:root[data-theme="dark"]{
  --bg:#0f1216; --card:#171b21; --ink:#e8ebef; --muted:#98a2b0;
  --line:#252b33; --accent:#5b8cff; --accent-soft:rgba(91,140,255,.16);
  --pos:#3ecf8e; --neg:#ff6b7d; --hold:#8b93a0; --shadow:0 1px 3px rgba(0,0,0,.5);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;
}
code{font-family:ui-monospace,"SFMono-Regular",Menlo,Consolas,monospace;font-size:.9em;
  background:var(--accent-soft);padding:1px 5px;border-radius:5px}
.masthead{padding:28px 20px 8px;max-width:1120px;margin:0 auto}
.masthead h1{margin:0;font-size:1.7rem;letter-spacing:-.02em}
.sub{color:var(--muted);margin:.35rem 0 0}
main{max-width:1120px;margin:0 auto;padding:12px 20px 40px;
  display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;
  padding:20px 22px;box-shadow:var(--shadow)}
.card:first-child,.card:nth-child(3),.card:nth-child(4){grid-column:1 / -1}
h2{margin:0 0 12px;font-size:1.15rem;letter-spacing:-.01em}
h3{margin:20px 0 10px;font-size:.95rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.muted{color:var(--muted)}
.pos{color:var(--pos)} .neg{color:var(--neg)}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.stat-grid.small{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-value{font-size:1.1rem;font-weight:650;margin-top:3px;font-variant-numeric:tabular-nums}
.stat.pos .stat-value{color:var(--pos)} .stat.neg .stat-value{color:var(--neg)}
.table-wrap{overflow-x:auto;margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.reason{white-space:normal;color:var(--muted);min-width:180px}
tr.champ{background:var(--accent-soft)}
.star{color:var(--accent)}
.chart{margin-top:8px}
.chart svg{width:100%;height:auto;display:block;background:var(--bg);
  border:1px solid var(--line);border-radius:10px}
.chart .line{fill:none;stroke:var(--accent);stroke-width:2.5;
  vector-effect:non-scaling-stroke;stroke-linejoin:round;stroke-linecap:round}
.chart .area{fill:var(--accent-soft);stroke:none}
.chart-axis{display:flex;justify-content:space-between;font-size:.8rem;margin-top:6px;
  font-variant-numeric:tabular-nums}
.chart-range{font-size:.78rem;margin-top:2px}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.72rem;
  font-weight:700;letter-spacing:.02em}
.badge-ok,.act-buy{background:rgba(18,133,90,.16);color:var(--pos)}
.badge-no,.act-sell{background:rgba(200,57,74,.16);color:var(--neg)}
.act-hold{background:rgba(122,130,140,.18);color:var(--hold)}
.badge.act-buy,.badge.act-sell,.badge.act-hold{}
.pill,.warn-pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:.75rem;
  border:1px solid var(--line);background:var(--bg);color:var(--muted);margin:2px}
.warn-pill{border-color:var(--neg);color:var(--neg);background:rgba(200,57,74,.08);
  font-weight:600;margin-left:8px}
.pill-row{display:flex;flex-wrap:wrap;gap:2px}
.experts{display:flex;flex-direction:column;gap:14px}
.expert{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--bg)}
.expert.muted-row{opacity:.6}
.expert-head{display:flex;align-items:center;justify-content:space-between;gap:8px}
.expert-name{font-weight:650}
.bar{height:9px;background:var(--line);border-radius:999px;overflow:hidden;margin:9px 0 7px}
.bar-fill{display:block;height:100%;border-radius:999px;background:var(--accent)}
.bar-fill.act-buy{background:var(--pos)} .bar-fill.act-sell{background:var(--neg)}
.bar-fill.act-hold{background:var(--hold)}
.expert-meta{font-size:.8rem;font-variant-numeric:tabular-nums}
details{margin-top:8px} summary{cursor:pointer;color:var(--accent);font-size:.82rem}
.cot{margin:8px 0 0;font-size:.88rem;line-height:1.55}
.abstain{margin-top:14px} .abstain-list{margin:8px 0 0;padding-left:20px;color:var(--muted);
  font-size:.85rem;line-height:1.7}
.forecast-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.forecast{border:1px solid var(--line);border-radius:12px;padding:14px;background:var(--bg)}
.forecast-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px}
.forecast-name{font-weight:650}
.fc-row{display:flex;justify-content:space-between;font-size:.86rem;padding:3px 0;
  font-variant-numeric:tabular-nums}
.fc-note{font-size:.78rem;margin:8px 0 0}
.foot{max-width:1120px;margin:0 auto;padding:0 20px 40px;font-size:.8rem;text-align:center}
@media (max-width:820px){
  main{grid-template-columns:1fr}
  .card:nth-child(3),.card:nth-child(4){grid-column:auto}
}
"""


__all__ = ["render_html"]
