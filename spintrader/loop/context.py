"""Market context: the snapshot both loops reason from.

The two-tier loop has two consumers of "what does the market look like right
now": the fast quant loop, which turns it into a trade intent, and the slow LLM
loop, which turns it into a mandate. Computing that snapshot in one place keeps
the two views consistent -- a fast loop acting on one picture while the mandate
was formed from another is how a system talks itself into a trade its own
mandate would forbid.

Everything here is causal and read-only. It reads trailing bars, computes
trailing statistics, and optionally infers the current regime with the forward
(filtered) algorithm -- never the smoothed one, which would be lookahead (see
:mod:`spintrader.quant.regime`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

from spintrader.agents.personas.spec import Horizon
from spintrader.core.types import AssetClass, Bar, Instrument, to_decimal
from spintrader.quant.features import build_features, log_returns, periods_per_year, rolling_std

log = logging.getLogger(__name__)

ZERO = Decimal("0")


@dataclass(slots=True)
class MarketContext:
    """A causal snapshot of one instrument, shared by both loops.

    The trend and volatility figures are the same trailing statistics the
    quant persona computes, exposed here so the mandate is formed from the
    identical numbers the fast loop will act on.
    """
    instrument: Instrument
    interval: str
    horizon: Horizon
    bars: list[Bar] = field(default_factory=list)
    last_close: Decimal = ZERO
    trend_strength: float = 0.0      # fast MA / slow MA - 1
    annual_vol: float = 0.0          # trailing realised, annualised
    regime_label: str | None = None
    regime_risk: Decimal = ZERO      # 0 benign .. 1 crisis
    regime_confidence: float = 0.0

    @property
    def asset_class(self) -> AssetClass:
        return self.instrument.asset_class

    @property
    def instrument_key(self) -> str:
        return self.instrument.key

    def ready(self, min_bars: int) -> bool:
        return len(self.bars) >= min_bars

    def summary(self) -> str:
        """One-line human-readable snapshot, for an LLM prompt or a log."""
        direction = "up" if self.trend_strength > 0 else "down" if self.trend_strength < 0 else "flat"
        regime = f", regime={self.regime_label}({self.regime_risk:.2f})" if self.regime_label else ""
        return (
            f"{self.instrument.symbol} {self.interval}: last {self.last_close}, "
            f"trend {self.trend_strength:+.3%} ({direction}), "
            f"annualised vol {self.annual_vol:.1%}{regime} "
            f"over {len(self.bars)} bars"
        )

    def features(self) -> dict[str, float]:
        """Structured numbers, for a schema-constrained LLM call or a voter."""
        return {
            "last_close": float(self.last_close),
            "trend_strength": self.trend_strength,
            "annual_vol": self.annual_vol,
            "regime_risk": float(self.regime_risk),
            "regime_confidence": self.regime_confidence,
            "n_bars": len(self.bars),
        }


def _trend_and_vol(
    bars: Sequence[Bar], fast_window: int, slow_window: int,
    vol_window: int, interval: str, continuous: bool,
) -> tuple[float, float]:
    """Trailing trend strength and annualised volatility from closes.

    Trend is fast MA over slow MA minus one; volatility is the trailing realised
    standard deviation of log returns, annualised. Both use only the final value
    of causal rolling statistics, so they are observable at the last bar and no
    earlier.
    """
    import numpy as np

    closes = np.array([float(b.close) for b in bars], dtype=float)
    if closes.size < 2 or np.any(closes <= 0):
        return 0.0, 0.0
    fast = float(closes[-min(fast_window, closes.size):].mean())
    slow = float(closes[-min(slow_window, closes.size):].mean())
    trend = (fast / slow - 1.0) if slow > 0 else 0.0

    scale = periods_per_year(interval, continuous=continuous) ** 0.5
    returns = log_returns(closes)
    annual_vol = float(rolling_std(returns, min(vol_window, returns.size))[-1] * scale)
    return trend, annual_vol


def assess_regime(
    bars: Sequence[Bar], interval: str, continuous: bool,
    n_states: int = 3,
) -> tuple[str | None, Decimal, float]:
    """Fit and filter a regime model, returning (label, risk_score, confidence).

    Returns ``(None, 0, 0)`` -- explicitly benign -- when the regime cannot be
    assessed: ``hmmlearn`` absent (the Windows dev case), or too few bars to fit.
    A missing regime must not block the loop, and defaulting to benign matches
    the ``Mandate`` and ``open_mandate`` conventions elsewhere. It is the
    optimistic default, so it is logged rather than silent.
    """
    from spintrader.quant.regime import RegimeError, RegimeModel

    try:
        features = build_features(bars, interval=interval, continuous=continuous)
    except ValueError as exc:
        log.info("regime: not enough bars to build features (%s)", exc)
        return None, ZERO, 0.0

    try:
        model = RegimeModel(n_states=n_states).fit(features)
        state = model.filter_latest(features)
    except RegimeError as exc:
        # Most commonly hmmlearn is not installed; treat as benign but visible.
        log.info("regime: unavailable (%s); defaulting to benign", exc)
        return None, ZERO, 0.0

    return state.label, to_decimal(state.risk_score), state.confidence


def build_context(
    instrument: Instrument,
    bars: Sequence[Bar],
    interval: str,
    horizon: Horizon,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
    vol_window: int = 20,
    continuous: bool = True,
    with_regime: bool = False,
) -> MarketContext:
    """Assemble a :class:`MarketContext` from a trailing bar window."""
    bars = list(bars)
    ctx = MarketContext(
        instrument=instrument, interval=interval, horizon=horizon, bars=bars,
    )
    if not bars:
        return ctx

    ctx.last_close = bars[-1].close
    ctx.trend_strength, ctx.annual_vol = _trend_and_vol(
        bars, fast_window, slow_window, vol_window, interval, continuous,
    )
    if with_regime:
        ctx.regime_label, ctx.regime_risk, ctx.regime_confidence = assess_regime(
            bars, interval, continuous,
        )
    return ctx


def gather_contexts(
    store,
    instruments: Sequence[Instrument],
    interval: str,
    horizon: Horizon,
    *,
    lookback: int = 1_000,
    continuous: bool = True,
    with_regime: bool = False,
    fast_window: int = 20,
    slow_window: int = 50,
    vol_window: int = 20,
) -> dict[str, MarketContext]:
    """Read the trailing window for each instrument and build its context."""
    out: dict[str, MarketContext] = {}
    for instrument in instruments:
        bars = store.read_bars(instrument.key, interval, limit=lookback)
        out[instrument.key] = build_context(
            instrument, bars, interval, horizon,
            fast_window=fast_window, slow_window=slow_window,
            vol_window=vol_window, continuous=continuous, with_regime=with_regime,
        )
    return out


__all__ = [
    "MarketContext", "assess_regime", "build_context", "gather_contexts",
]
