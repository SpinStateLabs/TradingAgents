"""Markov regime-switching persona -- trade only when the regime is risk-on.

The system already fits a Gaussian HMM to decide *how much* to risk (it scales
exposure via the mandate's regime_risk). This persona uses the same latent-state
machinery to decide *whether* to be in at all: it is long only while the filtered
regime is calm and the drift is positive, and it steps aside when the regime
turns stressed. Where the trend and reversion personas read the price level, this
reads the hidden state generating it.

It reuses :mod:`spintrader.quant.regime`, which matters for two reasons:

* **The state inference is causal.** ``RegimeModel.filter`` runs the forward
  algorithm only, so the state at t is conditioned on data up to t -- never the
  smoothed (whole-sequence) states, which would be lookahead.
* **The model is refit, not fit-once.** Markets are non-stationary; the model is
  refit every ``refit_interval`` bars on the trailing window so the regimes track
  the market rather than a stale segmentation.

Without ``hmmlearn`` (the Windows dev case) the model cannot fit, so the persona
degrades to doing nothing -- no trades, no error -- exactly as the regime layer
does elsewhere.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

import numpy as np

from spintrader.core.types import Bar, Instrument, Side, to_decimal
from spintrader.quant.features import build_features, log_returns, periods_per_year, rolling_std
from spintrader.risk.engine import Mandate, TradeIntent

log = logging.getLogger(__name__)

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class RegimeSwitchReading:
    close: float
    risk_score: float           # 0 calm .. 1 crisis (probability-weighted)
    regime_confidence: float
    drift: float                # trailing mean log return
    annual_vol: float
    vol_ok: bool
    risk_on: bool
    bullish: bool
    edge: float
    confidence: float
    fitted: bool                # False when hmmlearn is unavailable / fit failed


class RegimeSwitchingAgent:
    """Long-only, in the market only while the HMM regime is calm and drifting up."""

    name = "regime_switch_v1"

    def __init__(
        self,
        n_states: int = 3,
        fit_window: int = 500,
        refit_interval: int = 250,
        recompute_interval: int = 10,
        drift_window: int = 30,
        risk_on: Decimal | str | float = "0.40",
        risk_off: Decimal | str | float = "0.60",
        vol_ceiling: Decimal | str | float = "0.50",
        stop_pct: Decimal | str | float = "0.05",
        trail_pct: Decimal | str | float = "0.08",
        edge_floor: Decimal | str | float = "0.005",
        edge_ceiling: Decimal | str | float = "0.06",
        interval: str = "1d",
        continuous: bool = False,
        min_annual_vol: Decimal | str | float = "0.01",
        n_restarts: int = 3,
        spread_bps: Decimal | str | float = "5",
        name: str | None = None,
    ) -> None:
        self.n_states = n_states
        self.fit_window = fit_window
        self.refit_interval = max(1, refit_interval)
        self.recompute_interval = max(1, recompute_interval)
        self.drift_window = drift_window
        self.risk_on = float(to_decimal(risk_on))
        self.risk_off = float(to_decimal(risk_off))
        self.vol_ceiling = to_decimal(vol_ceiling)
        self.stop_pct = to_decimal(stop_pct)
        self.trail_pct = to_decimal(trail_pct)
        self.edge_floor = to_decimal(edge_floor)
        self.edge_ceiling = to_decimal(edge_ceiling)
        self.interval = interval
        self.continuous = continuous
        self.min_annual_vol = to_decimal(min_annual_vol)
        self.n_restarts = n_restarts
        self.spread_bps = to_decimal(spread_bps)
        if name:
            self.name = name

        self._ann_scale = math.sqrt(periods_per_year(interval, continuous=continuous))
        self._model = None
        self._bars_since_fit = 0
        self._regime_unavailable = False
        self._regime_countdown = 0
        self._cached_regime: tuple[float, float] | None = None
        self._long = False
        self._entry_mark: float | None = None
        self._peak_mark: float | None = None
        self._last_reading: RegimeSwitchReading | None = None

    # -- introspection -----------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        # Enough to build features (drops ~50 warm-up rows) and fit the HMM.
        return self.fit_window + 60

    @property
    def is_long(self) -> bool:
        return self._long

    @property
    def last_reading(self) -> RegimeSwitchReading | None:
        return self._last_reading

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name, "n_states": self.n_states,
            "fit_window": self.fit_window, "refit_interval": self.refit_interval,
            "risk_on": str(self.risk_on), "risk_off": str(self.risk_off),
            "interval": self.interval, "continuous": self.continuous,
            "warmup_bars": self.warmup_bars,
        }

    # -- strategy protocol -------------------------------------------------

    def fit(self, bars: Sequence[Bar]) -> None:
        self.reset()

    def reset(self) -> None:
        self._model = None
        self._bars_since_fit = 0
        self._regime_unavailable = False
        self._regime_countdown = 0
        self._cached_regime = None
        self._long = False
        self._entry_mark = None
        self._peak_mark = None
        self._last_reading = None

    def on_bar(self, cursor, instrument: Instrument, mandate: Mandate) -> Sequence[TradeIntent]:
        history = cursor.history(self.warmup_bars)
        if len(history) < self.warmup_bars:
            return ()

        reading = self._read(history)
        self._last_reading = reading
        if not reading.fitted:
            return ()            # no regime model -> stand aside, quietly

        if self._long:                       # exits first (L1)
            reason = self._exit_reason(reading)
            if reason is None:
                return ()
            self._long = False
            self._entry_mark = None
            self._peak_mark = None
            return (self._exit_intent(
                instrument, cursor,
                max(to_decimal(reading.annual_vol), self.min_annual_vol), reason,
            ),)

        if reading.annual_vol <= float(self.min_annual_vol):
            return ()
        if not reading.bullish:
            return ()
        if not mandate.allows(instrument.key):
            return ()
        if mandate.bias_for(instrument.key) < ZERO:
            return ()

        self._long = True
        self._entry_mark = reading.close
        self._peak_mark = reading.close
        return (self._entry_intent(instrument, cursor, reading,
                                   to_decimal(reading.annual_vol)),)

    # -- signal ------------------------------------------------------------

    def _ensure_model(self, bars: Sequence[Bar]):
        """Fit (or refit) the regime model on the trailing window; None if unavailable."""
        if self._regime_unavailable:
            return None
        need_fit = self._model is None or self._bars_since_fit >= self.refit_interval
        if not need_fit:
            self._bars_since_fit += 1
            return self._model

        from spintrader.quant.regime import RegimeError, RegimeModel
        try:
            features = build_features(
                bars[-self.fit_window:], interval=self.interval,
                continuous=self.continuous,
            )
            model = RegimeModel(n_states=self.n_states, n_restarts=self.n_restarts).fit(features)
        except (ValueError, RegimeError) as exc:
            # hmmlearn absent, or too few bars: stand aside for the whole run.
            log.info("regime_switch: model unavailable (%s); standing aside", exc)
            self._regime_unavailable = True
            self._model = None
            return None
        self._model = model
        self._bars_since_fit = 0
        return model

    def _read(self, history: Sequence[Bar]) -> RegimeSwitchReading:
        closes = np.array([float(b.close) for b in history], dtype=float)
        close = float(closes[-1])
        returns = log_returns(closes)
        annual_vol = float(rolling_std(returns, self.drift_window)[-1] * self._ann_scale)
        drift = float(returns[-self.drift_window:].mean())
        vol_ok = annual_vol <= float(self.vol_ceiling)

        model = self._ensure_model(history)
        if model is None:
            return RegimeSwitchReading(
                close=close, risk_score=0.0, regime_confidence=0.0, drift=drift,
                annual_vol=annual_vol, vol_ok=vol_ok, risk_on=False, bullish=False,
                edge=0.0, confidence=0.0, fitted=False,
            )

        # Re-infer the regime only every `recompute_interval` bars, reusing the
        # last inference in between. Regimes are sticky by construction (the HMM
        # rejects states lasting under a few bars), so this turns a per-bar
        # forward pass + feature build into an occasional one -- the difference
        # between a usable and an unusable 1m sweep (lessons L12). The drift and
        # volatility that gate the entry are still recomputed every bar.
        if self._cached_regime is None or self._regime_countdown <= 0:
            from spintrader.quant.regime import RegimeError
            try:
                features = build_features(
                    history[-self.fit_window:], interval=self.interval,
                    continuous=self.continuous,
                )
                state = model.filter_latest(features)
                self._cached_regime = (float(state.risk_score), float(state.confidence))
                self._regime_countdown = self.recompute_interval
            except (ValueError, RegimeError):
                self._cached_regime = None
                return RegimeSwitchReading(
                    close=close, risk_score=0.0, regime_confidence=0.0, drift=drift,
                    annual_vol=annual_vol, vol_ok=vol_ok, risk_on=False, bullish=False,
                    edge=0.0, confidence=0.0, fitted=False,
                )
        else:
            self._regime_countdown -= 1
        risk_score, regime_confidence = self._cached_regime

        risk_on = risk_score <= self.risk_on
        bullish = risk_on and drift > 0 and vol_ok
        edge = self._edge(drift)
        confidence = self._confidence(risk_score, regime_confidence, annual_vol)

        if self._long and self._peak_mark is not None:
            self._peak_mark = max(self._peak_mark, close)

        return RegimeSwitchReading(
            close=close, risk_score=risk_score, regime_confidence=regime_confidence,
            drift=drift, annual_vol=annual_vol, vol_ok=vol_ok, risk_on=risk_on,
            bullish=bullish, edge=edge, confidence=confidence, fitted=True,
        )

    def _edge(self, drift: float) -> float:
        # Annualise the per-bar drift into an expected forward return proxy, then
        # clip -- the same honest treatment the other personas apply to a weak
        # return forecast.
        floor, ceiling = float(self.edge_floor), float(self.edge_ceiling)
        return min(ceiling, max(floor, drift * self.drift_window))

    def _confidence(self, risk_score: float, regime_confidence: float, annual_vol: float) -> float:
        calm = max(0.0, 1.0 - risk_score / self.risk_on) if self.risk_on > 0 else 0.0
        vol_span = float(self.vol_ceiling)
        vol_score = min(1.0, max(0.0, (vol_span - annual_vol) / vol_span)) if vol_span > 0 else 0.0
        blended = 0.5 * min(1.0, calm) + 0.3 * regime_confidence + 0.2 * vol_score
        return min(1.0, max(0.0, 0.60 + 0.35 * blended))

    def _exit_reason(self, reading: RegimeSwitchReading) -> str | None:
        if reading.risk_score >= self.risk_off:
            return "regime_stressed"
        if reading.drift <= 0:
            return "drift_gone"
        if not reading.vol_ok:
            return "vol_spike"
        if self._entry_mark is not None and self._entry_mark > 0:
            if reading.close <= self._entry_mark * (1.0 - float(self.stop_pct)):
                return "hard_stop"
        if self._peak_mark is not None and self._peak_mark > 0:
            if reading.close <= self._peak_mark * (1.0 - float(self.trail_pct)):
                return "trailing_stop"
        return None

    # -- intents -----------------------------------------------------------

    def _entry_intent(self, instrument, cursor, reading, volatility) -> TradeIntent:
        return TradeIntent(
            instrument=instrument, side=Side.BUY, edge=to_decimal(reading.edge),
            confidence=to_decimal(reading.confidence), volatility=volatility,
            quote=cursor.quote(self.spread_bps), strategy=self.name,
        )

    def _exit_intent(self, instrument, cursor, volatility, reason) -> TradeIntent:
        defensive_edge = max(self.edge_ceiling, volatility * volatility * 4)
        return TradeIntent(
            instrument=instrument, side=Side.SELL, edge=defensive_edge,
            confidence=ONE, volatility=volatility,
            quote=cursor.quote(self.spread_bps), strategy=f"{self.name}:{reason}",
        )


__all__ = ["RegimeSwitchingAgent", "RegimeSwitchReading"]
