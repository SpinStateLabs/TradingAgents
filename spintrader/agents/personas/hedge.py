"""Game-theory persona -- a no-regret (Hedge) ensemble over expert signals.

Trading is a repeated game against an adversarial market. The classic
game-theoretic answer is a *no-regret* algorithm: keep several experts, and
reweight them by how they have actually paid off, so that over time you do nearly
as well as the best expert in hindsight -- Freund & Schapire's Hedge / the
multiplicative-weights algorithm, the same machinery behind the minimax theorem
and Hart & Mas-Colell's regret matching.

Here the experts are three deterministic directional signals that disagree by
construction -- momentum, reversion and breakout -- and the meta-learner is
Hedge: each bar, every expert's weight is multiplied by ``exp(eta * reward)``
where the reward is last bar's vote times the realised return. The ensemble goes
long when the weight-blended vote is bullish enough. Because the whole weight
path is replayed from the trailing window each bar, the persona is causal and
carries no hidden state across the backtester's fold resets.

Long-only, like the others: it acts on a positive blended vote and exits when the
ensemble turns or a stop fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from spintrader.core.types import Bar, Instrument, Side, to_decimal
from spintrader.quant.features import log_returns, periods_per_year, rolling_mean, rolling_std
from spintrader.risk.engine import Mandate, TradeIntent

ZERO = Decimal("0")
ONE = Decimal("1")

EXPERTS = ("momentum", "reversion", "breakout")


@dataclass(slots=True)
class HedgeReading:
    close: float
    weights: tuple[float, ...]      # current Hedge weights over the experts
    votes: tuple[float, ...]        # current expert votes, in [-1, 1]
    blended: float                  # weight-blended vote, in [-1, 1]
    annual_vol: float
    vol_ok: bool
    bullish: bool
    edge: float
    confidence: float


class HedgeEnsembleAgent:
    """No-regret ensemble: Hedge-weighted momentum / reversion / breakout."""

    name = "hedge_ensemble_v1"

    def __init__(
        self,
        lookback: int = 200,
        eta: Decimal | str | float = "2.0",         # Hedge learning rate
        fast_window: int = 10,
        slow_window: int = 30,
        z_window: int = 20,
        breakout_window: int = 20,
        entry_threshold: Decimal | str | float = "0.15",
        exit_threshold: Decimal | str | float = "0.0",
        vol_ceiling: Decimal | str | float = "0.40",
        stop_pct: Decimal | str | float = "0.05",
        trail_pct: Decimal | str | float = "0.08",
        edge_floor: Decimal | str | float = "0.005",
        edge_ceiling: Decimal | str | float = "0.06",
        interval: str = "1d",
        continuous: bool = False,
        min_annual_vol: Decimal | str | float = "0.01",
        spread_bps: Decimal | str | float = "5",
        name: str | None = None,
    ) -> None:
        if slow_window <= fast_window:
            raise ValueError("slow_window must exceed fast_window")
        self.lookback = lookback
        self.eta = float(to_decimal(eta))
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.z_window = z_window
        self.breakout_window = breakout_window
        self.entry_threshold = float(to_decimal(entry_threshold))
        self.exit_threshold = float(to_decimal(exit_threshold))
        self.vol_ceiling = to_decimal(vol_ceiling)
        self.stop_pct = to_decimal(stop_pct)
        self.trail_pct = to_decimal(trail_pct)
        self.edge_floor = to_decimal(edge_floor)
        self.edge_ceiling = to_decimal(edge_ceiling)
        self.interval = interval
        self.continuous = continuous
        self.min_annual_vol = to_decimal(min_annual_vol)
        self.spread_bps = to_decimal(spread_bps)
        if name:
            self.name = name

        self._ann_scale = math.sqrt(periods_per_year(interval, continuous=continuous))
        self._long = False
        self._entry_mark: float | None = None
        self._peak_mark: float | None = None
        self._last_reading: HedgeReading | None = None

    # -- introspection -----------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self.lookback + self.slow_window + 2

    @property
    def is_long(self) -> bool:
        return self._long

    @property
    def last_reading(self) -> HedgeReading | None:
        return self._last_reading

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name, "lookback": self.lookback, "eta": str(self.eta),
            "experts": list(EXPERTS), "entry_threshold": str(self.entry_threshold),
            "interval": self.interval, "continuous": self.continuous,
            "warmup_bars": self.warmup_bars,
        }

    # -- strategy protocol -------------------------------------------------

    def fit(self, bars: Sequence[Bar]) -> None:
        self.reset()

    def reset(self) -> None:
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

    # -- expert votes (each causal, in [-1, 1]) ----------------------------

    def _expert_votes(self, closes: np.ndarray) -> np.ndarray:
        """A (n_bars, n_experts) matrix of causal votes in [-1, 1]."""
        n = closes.size
        fast = rolling_mean(closes, self.fast_window)
        slow = rolling_mean(closes, self.slow_window)
        momentum = np.sign(fast - slow)

        returns = log_returns(closes)
        mean = rolling_mean(closes, self.z_window)
        std = rolling_std(closes, self.z_window)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(std > 1e-12, (closes - mean) / std, 0.0)
        reversion = np.clip(-z / 2.0, -1.0, 1.0)      # buy dips, fade rallies

        # Breakout: +1 above the prior window's high, -1 below its low. Vectorised
        # over the full-window region via a sliding view (exact same max/min over
        # the same strictly-prior windows), with the short warm-up done directly.
        breakout = np.zeros(n)
        w = self.breakout_window
        if n > w:
            prior = sliding_window_view(closes, w)[:n - w]     # prior[i-w] = closes[i-w:i]
            cur = closes[w:n]
            breakout[w:n] = np.where(cur > prior.max(axis=1), 1.0,
                                     np.where(cur < prior.min(axis=1), -1.0, 0.0))
        for i in range(1, min(w, n)):                          # warm-up: expanding prior
            seg = closes[:i]
            breakout[i] = 1.0 if closes[i] > seg.max() else (-1.0 if closes[i] < seg.min() else 0.0)
        return np.column_stack([momentum, reversion, breakout])

    def _hedge_weights(self, votes: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Replay Hedge over the window; return the final expert weights.

        Reward for expert j at bar i is ``votes[i-1, j] * returns[i]`` -- last
        bar's vote judged by this bar's realised return. Weights are updated
        multiplicatively, so an expert that has been paying off gains influence.
        """
        n_experts = votes.shape[1]
        if votes.shape[0] < 2:
            return np.full(n_experts, 1.0 / n_experts)
        # The sequential multiplicative update is, after normalisation, exactly a
        # softmax over each expert's cumulative reward -- the per-step max
        # subtraction that kept the loop bounded cancels in the final ratio. So
        # replace the O(window) Python loop with one vectorised dot product.
        total_reward = (votes[:-1] * returns[1:, None]).sum(axis=0)
        log_w = self.eta * total_reward
        log_w -= log_w.max()
        w = np.exp(log_w)
        total = w.sum()
        return w / total if total > 0 else np.full(n_experts, 1.0 / n_experts)

    def _read(self, history: Sequence[Bar]) -> HedgeReading:
        closes = np.array([float(b.close) for b in history], dtype=float)[-self.lookback - self.slow_window:]
        close = float(closes[-1])
        returns = log_returns(closes)

        votes = self._expert_votes(closes)
        weights = self._hedge_weights(votes, returns)
        current_votes = votes[-1]
        blended = float(np.dot(weights, current_votes))

        annual_vol = float(rolling_std(returns, self.z_window)[-1] * self._ann_scale)
        vol_ok = annual_vol <= float(self.vol_ceiling)
        bullish = blended >= self.entry_threshold and vol_ok

        edge = self._edge(blended, returns)
        confidence = self._confidence(blended, weights, annual_vol)

        if self._long and self._peak_mark is not None:
            self._peak_mark = max(self._peak_mark, close)

        return HedgeReading(
            close=close, weights=tuple(float(x) for x in weights),
            votes=tuple(float(x) for x in current_votes), blended=blended,
            annual_vol=annual_vol, vol_ok=vol_ok, bullish=bullish,
            edge=edge, confidence=confidence,
        )

    def _edge(self, blended: float, returns: np.ndarray) -> float:
        # Blended directional conviction times a typical bar move -- an honest
        # expected-return proxy -- then clipped like the other personas.
        typical = float(np.abs(returns[-self.z_window:]).mean()) if returns.size else 0.0
        floor, ceiling = float(self.edge_floor), float(self.edge_ceiling)
        return min(ceiling, max(floor, blended * typical * self.z_window))

    def _confidence(self, blended: float, weights: np.ndarray, annual_vol: float) -> float:
        conviction = min(1.0, abs(blended))
        # Concentration: how decisively the meta-learner has picked an expert.
        concentration = float(weights.max())
        vol_span = float(self.vol_ceiling)
        vol_score = min(1.0, max(0.0, (vol_span - annual_vol) / vol_span)) if vol_span > 0 else 0.0
        blended_score = 0.5 * conviction + 0.3 * concentration + 0.2 * vol_score
        return min(1.0, max(0.0, 0.60 + 0.35 * blended_score))

    def _exit_reason(self, reading: HedgeReading) -> str | None:
        if reading.blended <= self.exit_threshold:
            return "ensemble_turned"
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


__all__ = ["HedgeEnsembleAgent", "HedgeReading", "EXPERTS"]
