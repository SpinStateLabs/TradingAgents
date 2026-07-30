"""High-order Markov-chain persona -- forecast the next bar from recent symbols.

A different edge again: where trend extrapolates and reversion fades, this asks a
narrower empirical question -- *given the last k return-symbols, what has tended
to happen next?* -- and forecasts the expected next-bar return as the
probability-weighted mean over the observed next states. Order 1 is a plain
Markov chain on discretised returns; higher orders condition on the last k
symbols, which is where short-horizon microstructure structure, if any, lives.

Honest by construction:

* **Causal.** The transition table is built only from pairs observed within the
  trailing window; the current k-gram's *next* symbol is the unknown it
  forecasts, never a pair it trained on.
* **It forecasts a return, not a label.** The edge is E[r_next | k-gram] =
  sum_s P(s | k-gram) * mean_return_in_state_s -- a real expected return the risk
  engine can size, not a bare up/down call.
* **It backs off when it has never seen the k-gram.** Rather than inventing a
  probability from no data, it falls to order k-1, ..., 0 (the base rate), and
  refuses to trade a forecast with too little support.

Long-only, like the other personas: it acts on a positive expected return and
exits when that decays or a stop fires.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Sequence

import numpy as np

from spintrader.core.types import Bar, Instrument, Side, to_decimal
from spintrader.quant.features import log_returns, periods_per_year, rolling_std
from spintrader.risk.engine import Mandate, TradeIntent

ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(slots=True)
class MarkovReading:
    close: float
    p_up: float                 # P(next in the top state)
    expected_return: float      # E[r_next | current k-gram]
    support: int                # observations behind the forecast
    order_used: int             # the order the forecast actually used after backoff
    annual_vol: float
    vol_ok: bool
    bullish: bool
    edge: float
    confidence: float


class HighOrderMarkovAgent:
    """Order-k Markov chain over discretised returns, forecasting the next bar."""

    name = "markov_chain_v1"

    def __init__(
        self,
        order: int = 1,
        n_states: int = 3,
        lookback: int = 200,
        dead_zone: Decimal | str | float = "0.25",   # flat band, in std of returns
        min_support: int = 8,
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
        if order < 1:
            raise ValueError("order must be at least 1")
        if n_states < 2:
            raise ValueError("n_states must be at least 2")
        if lookback < n_states ** (order + 1) * 4:
            # Too little data to populate the table meaningfully; not fatal, but
            # the caller should know the grid is under-powered.
            pass

        self.order = order
        self.n_states = n_states
        self.lookback = lookback
        self.dead_zone = float(to_decimal(dead_zone))
        self.min_support = min_support
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
        self._last_reading: MarkovReading | None = None

    # -- introspection -----------------------------------------------------

    @property
    def warmup_bars(self) -> int:
        return self.lookback + self.order + 2

    @property
    def is_long(self) -> bool:
        return self._long

    @property
    def last_reading(self) -> MarkovReading | None:
        return self._last_reading

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name, "order": self.order, "n_states": self.n_states,
            "lookback": self.lookback, "dead_zone": str(self.dead_zone),
            "min_support": self.min_support, "interval": self.interval,
            "continuous": self.continuous, "warmup_bars": self.warmup_bars,
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

    # -- signal ------------------------------------------------------------

    def _symbolise(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Discretise returns into states and record each state's mean return."""
        std = float(returns.std(ddof=1)) if returns.size > 1 else 0.0
        thr = self.dead_zone * std
        if self.n_states == 2:
            symbols = (returns > 0).astype(int)          # 0 down, 1 up
        else:
            # 0 down, 1 flat, 2 up (a symmetric dead zone; middle states share it)
            symbols = np.ones(returns.size, dtype=int)
            symbols[returns > thr] = self.n_states - 1
            symbols[returns < -thr] = 0
            if self.n_states > 3:
                # Spread the interior states across the flat band by quantile.
                interior = (symbols == 1)
                if interior.sum() > 0:
                    band = returns[interior]
                    edges = np.quantile(band, np.linspace(0, 1, self.n_states - 1))
                    symbols[interior] = 1 + np.clip(
                        np.searchsorted(edges, band, side="right") - 1,
                        0, self.n_states - 3,
                    )
        state_mean = np.zeros(self.n_states)
        for s in range(self.n_states):
            mask = symbols == s
            state_mean[s] = float(returns[mask].mean()) if mask.any() else 0.0
        return symbols, state_mean

    def _forecast(self, symbols: np.ndarray) -> tuple[np.ndarray, int, int]:
        """P(next state) for the current k-gram, backing off on thin support."""
        current = tuple(symbols[-self.order:])
        for k in range(self.order, -1, -1):
            counts = np.zeros(self.n_states)
            gram = current[self.order - k:] if k > 0 else ()
            # Count observed (k-gram -> next) pairs within the window.
            for i in range(k, symbols.size):
                if k == 0 or tuple(symbols[i - k:i]) == gram:
                    counts[symbols[i]] += 1
            support = int(counts.sum())
            if support >= self.min_support:
                return counts / support, support, k
        # Nothing met support even at order 0 (degenerate window).
        return np.full(self.n_states, 1.0 / self.n_states), 0, 0

    def _read(self, history: Sequence[Bar]) -> MarkovReading:
        closes = np.array([float(b.close) for b in history], dtype=float)
        close = float(closes[-1])
        returns = log_returns(closes)[-self.lookback:]

        symbols, state_mean = self._symbolise(returns)
        probs, support, order_used = self._forecast(symbols)

        expected_return = float(np.dot(probs, state_mean))
        p_up = float(probs[-1])

        annual_vol = float(rolling_std(log_returns(closes), self.lookback)[-1] * self._ann_scale)
        vol_ok = annual_vol <= float(self.vol_ceiling)
        bullish = (
            expected_return > 0 and support >= self.min_support and vol_ok
        )

        edge = self._edge(expected_return)
        confidence = self._confidence(p_up, support, annual_vol)

        if self._long and self._peak_mark is not None:
            self._peak_mark = max(self._peak_mark, close)

        return MarkovReading(
            close=close, p_up=p_up, expected_return=expected_return,
            support=support, order_used=order_used, annual_vol=annual_vol,
            vol_ok=vol_ok, bullish=bullish, edge=edge, confidence=confidence,
        )

    def _edge(self, expected_return: float) -> float:
        floor, ceiling = float(self.edge_floor), float(self.edge_ceiling)
        return min(ceiling, max(floor, expected_return))

    def _confidence(self, p_up: float, support: int, annual_vol: float) -> float:
        conviction = min(1.0, abs(2.0 * p_up - 1.0) * 2.0)
        evidence = min(1.0, support / 30.0)
        vol_span = float(self.vol_ceiling)
        vol_score = min(1.0, max(0.0, (vol_span - annual_vol) / vol_span)) if vol_span > 0 else 0.0
        blended = 0.5 * conviction + 0.3 * evidence + 0.2 * vol_score
        return min(1.0, max(0.0, 0.60 + 0.35 * blended))

    def _exit_reason(self, reading: MarkovReading) -> str | None:
        if reading.expected_return <= 0:
            return "forecast_decayed"
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


__all__ = ["HighOrderMarkovAgent", "MarkovReading"]
